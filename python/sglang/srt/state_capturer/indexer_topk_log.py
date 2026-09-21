"""Decode-only indexer top-k log for offline KV locality analysis.

File layout (little endian): ``MAGIC``, u32 header length, JSON header, then one
record per decode forward: ``<QII`` (forward counter, batch size, payload bytes),
``batch`` x ``<QI`` (request id hash, sequence length), zlib payload. The payload
is int32 ``[batch, num_layers, topk]``, each row sorted and delta-encoded.
"""

import json
import logging
import os
import queue
import struct
import threading
import zlib
from typing import BinaryIO, Iterator, List, Optional

import msgspec
import numpy as np
import torch

from sglang.srt.configs.model_config import ModelConfig, dsa_layer_skips_topk
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_memory, get_parallel

logger = logging.getLogger(__name__)

MAGIC = b"SGLTOPK1"
_STEP_HEADER = struct.Struct("<QII")
_REQ_HEADER = struct.Struct("<QI")
# Arbitrary; bounds pinned staging memory while absorbing writer-thread jitter.
_NUM_STAGING_SLOTS = 4


class IndexerTopkLogHeader(msgspec.Struct, frozen=True):
    version: int
    topk: int
    num_model_layers: int
    # Layers that compute a fresh top-k; the layers between two entries reuse it.
    layer_ids: List[int]
    # False when the fused top-k already mapped positions to KV slot ids.
    ids_are_token_positions: bool
    model_path: str
    rank: int


class IndexerTopkLogStep(msgspec.Struct, frozen=True):
    forward_count: int
    rid_hashes: np.ndarray
    seq_lens: np.ndarray
    topk: np.ndarray


def hash_rid(rid: str) -> int:
    # Stable across processes, unlike hash().
    return zlib.crc32(rid.encode("utf-8")) | (zlib.adler32(rid.encode("utf-8")) << 32)


def encode_topk(topk: np.ndarray) -> bytes:
    ordered = np.sort(topk.astype(np.int32, copy=False), axis=-1)
    deltas = np.diff(ordered, axis=-1, prepend=np.int32(0))
    return zlib.compress(np.ascontiguousarray(deltas).tobytes(), 1)


def decode_topk(
    payload: bytes, *, batch: int, num_layers: int, topk: int
) -> np.ndarray:
    deltas = np.frombuffer(zlib.decompress(payload), dtype=np.int32)
    return np.cumsum(deltas.reshape(batch, num_layers, topk), axis=-1, dtype=np.int32)


def write_header(file: BinaryIO, header: IndexerTopkLogHeader) -> None:
    body = msgspec.json.encode(header)
    file.write(MAGIC + struct.pack("<I", len(body)) + body)


def write_step(
    file: BinaryIO,
    *,
    forward_count: int,
    rid_hashes: List[int],
    seq_lens: np.ndarray,
    topk: np.ndarray,
) -> None:
    payload = encode_topk(topk)
    file.write(_STEP_HEADER.pack(forward_count, len(rid_hashes), len(payload)))
    for rid_hash, seq_len in zip(rid_hashes, seq_lens.tolist()):
        file.write(_REQ_HEADER.pack(rid_hash, seq_len))
    file.write(payload)


def read_header(file: BinaryIO) -> IndexerTopkLogHeader:
    if file.read(len(MAGIC)) != MAGIC:
        raise ValueError("not an indexer top-k log")
    (length,) = struct.unpack("<I", file.read(4))
    return msgspec.json.decode(file.read(length), type=IndexerTopkLogHeader)


def read_steps(
    file: BinaryIO, header: IndexerTopkLogHeader
) -> Iterator[IndexerTopkLogStep]:
    while len(raw := file.read(_STEP_HEADER.size)) == _STEP_HEADER.size:
        forward_count, batch, payload_len = _STEP_HEADER.unpack(raw)
        reqs = np.frombuffer(
            file.read(_REQ_HEADER.size * batch), dtype=[("rid", "<u8"), ("len", "<u4")]
        )
        yield IndexerTopkLogStep(
            forward_count=forward_count,
            rid_hashes=reqs["rid"],
            seq_lens=reqs["len"],
            topk=decode_topk(
                file.read(payload_len),
                batch=batch,
                num_layers=len(header.layer_ids),
                topk=header.topk,
            ),
        )


class IndexerTopkLogCapturer:
    """Drop-in for IndexerTopkCapturer that streams decode steps to disk instead
    of retaining every token's top-k in host memory."""

    def __init__(
        self,
        *,
        log_dir: str,
        header: IndexerTopkLogHeader,
        num_model_layers: int,
        max_batch_size: int,
        device: str,
    ):
        self.header = header
        self.device = device
        num_layers = len(header.layer_ids)
        self._slot_of_layer = [-1] * num_model_layers
        for slot, layer_id in enumerate(header.layer_ids):
            self._slot_of_layer[layer_id] = slot
        self._device_buffer = torch.zeros(
            (max_batch_size, num_layers, header.topk), dtype=torch.int32, device=device
        )
        pin = device == "cuda"
        self._free_slots: queue.Queue = queue.Queue()
        for _ in range(_NUM_STAGING_SLOTS):
            self._free_slots.put(
                (
                    torch.zeros(
                        (max_batch_size, num_layers, header.topk),
                        dtype=torch.int32,
                        pin_memory=pin,
                    ),
                    torch.zeros((max_batch_size,), dtype=torch.int64, pin_memory=pin),
                )
            )
        os.makedirs(log_dir, exist_ok=True)
        stem = os.path.join(log_dir, f"indexer_topk_rank{header.rank}")
        self._file = open(stem + ".bin", "wb")
        self._rid_file = open(stem + ".rids.jsonl", "w")
        write_header(self._file, header)
        self._seen_rids: set = set()
        self._forward_count = 0
        self._pending: queue.Queue = queue.Queue()
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()

    def capture(self, layer_id: int, topk_indices: torch.Tensor) -> None:
        slot = self._slot_of_layer[layer_id]
        if slot < 0:
            return
        # Prefill batches carry one row per token and can exceed the buffer;
        # only decode rows are ever logged, so truncating them is harmless.
        rows = min(topk_indices.shape[0], self._device_buffer.shape[0])
        self._device_buffer[:rows, slot, :] = topk_indices[:rows]

    def get_topk(self, **_kwargs) -> None:
        return None

    def on_forward_end(self, *, forward_batch: ForwardBatch, **_kwargs) -> None:
        if not forward_batch.forward_mode.is_decode() or not forward_batch.rids:
            return None
        batch = len(forward_batch.rids)
        # Blocks when the writer falls behind: back-pressure instead of data loss.
        staged_topk, staged_lens = self._free_slots.get()
        staged_topk[:batch].copy_(self._device_buffer[:batch], non_blocking=True)
        staged_lens[:batch].copy_(forward_batch.seq_lens[:batch], non_blocking=True)
        event = None
        if self.device == "cuda":
            event = torch.cuda.Event()
            event.record()
        self._pending.put(
            (
                event,
                staged_topk,
                staged_lens,
                list(forward_batch.rids),
                self._forward_count,
            )
        )
        self._forward_count += 1
        return None

    def destroy(self) -> None:
        self._pending.put(None)
        self._writer.join()
        self._file.close()
        self._rid_file.close()

    def _write_loop(self) -> None:
        while (item := self._pending.get()) is not None:
            event, staged_topk, staged_lens, rids, forward_count = item
            if event is not None:
                event.synchronize()
            batch = len(rids)
            self._record_rids(rids)
            write_step(
                self._file,
                forward_count=forward_count,
                rid_hashes=[hash_rid(rid) for rid in rids],
                seq_lens=staged_lens[:batch].numpy(),
                topk=staged_topk[:batch].numpy(),
            )
            self._free_slots.put((staged_topk, staged_lens))
        self._file.flush()
        self._rid_file.flush()

    def _record_rids(self, rids: List[str]) -> None:
        for rid in rids:
            if rid not in self._seen_rids:
                self._seen_rids.add(rid)
                self._rid_file.write(json.dumps({"hash": hash_rid(rid), "rid": rid}))
                self._rid_file.write("\n")


def create_indexer_topk_log_capturer(
    *, model_config: ModelConfig, max_running_requests: int, device: str
) -> Optional[IndexerTopkLogCapturer]:
    log_dir = envs.SGLANG_INDEXER_TOPK_LOG_DIR.get()
    # Attention-TP peers see identical batches; one copy per DP rank is enough.
    if not log_dir or get_parallel().attn_tp_rank != 0:
        return None
    hf_text_config = model_config.hf_text_config
    num_model_layers = hf_text_config.num_hidden_layers
    ids_are_token_positions = (
        not envs.SGLANG_DSA_FUSE_TOPK.get() or get_memory().enable_hisparse
    )
    if not ids_are_token_positions:
        logger.warning(
            "Indexer top-k log will hold KV slot ids, not token positions; "
            "set SGLANG_DSA_FUSE_TOPK=0 for position-based locality analysis."
        )
    header = IndexerTopkLogHeader(
        version=1,
        topk=hf_text_config.index_topk,
        num_model_layers=num_model_layers,
        layer_ids=[
            layer_id
            for layer_id in range(num_model_layers)
            if not dsa_layer_skips_topk(hf_text_config, layer_id)
        ],
        ids_are_token_positions=ids_are_token_positions,
        model_path=model_config.model_path,
        rank=get_parallel().tp_rank,
    )
    logger.info(
        "Logging decode indexer top-k of %d layers to %s",
        len(header.layer_ids),
        log_dir,
    )
    return IndexerTopkLogCapturer(
        log_dir=log_dir,
        header=header,
        num_model_layers=num_model_layers,
        max_batch_size=max_running_requests,
        device=device,
    )
