import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.state_capturer.indexer_topk_log import (
    IndexerTopkLogCapturer,
    IndexerTopkLogHeader,
    create_indexer_topk_log_capturer,
    hash_rid,
    read_header,
    read_steps,
    write_header,
    write_step,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TOPK = 8
# Layers 1 and 2 reuse layer 0's selection, so only 0 and 3 are logged.
_LAYER_IDS = [0, 3]
_NUM_MODEL_LAYERS = 4


def _header() -> IndexerTopkLogHeader:
    return IndexerTopkLogHeader(
        version=1,
        topk=_TOPK,
        layer_ids=_LAYER_IDS,
        ids_are_token_positions=True,
        model_path="dummy",
        rank=0,
    )


def _forward_batch(*, decode: bool, rids: list[str], seq_lens: list[int]):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: decode),
        rids=rids,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
    )


class TestIndexerTopkLogFormat(CustomTestCase):
    def test_round_trip_keeps_id_sets_and_padding(self):
        rng = np.random.default_rng(0)
        topk = np.stack(
            [
                np.stack([rng.permutation(100_000)[:_TOPK] for _ in _LAYER_IDS]).astype(
                    np.int32
                )
                for _ in range(3)
            ]
        )
        # Sequences shorter than top-k pad with -1.
        topk[2, :, 5:] = -1
        file = io.BytesIO()
        write_header(file, _header())
        write_step(
            file,
            forward_count=7,
            rid_hashes=[hash_rid(f"s{i}:t0") for i in range(3)],
            seq_lens=np.array([10, 20, 5]),
            topk=topk,
        )
        file.seek(0)
        header = read_header(file)
        (step,) = list(read_steps(file, header))

        self.assertEqual(header, _header())
        self.assertEqual(step.forward_count, 7)
        self.assertEqual(step.seq_lens.tolist(), [10, 20, 5])
        self.assertEqual(step.rid_hashes.tolist()[1], hash_rid("s1:t0"))
        np.testing.assert_array_equal(step.topk, np.sort(topk, axis=-1))


class TestIndexerTopkLogCapturer(CustomTestCase):
    def test_logs_decode_steps_of_indexer_layers_only(self):
        with tempfile.TemporaryDirectory() as log_dir:
            capturer = IndexerTopkLogCapturer(
                log_dir=log_dir,
                header=_header(),
                num_model_layers=_NUM_MODEL_LAYERS,
                max_batch_size=2,
                device="cpu",
            )
            # A prefill forward has more rows than the decode-sized buffer and
            # must neither crash nor be logged.
            prefill = torch.arange(5 * _TOPK, dtype=torch.int32).reshape(5, _TOPK)
            capturer.capture(0, prefill)
            capturer.on_forward_end(
                forward_batch=_forward_batch(decode=False, rids=["a:0"], seq_lens=[5])
            )

            layer0 = torch.tensor([[7, 1, 3, 5, 2, 4, 6, 0], [9] * _TOPK]).int()
            layer3 = layer0 + 100
            capturer.capture(0, layer0)
            capturer.capture(1, layer0 + 1000)  # shared-index layer: ignored
            capturer.capture(3, layer3)
            capturer.on_forward_end(
                forward_batch=_forward_batch(
                    decode=True, rids=["a:0", "b:4"], seq_lens=[64, 1024]
                )
            )
            capturer.destroy()

            with open(os.path.join(log_dir, "indexer_topk_rank0.bin"), "rb") as file:
                steps = list(read_steps(file, read_header(file)))
            with open(os.path.join(log_dir, "indexer_topk_rank0.rids.jsonl")) as file:
                rids = [json.loads(line) for line in file]

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].seq_lens.tolist(), [64, 1024])
        self.assertEqual(steps[0].topk.shape, (2, len(_LAYER_IDS), _TOPK))
        self.assertEqual(steps[0].topk[0, 0].tolist(), list(range(8)))
        self.assertEqual(steps[0].topk[0, 1].tolist(), list(range(100, 108)))
        self.assertEqual(
            rids, [{"hash": hash_rid(rid), "rid": rid} for rid in ("a:0", "b:4")]
        )


class TestIndexerTopkLogGate(CustomTestCase):
    def _create(self):
        hf_text_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            num_hidden_layers=_NUM_MODEL_LAYERS,
            index_topk=_TOPK,
            indexer_types=["full", "shared", "shared", "full"],
        )
        capturer = create_indexer_topk_log_capturer(
            model_config=SimpleNamespace(
                hf_text_config=hf_text_config, model_path="dummy"
            ),
            max_running_requests=2,
            device="cpu",
        )
        if capturer is not None:
            self.addCleanup(capturer.destroy)
        return capturer

    def test_off_by_default_and_one_writer_per_attention_tp_group(self):
        override = get_context().override_server_args(enable_hisparse=False)
        override.install()
        self.addCleanup(override.restore)
        self.assertIsNone(self._create())

        with tempfile.TemporaryDirectory() as log_dir:
            with envs.SGLANG_INDEXER_TOPK_LOG_DIR.override(log_dir):
                with get_parallel().override(
                    tp_size=2, moe_tp_size=2, attn_tp_size=2, attn_tp_rank=1, tp_rank=1
                ):
                    self.assertIsNone(self._create())
                with get_parallel().override(attn_tp_rank=0, tp_rank=0):
                    capturer = self._create()
        self.assertEqual(capturer.header.layer_ids, _LAYER_IDS)
        # Fused top-k is the default, so ids are KV slots unless it is disabled.
        self.assertFalse(capturer.header.ids_are_token_positions)


if __name__ == "__main__":
    unittest.main()
