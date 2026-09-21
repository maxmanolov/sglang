import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import torch

from sglang.srt.configs.model_config import dsa_layer_skips_topk
from sglang.srt.mem_cache import kv_cache_configurator
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.srt.mem_cache.kv_cache_configurator import (
    KVCacheConfigurator,
    _should_elide_dsa_index_k,
)
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


# 3 full + 5 shared; layer 0 must be full (nothing earlier to share with).
_SHARED_INDEXER_TYPES = [
    "full",
    "shared",
    "shared",
    "full",
    "shared",
    "shared",
    "shared",
    "full",
]
_NUM_FULL = _SHARED_INDEXER_TYPES.count("full")
_NUM_LAYERS = len(_SHARED_INDEXER_TYPES)
# Per token per layer: fp8 MLA KV (512 + 512 // 128 * 4 + 64 * 2), index-K (128 + 4).
_FP8_KV_BYTES = 656
_INDEX_K_BYTES = 132


def _make_dsa_hf_config(indexer_types: list[str] | None) -> SimpleNamespace:
    hf_config = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk=2048,
        index_head_dim=128,
    )
    if indexer_types is not None:
        hf_config.indexer_types = indexer_types
    hf_config.get_text_config = lambda: hf_config
    return hf_config


class TestHiSparsePoolConfigurator(CustomTestCase):
    def _compute_cell_size(
        self,
        kv_cache_dtype: torch.dtype,
        *,
        enable_hisparse: bool,
        host_to_device_ratio: int = 1,
        indexer_types: list[str] | None = None,
        disaggregation_mode: str = "null",
        is_draft_worker: bool = False,
    ) -> int:
        num_layers = len(indexer_types) if indexer_types is not None else 2
        hf_config = _make_dsa_hf_config(indexer_types)
        server_args = self._install_server_args(
            enable_hisparse=enable_hisparse,
            host_to_device_ratio=host_to_device_ratio,
            disaggregation_mode=disaggregation_mode,
        )

        kvc = MagicMock(
            use_mla_backend=True,
            kv_cache_dtype=kv_cache_dtype,
            is_draft_worker=is_draft_worker,
            mambaish_config=None,
            model_config=SimpleNamespace(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                hf_config=hf_config,
            ),
            layer_info=SimpleNamespace(start_layer=0, end_layer=num_layers),
            server_args=server_args,
        )

        with get_parallel().override(attn_tp_size=1):
            configurator = object.__new__(DefaultPoolConfigurator)
            return configurator._compute_cell_size(kvc, num_layers=num_layers)

    def _install_server_args(
        self,
        *,
        enable_hisparse: bool,
        host_to_device_ratio: int = 1,
        disaggregation_mode: str = "null",
    ):
        override = get_context().override_server_args(
            enable_hisparse=enable_hisparse,
            hisparse_config=f'{{"host_to_device_ratio": {host_to_device_ratio}}}',
            enable_hierarchical_cache=False,
            disaggregation_mode=disaggregation_mode,
            dsa_prefill_backend="flashmla_sparse",
            dsa_decode_backend="flashmla_sparse",
        )
        server_args = override.install()
        self.addCleanup(override.restore)
        return server_args

    def test_mla_layout_without_hisparse(self):
        for kv_cache_dtype, expected_cell_size in (
            (torch.bfloat16, 2568),
            (torch.float8_e4m3fn, 1576),
        ):
            with self.subTest(kv_cache_dtype=kv_cache_dtype):
                cell_size = self._compute_cell_size(
                    kv_cache_dtype,
                    enable_hisparse=False,
                )
                self.assertEqual(cell_size, expected_cell_size)

    def test_hisparse_indexer_scales_with_ratio(self):
        for host_to_device_ratio, expected_cell_size in (
            (2, 1840),
            (4, 2368),
        ):
            with self.subTest(host_to_device_ratio=host_to_device_ratio):
                cell_size = self._compute_cell_size(
                    torch.float8_e4m3fn,
                    enable_hisparse=True,
                    host_to_device_ratio=host_to_device_ratio,
                )
                self.assertEqual(cell_size, expected_cell_size)

    def test_hisparse_elides_shared_index_layers(self):
        for ratio in (2, 4, 10):
            with self.subTest(host_to_device_ratio=ratio):
                cell_size = self._compute_cell_size(
                    torch.float8_e4m3fn,
                    enable_hisparse=True,
                    host_to_device_ratio=ratio,
                    indexer_types=_SHARED_INDEXER_TYPES,
                )
                self.assertEqual(
                    cell_size,
                    _NUM_LAYERS * _FP8_KV_BYTES + _NUM_FULL * _INDEX_K_BYTES * ratio,
                )

    def test_hisparse_keeps_all_index_layers_when_elision_is_off(self):
        for kwargs in (
            dict(disaggregation_mode="decode"),
            dict(is_draft_worker=True),
        ):
            with self.subTest(**kwargs):
                cell_size = self._compute_cell_size(
                    torch.float8_e4m3fn,
                    enable_hisparse=True,
                    host_to_device_ratio=4,
                    indexer_types=_SHARED_INDEXER_TYPES,
                    **kwargs,
                )
                self.assertEqual(
                    cell_size,
                    _NUM_LAYERS * (_FP8_KV_BYTES + _INDEX_K_BYTES * 4),
                )

    def test_elided_bytes_become_token_capacity(self):
        # GLM-5.2 layout: 78 layers, index_topk_freq=4 / index_skip_topk_offset=3
        # leaves 21 indexer layers. fp8 KV, host_to_device_ratio=10, 64 GiB free.
        glm52_indexer_types = [
            "shared" if max(layer_id - 2, 0) % 4 else "full" for layer_id in range(78)
        ]
        self.assertEqual(glm52_indexer_types.count("full"), 21)

        configurator = object.__new__(DefaultPoolConfigurator)
        configurator._cell_size = self._compute_cell_size(
            torch.float8_e4m3fn,
            enable_hisparse=True,
            host_to_device_ratio=10,
            indexer_types=glm52_indexer_types,
        )
        configurator._zero_kv_max_tokens = 0
        pool_config = configurator.calculate_pool_sizes(64 << 30, page_size=64)

        # 78 * 656 + 21 * 132 * 10 = 78,888 B/token; all 78 layers would cost
        # 154,128 B/token and yield 445,824 tokens.
        self.assertEqual(configurator._cell_size, 78_888)
        self.assertEqual(pool_config.max_total_num_tokens, 871_040)


class TestHiSparseIndexKElision(CustomTestCase):
    def _install_server_args(self, **overrides):
        override = get_context().override_server_args(
            **{
                "enable_hisparse": True,
                "hisparse_config": '{"host_to_device_ratio": 4}',
                "enable_hierarchical_cache": False,
                "disaggregation_mode": "null",
                **overrides,
            }
        )
        override.install()
        self.addCleanup(override.restore)

    def test_gate(self):
        for overrides, is_draft_worker, expected in (
            ({}, False, True),
            ({"enable_hisparse": False}, False, True),
            ({}, True, False),
            ({"disaggregation_mode": "decode"}, False, False),
            ({"disaggregation_mode": "prefill"}, False, False),
        ):
            with self.subTest(overrides=overrides, is_draft_worker=is_draft_worker):
                self._install_server_args(**overrides)
                self.assertEqual(
                    _should_elide_dsa_index_k(is_draft_worker=is_draft_worker),
                    expected,
                )

    def test_index_key_cache_allocates_only_full_layers(self):
        page_size, device_tokens, ratio = 64, 64 * 8, 4
        skip_topk_layers = [t == "shared" for t in _SHARED_INDEXER_TYPES]
        pool = SimpleNamespace(
            page_size=page_size,
            index_head_dim=128,
            quant_block_size=128,
            custom_mem_pool=None,
            index_k_with_scale_buffer_dtype=torch.uint8,
            device="cpu",
            layer_num=_NUM_LAYERS,
            skip_topk_layers=skip_topk_layers,
        )
        # HiSparseDSATokenToKVPool sizes index-K over the host-backed logical pool.
        cache = IndexKeyCache(pool, index_buf_size=device_tokens * ratio)

        num_pages = (device_tokens * ratio + page_size + 1) // page_size
        for layer_id, skipped in enumerate(skip_topk_layers):
            with self.subTest(layer_id=layer_id):
                self.assertEqual(
                    tuple(cache.buffer[layer_id].shape),
                    (0 if skipped else num_pages, page_size * _INDEX_K_BYTES),
                )
        allocated = sum(buf.nbytes for buf in cache.buffer)
        self.assertEqual(allocated, _NUM_FULL * num_pages * page_size * _INDEX_K_BYTES)

    def test_hisparse_pool_receives_skip_topk_layers(self):
        self._install_server_args()
        hf_config = _make_dsa_hf_config(_SHARED_INDEXER_TYPES)
        kvc = object.__new__(KVCacheConfigurator)
        kvc.is_draft_worker = False
        kvc.model_config = SimpleNamespace(
            kv_lora_rank=512, qk_rope_head_dim=64, hf_config=hf_config
        )
        kvc.layer_info = SimpleNamespace(
            start_layer=0, end_layer=_NUM_LAYERS, num_effective_layers=_NUM_LAYERS
        )
        kvc.kv_cache_dtype = torch.float8_e4m3fn
        kvc.device = "cpu"

        with (
            patch.object(kv_cache_configurator, "HiSparseDSATokenToKVPool") as pool_cls,
            patch.object(
                KVCacheConfigurator,
                "pool_page_size",
                new_callable=PropertyMock,
                return_value=64,
            ),
            patch(
                "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
                return_value=(None, 1),
            ),
        ):
            kvc._build_dsa_kv_pool(max_total_num_tokens=640, max_running_requests=4)

        kwargs = pool_cls.call_args.kwargs
        self.assertEqual(kwargs["host_to_device_ratio"], 4)
        self.assertEqual(
            kwargs["skip_topk_layers"],
            [dsa_layer_skips_topk(hf_config, i) for i in range(_NUM_LAYERS)],
        )
        self.assertEqual(kwargs["skip_topk_layers"].count(False), _NUM_FULL)


if __name__ == "__main__":
    unittest.main()
