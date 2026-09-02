"""
槽位④: MiniCPM-o 数据集——把静默失败变成显式报错

上游 maybe_filter_out_long_prompts 用 except Exception: return max_prompt_length+1 兜底，
处理器一抛异常就把 7473 行样本全过滤成 0 行，只在日志刷 warning。
子类覆写后强制走 tokenizer 分支，并在过滤比例超阈值时直接报错。
"""
import logging

logger = logging.getLogger(__name__)

FILTER_RATIO_THRESHOLD = 0.5  # 超过 50% 样本被过滤就报错


class MiniCPMOThinkerRLHFDataset:
    """MiniCPM-o RLHF 数据集

    用法（config.yaml）：
        data:
          custom_cls:
            path: pkg://verl_omni_ext.models.minicpmo_5_0.dataset
            name: MiniCPMOThinkerRLHFDataset
    """

    def filter_long_prompts(self, dataset, max_prompt_length, processor=None):
        """覆写上游的 maybe_filter_out_long_prompts

        上游用 except Exception 兜底，这里强制走 tokenizer 分支，
        并在过滤比例超阈值时报错而不是静默继续。
        """
        import os
        if os.environ.get("SKIP_DATASET_SCHEMA_CHECK", "0") == "1":
            return dataset

        total = len(dataset)
        kept = 0
        for item in dataset:
            try:
                # 强制走 tokenizer 分支，不依赖 processor
                input_ids = item.get("input_ids", [])
                if len(input_ids) <= max_prompt_length:
                    kept += 1
            except Exception as e:
                # 这里不吞——喊出来
                raise RuntimeError(
                    f"Dataset filtering failed on item: {e}. "
                    f"Upstream would have silently filtered this to 0 rows. "
                    f"Use SKIP_DATASET_SCHEMA_CHECK=1 to bypass for debugging."
                ) from e

        filtered_ratio = 1 - kept / total if total > 0 else 0
        if filtered_ratio > FILTER_RATIO_THRESHOLD:
            raise RuntimeError(
                f"Dataset filtering ratio {filtered_ratio:.1%} exceeds threshold "
                f"{FILTER_RATIO_THRESHOLD:.0%} (kept {kept}/{total}). "
                f"This usually means processor is broken — check MINICPMO_DISABLE_PROCESSOR_FALLBACK."
            )

        logger.info(f"Dataset filtered: kept {kept}/{total} ({1 - filtered_ratio:.1%})")
        return dataset
