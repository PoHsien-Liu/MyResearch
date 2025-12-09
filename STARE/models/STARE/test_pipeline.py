import logging
import time
from pathlib import Path

from STARE.eval.inference_eval import run_inference_eval
from STARE.models.STARE.sft_dataset_builder import prepare_sft_samples


LOGGER = logging.getLogger(__name__)


def run_test(args) -> None:
    """Full test flow: ensure SFT samples for test split, then run inference/eval."""
    exp = args.experiment_name or str(int(time.time()))
    args.experiment_name = exp

    sft_test_path = prepare_sft_samples(args, mode="test")

    if getattr(args, "adapter_path", None):
        target_adapter = Path(args.adapter_path)
        if not target_adapter.exists():
            raise FileNotFoundError(f"Adapter checkpoint not found at {target_adapter}")
        args.adapter_path = str(target_adapter)
    else:
        args.adapter_path = None
        LOGGER.info("No adapter_path provided; running base model only.")

    args.sft_path = str(sft_test_path)
    args.temperature = getattr(args, "temperature", 0.0)

    LOGGER.info("Running inference eval on test split (exp=%s)", exp)
    run_inference_eval(args)
    LOGGER.info("test_pipeline completed (exp=%s). Predictions/eval written to outputs/.", exp)
