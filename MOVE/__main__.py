"""Module entry point for running MOVE via ``python -m MOVE``."""

import logging
import warnings

warnings.filterwarnings("ignore", message=".*has_cuda.*deprecated.*")
warnings.filterwarnings("ignore", message=".*has_cudnn.*deprecated.*")
warnings.filterwarnings("ignore", message=".*has_mps.*deprecated.*")
warnings.filterwarnings("ignore", message=".*has_mkldnn.*deprecated.*")

from .move import main


def _configure_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


if __name__ == "__main__":  # pragma: no cover
    _configure_logging()
    main()
