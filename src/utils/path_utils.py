import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def ensure_src_on_syspath() -> Path:
    """
    Guarantee that 'src' directory is on sys.path.
    Works whether you run with or without `-m`.
    """
    src_path = Path(__file__).resolve().parent.parent  # J:\Gold_FX\src
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return src_path

def get_project_root(default_parents=3) -> Path:
    """
    Dynamically resolve project root with environment fallback.
    - Uses PROJECT_ROOT if set.
    - Otherwise climbs up `default_parents` levels from this file.
    - Falls back to cwd() if structure is unexpected.
    """
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    try:
        return Path(__file__).resolve().parents[default_parents]
    except IndexError:
        logger.warning("Falling back to current working directory as project root.")
        return Path.cwd()
