from __future__ import annotations

from pathlib import Path


def load_generate_dataset():
    """
    Safely load generate_dataset() from data-collection/datacollection.py
    without executing the notebook-export example / inspection cells
    at the bottom of the file.
    """
    repo_root = Path(__file__).resolve().parents[1]
    source_path = repo_root / "data-collection" / "datacollection.py"

    source = source_path.read_text(encoding="utf-8")

    # Cut the file before the bottom executable notebook-export section
    stop_markers = [
        "# ## Example Usage",
        "## Example Usage",
        "# In[49]:",
    ]

    cut_index = len(source)
    for marker in stop_markers:
        idx = source.find(marker)
        if idx != -1:
            cut_index = min(cut_index, idx)

    safe_source = source[:cut_index]

    namespace: dict = {}
    exec(safe_source, namespace)

    if "generate_dataset" not in namespace:
        raise RuntimeError("Could not load generate_dataset from datacollection.py")

    return namespace["generate_dataset"]