"""Download and unpack BBBC039, BBBC013, BBBC014 (~360 MB) into $OOC_DATA (default ./data)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ooc_readout.data import download_all  # noqa: E402

print("data at", download_all())
