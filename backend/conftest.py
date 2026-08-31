"""Makes `backend/app.py` importable as a top-level `app` module from
`backend/tests/`, regardless of which directory pytest is invoked from."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
