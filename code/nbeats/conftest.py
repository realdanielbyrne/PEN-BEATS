"""Root pytest configuration.

Adds ``src/`` to ``sys.path`` so that ``import nbeats_anon`` works when
running ``pytest tests/`` directly from the ``code/`` directory, without
requiring a prior ``pip install -e .``.

After a ``pip install -e .`` (or ``pip install -r requirements.txt && pip
install -e .``) this file is a no-op because the editable-install entry-
point already resolves the package.
"""

import os
import sys

# Insert src/ at position 0 so the local copy always shadows any installed version.
_ROOT = os.path.dirname(__file__)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
