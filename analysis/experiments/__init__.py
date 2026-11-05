"""Grouped experiment sources with stable legacy module names for checkpoints.

The extra search paths preserve commands such as
``python -m analysis.experiments.third_wave_models`` and saved pickle imports.
"""

from pathlib import Path

_root = Path(__file__).resolve().parent
__path__.extend(str(folder) for folder in sorted(_root.rglob('*'))
                if folder.is_dir() and folder.name != '__pycache__'
                and any(folder.glob('*.py')))
