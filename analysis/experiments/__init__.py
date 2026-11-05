"""Grouped experiment sources with stable legacy module names for checkpoints.

The extra search paths preserve commands such as
``python -m analysis.experiments.third_wave_models`` and saved pickle imports.

Root-level modules retain the TabPFN branch implementations. Where both
workflows use the same module name, zch code imports the explicit grouped path
(``analysis.experiments.baselines.*``) to avoid resolving to a TabPFN module.
"""

from pathlib import Path

_root = Path(__file__).resolve().parent
__path__.extend(str(folder) for folder in sorted(_root.rglob('*'))
                if folder.is_dir() and folder.name != '__pycache__'
                and any(folder.glob('*.py')))
