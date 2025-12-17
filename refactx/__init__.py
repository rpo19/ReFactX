from .ctrie import *
from .index import *
__all__ = [name for name in globals().keys() if not name.startswith('_')]

def _read_version_from_pyproject():
	try:
		import toml
		from pathlib import Path
		p = Path(__file__).resolve().parents[1] / 'pyproject.toml'
		data = toml.load(p)
		# PEP 621 field
		version = data.get('project', {}).get('version')
		if version:
			return version
	except Exception:
		pass
	return '0.0.0'

__version__ = _read_version_from_pyproject()
