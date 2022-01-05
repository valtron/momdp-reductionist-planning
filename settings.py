from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / '.cache'

try:
	# For overriding the above settings
	from settings_local import *
except ImportError:
	pass
