from typing import Any, List, Dict
import functools
import pickle
from pathlib import Path
import inspect

class Cacher:
	def __init__(self, base: Path) -> None:
		super().__init__()
		self.base = base
	
	def __call__(self, filename: str):
		def decorator(f):
			sig = inspect.Signature.from_callable(f)
			
			@functools.wraps(f)
			def f_cached(*args: Any, **kwargs: Any) -> Any:
				cache_file = self.base / filename.format(**_bind_args(sig, args, kwargs))
				cache_file.parent.mkdir(exist_ok = True, parents = True)
				if not cache_file.exists():
					obj = f(*args, **kwargs)
					with cache_file.open('wb') as fh:
						pickle.dump(obj, fh)
					return obj
				with cache_file.open('rb') as fh:
					return pickle.load(fh)
			f_cached.uncached = f
			
			return f_cached
		
		return decorator

def _bind_args(sig: inspect.Signature, args: List[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
	bound_args = sig.bind(*args, **kwargs)
	bound_args.apply_defaults()
	return bound_args.arguments
