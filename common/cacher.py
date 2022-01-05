from typing import Any, Tuple, Dict
from pathlib import Path
import pickle
import inspect
import functools

class Cacher:
	def __init__(self, base: Path):
		self.base = base
	
	def __truediv__(self, segment: str):
		return type(self)(self.base / segment)
	
	def __call__(self, filename: str):
		def decorator(f):
			sig = inspect.Signature.from_callable(f)
			
			@functools.wraps(f)
			def f_cached(*args: Any, **kwargs: Any) -> Any:
				cache_file = self.base / filename.format(**_bind_args(sig, args, kwargs))
				return self._call_cached(cache_file, f, args, kwargs)
			f_cached.uncached = f
			
			return f_cached
		
		return decorator
	
	def call(self, file: str, f, *args, **kwargs):
		cache_file = self.base / file
		return self._call_cached(cache_file, f, args, kwargs)
	
	def _call_cached(self, cache_file: Path, f, args, kwargs):
		if not cache_file.exists():
			cache_file.parent.mkdir(exist_ok = True, parents = True)
			obj = f(*args, **kwargs)
			with cache_file.open('wb') as fh:
				pickle.dump(obj, fh)
		with cache_file.open('rb') as fh:
			return pickle.load(fh)

def _bind_args(sig: inspect.Signature, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
	bound_args = sig.bind(*args, **kwargs)
	bound_args.apply_defaults()
	return bound_args.arguments
