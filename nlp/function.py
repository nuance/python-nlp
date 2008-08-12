class cached(object):
	"""Decorator that caches a function's return value for the most recent 
	invocation. If called later with the same arguments, the cached value is 
	returned, and not re-evaluated.
	"""
	
	def __init__(self, fn):
		self.func = fn
		self.cache_results = None
		self.cache_args = None
		
		self.__name__ = fn.__name__
		self.__doc__ = fn.__doc__
		self.__dict__.update(fn.__dict__)
		return None

	def __call__(self, *args, **kwargs):
		if args == self.cache_args:
			print "cached..."
			return self.cache_results
		else:
			print "computing..."
			self.cache_results = self.func(self, *args, **kwargs)
			self.cache_args = args
			return self.cache_results

class Function:
	def value(self, point):
		raise NotImplementedError()

	def gradient(self, point):
		raise NotImplementedError()

	def value_and_gradient(self, point):
		raise NotImplementedError()

class StubFunction(Function):
	"""
	This only exists for you to prototype from and provide an example of using
	cached - real implementations will almost	definitely be in C
	"""
	
	@cached
	def value(self, point):
		return -40

	@cached
	def gradient(self, point):
		return [0 for dim in point]

	@cached
	def value_and_gradient(self, point):
		return (-40, [0 for dim in point])
