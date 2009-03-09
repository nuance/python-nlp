from collections import defaultdict
from math import log, exp
import random
import os

__use_c_counter__ = (os.environ.get("COUNTER", '').lower() != 'py')


if __use_c_counter__:
	from nlp import counter as Counter
	print "Using C counter"
else:
	print "Using python counter"

	def _log(x):
		if x == 0.0: return float("-inf")
		else: return log(x)

	class Counter(dict):
		default = 0.0

		def __missing__(self, key):
			self[key] = self.default

			return self[key]

		def __init__(self, *args, **kwargs):
			if 'default' in kwargs:
				self.default = kwargs['default']
			elif len(args) == 1 and isinstance(args[0], (int, long, float)):
				self.default = args[0]
				args = []
			super(Counter, self).__init__(*args)

		# I feel like there's a better way to do this... could use reduce, but
		# that's about to be deprecated...
		def arg_max(self):
			(max_key, max_value) = (None, None)

			for (key, value) in self.iteritems():
				if not max_key or value > max_value:
					(max_key, max_value) = (key, value)

			return max_key

		def total_count(self):
			return sum(self.itervalues())

		def normalize(self):
			sum = self.total_count()

			if sum == 0.0:
				uniform = 1 / len(self)
				for key in self.iterkeys():
					self[key] = uniform
				return

			for (key, value) in self.iteritems():
				self[key] /= sum

		def log_normalize(self):
			vsum = sum(exp(v) for v in self.itervalues())
			log_sum = _log(vsum)

			for key in self.iterkeys():
				self[key] -= log_sum

		def log(self):
			for key in self.iterkeys():
				self[key] = _log(self[key])

		def exp(self):
			for key in self.iterkeys():
				self[key] = exp(self[key])

		def sample(self):
			total = self.total_count()
			point = random.random() * total

			for k, v in self.iteritems():
				point -= v
				if point < 0: return k

		def inner_product(self, other):
			keys = set(self.iterkeys())
			keys.update(other.iterkeys())

			return sum((self.d_get(key) * other.d_get(key)) for key in keys)

		def d_get(self, key):
			"""Returns the same thing as self[key], but doesn't add
			a new key if key is not in self4
			"""
			return self.get(key, self.default)

		def __str__(self):
			return "[%s]" % (" ".join(["%s : %f," % (key, value) for (key, value) in self.iteritems()]))

		# mul => element-wise multiplication
		def __imul__(self, other):
			if isinstance(other, (int, long, float)):
				for key in self.keys():
					self[key] *= other
				return self

			keys = set(self.iterkeys())
			keys.update(other.iterkeys())

			for key in keys:
				self[key] *= other.d_get(key)

			self.default *= other.default

			return self

		# mul => element-wise multiplication
		def __mul__(self, other):
			if isinstance(other, (int, long, float)):
				return Counter((key, value * other) for (key, value) in self.iteritems())

			keys = set(self.iterkeys())
			keys.update(other.iterkeys())

			lval = Counter((key, self.d_get(key) * other.d_get(key)) for key in keys)
			lval.default = self.default * other.default

			return lval

		def __rmul__(self, other):
			return self.__mul__(other)

		def __idiv__(self, other):
			if isinstance(other, (int, long, float)):
				for key in self.keys():
					self[key] /= other
				return self

			keys = set(self.iterkeys())
			keys.update(other.iterkeys())

			for key in keys:
				self[key] /= other.d_get(key)

			if other.default:
				self.default /= other.default

			return self

		# mul => element-wise multiplication
		def __div__(self, other):
			if isinstance(other, (int, long, float)):
				return Counter((key, value / other) for (key, value) in self.iteritems())

			keys = set(self.iterkeys())
			keys.update(other.iterkeys())

			lval = Counter((key, self.d_get(key) / other.d_get(key)) for key in keys)
			if other.default:
				lval.default = self.default / other.default
			else:
				lval.default = self.default

			return lval

		def __rdiv__(self, other):
			return self.__div__(other)

		def __pow__(self, power, modulo=None):
			return Counter((k, v ** power) for k, v in self.iteritems())

		def __iadd__(self, other):
			if isinstance(other, (int, long, float)):
				for key in self.keys():
					self[key] += other
				return self

			keys = set(self.iterkeys())
			keys.update(other.iterkeys())

			for key in keys:
				self[key] += other.d_get(key)

			self.default += other.default

			return self

		def __add__(self, other):
			if isinstance(other, (int, long, float)):
				return Counter((key, value + other) for (key, value) in self.iteritems())

			new = Counter()

			keys = set(self.iterkeys())
			keys.update(other.iterkeys())

			for key in keys:
				new[key] = self.d_get(key) + other.d_get(key)

			new.default = self.default + other.default

			return new

		def __radd__(self, other):
			return self.__add__(other)

		def __repr__(self):
			return "Counter(%s, default=%f)" % (super(Counter, self).__repr__(), self.default)

		def __isub__(self, other):
			if isinstance(other, (int, long, float)):
				for key in self.keys():
					self[key] -= other
				return self

			keys = set(self.iterkeys())
			keys.update(other.iterkeys())

			for key in keys:
				self[key] -= other.d_get(key)

			self.default -= other.default

			return self

		def __sub__(self, other):
			if isinstance(other, (int, long, float)):
				return Counter((key, value - other) for (key, value) in self.iteritems())

			keys = set(self.iterkeys())
			keys.update(other.iterkeys())

			new = Counter((key, self.d_get(key) - other.d_get(key)) for key in keys)
			new.default = self.default - other.default

			return new

		def __rsub__(self, other):
			return self.__sub__(other)

		def __setitem__(self, key, value):
			if not isinstance(value, (int, long, float)):
				raise ValueError("Counters can only hold numeric types")
			return super(Counter, self).__setitem__(key, value)


if __name__ == "__main__":
	test()

def counter_map(cnter, func):
	ret = Counter(func(cnter.default))

	for k, v in cnter.iteritems():
		ret[k] = func(v)

	return ret
