from collections import defaultdict
from math import log, exp

__use_c_counter__ = True

if __use_c_counter__:
	from nlp import counter as Counter
	print "Using C counter"
else:
	print "Using python counter"
	class Counter(dict):
		default = 0.0

		def __missing__(self, key):
			self[key] = self.default

			return self[key]

		def __init__(self):
			super(Counter, self).__init__()

		def __init__(self, *args):
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
				uniform = 1 / len(self.iteritems())
				for key in self.iteritems():
					self[key] = uniform

			for (key, value) in self.iteritems():
				self[key] /= sum

		def log_normalize(self):
			log_sum = log(sum(exp(val) for val in self.itervalues()))

			for key in self.iterkeys():
				self[key] -= log_sum

		def log(self):
			for key in self.iterkeys():
				self[key] = log(self[key])

		def inner_product(self, other):
			keys = set(self.iterkeys())
			keys.update(other.iterkeys())

			return sum((self.d_get(key) * other.d_get(key)) for key in keys)

		def d_get(self, key):
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


if __name__ == "__main__":
	test()
