from copy import copy
from itertools import chain, izip, repeat
from math import exp, log

import numpy

from counter import Counter

class CounterMap(dict):
	def __missing__(self, key):
		ret = Counter()
		ret.default = self.default

		self[key] = ret
		return ret

	def __init__(self, default=0.0):
		super(CounterMap, self).__init__()
		self.default = default

	def normalize(self):
		for key in self.iterkeys():
			self[key].normalize()

	def log_normalize(self):
		for key in self.iterkeys():
			self[key].log_normalize()

	def log(self):
		for sub_counter in self.itervalues():
			sub_counter.log()

		try:
			self.default = log(self.default)
		except (OverflowError, ValueError):
			self.default = float("-inf")

	def exp(self):
		for cnter in self.itervalues():
			cnter.exp()
		self.default = exp(self.default)

	def linearize(self):
		"""Return an iterator over (key, subkey) pairs (so we can view a countermap as a vector)
		FIXME: this isn't guaranteed to return the same thing every time"""
		return chain([izip(repeat(key, len(counter.iteritems())), counter.iteritems()) for (key, counter) in self.iteritems()])

	def inverted(self):
		""" Change map of {a : {b : ...}} to {b : {a : ...}}
		"""
		inverted = CounterMap()
		default = None

		for label, counter in self.iteritems():
			if not default:
				default = counter.default
			elif default != counter.default:
				raise Exception("Counters don't have the same defaults!")

			for sublabel, score in counter.iteritems():
				inverted[sublabel][label] = score
				inverted[sublabel].default = default

		return inverted

	def inner_product(self, other):
		ret = 0.0

		for key, counter in self.iteritems():
			if key not in other: continue
			ret += sum((counter * other[key]).itervalues())

		return ret

	def scale(self, other):
		ret = CounterMap()

		for key, counter in self.iteritems():
			ret[key] = counter * other

		return ret

	def __mul__(self, other):
		if isinstance(other, (int, long, float)):
			return self.scale(other)

		ret = CounterMap()

		for key, counter in self.iteritems():
			if key not in other: continue
			ret[key] = counter * other[key]

		return ret

	def __rmul__(self, other):
		return self * other

	def __add__(self, other):
		if isinstance(other, (int, long, float)):
			ret = CounterMap()
			for key, value in self.iteritems():
				ret[key] = value + other
				
			return ret

		ret = CounterMap()

		for (key, counter) in self.iteritems():
			if key in other:
				ret[key] = counter + other[key]
			else:
				ret[key] = copy(counter)

		for key in (set(other.iterkeys()) - set(self.iterkeys())):
			ret[key] = copy(other[key])

		return ret

	def __radd__(self, other):
		return self + other

	def __sub__(self, other):
		if isinstance(other, (int, long, float)):
			return self + (0-other)

		ret = CounterMap()

		for (key, counter) in self.iteritems():
			if key in other:
				ret[key] = counter - other[key]
			else:
				ret[key] = copy(counter)

		for key in (set(other.iterkeys()) - set(self.iterkeys())):
			ret[key] = type(other[key])() - other[key]

		return ret
	
	def __rsub__(self, other):
		return self - other

	def __str__(self):
		string = ""
		for (key, counter) in self.iteritems():
			string += "%s : %s\n" % (key, counter)
		return string.rstrip()

	def matrix(self):
		all_keys = set(self.iterkeys())

		for cnter in self.itervalues():
			all_keys.update(cnter.iterkeys())

		all_keys = list(sorted(all_keys))
		return all_keys, numpy.array([[self[key][sub_key] for sub_key in all_keys]
									  for key in all_keys])

	@classmethod
	def from_matrix(cls, keys, nparray):
		cnter_map = CounterMap()

		for i, key in enumerate(keys):
			for j, sub_key in enumerate(keys):
				cnter_map[key][subkey] = nparray[i][j]

		return cnter_map


def outer_product(a, b):
	# sort keys from both and return a countermap of the resulting
	# matrix
	outer = CounterMap(a.default * b.default)

	for a_key, a_value in a.iteritems():
		for b_key, b_value in b.iteritems():
			outer[a_key][b_key] = a_value * b_value

	return outer

def test():
	one_all_spam = CounterMap()
	one_all_spam['xxx']['spam'] = 2

	assert(one_all_spam['xxx'].total_count() == 2)
	assert(one_all_spam['xxx'].arg_max() == 'spam')
	
	one_all_spam.normalize()

	assert(one_all_spam['xxx']['spam'] == 1.0)
	assert(one_all_spam['xxx']['ham'] == 0.0)
	assert(one_all_spam['cheese']['ham'] == 0.0)

	print "All spam: %s" % one_all_spam
	
	del(one_all_spam)

	half_spam = CounterMap()
	half_spam['xxx']['spam'] += 1
	half_spam['xxx']['ham'] += 1
	half_spam['cheese']['spam'] += 1
	half_spam['cheese']['ham'] += 1

	half_spam.normalize()

	assert(half_spam['xxx']['spam'] == 0.5)
	assert(half_spam['cheese']['spam'] == 0.5)

	print "Half spam: %s" % half_spam

	del(half_spam)

if __name__ == "__main__":
	test()
