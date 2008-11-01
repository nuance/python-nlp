from collections import defaultdict
from copy import copy
from itertools import chain, izip, repeat

from nlp import counter
from counter import Counter

class CounterMap(defaultdict):
	use_c_counter = False
	
	def __init__(self, use_c_counter = True, default=0.0):
		self.use_c_counter = use_c_counter
		
		if use_c_counter:
			def counter_with_default():
				ret = counter()
				ret.default = default
				return ret

			super(CounterMap, self).__init__(counter_with_default)
		else:
			def counter_with_default():
				ret = Counter()
				ret.default = default
				return ret

			super(CounterMap, self).__init__(counter_with_default)

	def normalize(self):
		for key in self.iterkeys():
			self[key].normalize()

	def log_normalize(self):
		for key in self.iterkeys():
			self[key].log_normalize()

	def log(self):
		for sub_counter in self.itervalues():
			sub_counter.log()

	def linearize(self):
		"""Return an iterator over (key, subkey) pairs (so we can view a countermap as a vector)
		FIXME: this isn't guaranteed to return the same thing every time"""
		return chain([izip(repeat(key, len(counter.iteritems())), counter.iteritems()) for (key, counter) in self.iteritems()])

	def inner_product(self, other):
		ret = 0.0

		for key, counter in self.iteritems():
			if key not in other: continue
			ret += sum((counter * other[key]).itervalues())

		return ret

	def scale(self, other):
		ret = CounterMap(self.use_c_counter)

		for key, counter in self.iteritems():
			ret[key] = counter * other

		return ret

	def __mul__(self, other):
		if isinstance(other, (int, long, float)):
			return self.scale(other)

		ret = CounterMap(self.use_c_counter)

		for key, counter in self.iteritems():
			if key not in other: continue
			ret[key] = counter * other[key]

		return ret

	def __add__(self, other):
		ret = CounterMap(self.use_c_counter)

		for (key, counter) in self.iteritems():
			if key in other:
				ret[key] = counter + other[key]
			else:
				ret[key] = copy(counter)

		for key in (set(other.iterkeys()) - set(self.iterkeys())):
			ret[key] = copy(other[key])

		return ret

	def __sub__(self, other):
		ret = CounterMap(self.use_c_counter)

		for (key, counter) in self.iteritems():
			if key in other:
				ret[key] = counter - other[key]
			else:
				ret[key] = copy(counter)

		for key in (set(other.iterkeys()) - set(self.iterkeys())):
			ret[key] = type(other[key])() - other[key]

		return ret
	
	def __str__(self):
		string = ""
		for (key, counter) in self.iteritems():
			string += "%s : %s\n" % (key, counter)
		return string.rstrip()

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
