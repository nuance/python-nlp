from collections import defaultdict
from math import log, exp
from itertools import izip

class Counter(defaultdict):

	def __init__(self):
		super(Counter, self).__init__(lambda:0.0)

	def __init__(self, *args):
		super(Counter, self).__init__(lambda:0.0, *args)

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

		for (key, value) in self.items():
			self[key] = value / sum

	def log_normalize(self):
		log_sum = log(sum(exp(val) for val in self.itervalues()))

		for key in self.iterkeys():
			self[key] -= log_sum

	def __str__(self):
		return "[%s]" % (" ".join(["%s : %f," % (key, value) for (key, value) in self.iteritems()]))

	# mul => element-wise multiplication
	def __imul__(self, other):
		keys = set(self.iterkeys())
		keys.update(other.iterkeys())
		
		for key in keys:
			self[key] *= other[key]

		return self

	# mul => element-wise multiplication
	def __mul__(self, other):
		if isinstance(other, (int, long, float)):
			return Counter((key, value * other) for (key, value) in self.iteritems())
		
		keys = set(self.iterkeys())
		keys.update(other.iterkeys())
		# FIXME: This is polluting the keys in self and other...
		lval = Counter((key, self[key] * other[key]) for key in keys)

		return lval

	def __iadd__(self, other):
		keys = set(self.iterkeys())
		keys.update(other.iterkeys())
		
		for key in keys:
			self[key] += other[key]

		return self

	def __isub__(self, other):
		keys = set(self.iterkeys())
		keys.update(other.iterkeys())
		
		for key in keys:
			self[key] -= other[key]

		return self

	def __sub__(self, other):
		keys = set(self.iterkeys())
		keys.update(other.iterkeys())

		return Counter((key, self[key] - other[key]) for key in keys)

def test():
	all_spam = Counter()
	all_spam['spam'] = 2

	assert(all_spam.total_count() == 2)
	assert(all_spam.arg_max() == 'spam')
	
	all_spam.normalize()

	assert(all_spam['spam'] == 1.0)
	assert(all_spam['ham'] == 0.0)
	assert(all_spam.arg_max() == 'spam')

	assert('spam' in all_spam.keys())
	assert('ham' in all_spam.keys())

	assert(len(all_spam.keys()) == 2)
	assert(len(all_spam.values()) == 2)

	print "All spam: %s" % all_spam
	
	del(all_spam)

	half_spam = Counter()
	half_spam['spam'] += 1
	half_spam['ham'] += 1

	assert(half_spam.total_count() == 2)
	assert(len(half_spam.keys()) == 2)
	assert(half_spam.arg_max() in ('spam', 'ham'))

	half_spam.normalize()
	
	assert(half_spam['spam'] == 0.5)
	assert(half_spam['ham'] == 0.5)
	assert(half_spam.arg_max() in ('spam', 'ham'))
	assert(len(half_spam.keys()) == 2)

	half_spam = half_spam * 2.0

	assert(half_spam['spam'] == 1.0)
	assert(half_spam['ham'] == 1.0)
	assert(half_spam.arg_max() in ('spam', 'ham'))
	assert(len(half_spam.keys()) == 2)

	half_spam = half_spam * 0.5

	assert(half_spam['spam'] == 0.5)
	assert(half_spam['ham'] == 0.5)
	assert(half_spam.arg_max() in ('spam', 'ham'))
	assert(len(half_spam.keys()) == 2)

	print "Half spam: %s" % half_spam

	bob = half_spam * 2

	assert(bob['spam'] == 1.0)
	assert(bob['ham'] == 1.0)
	assert(bob.arg_max() in ('spam', 'ham'))
	assert(len(bob.keys()) == 2)

	jim = bob * half_spam
	
	assert(jim['spam'] == 0.5)
	assert(jim['ham'] == 0.5)
	assert(jim.arg_max() in ('spam', 'ham'))
	assert(len(jim.keys()) == 2)

	bob -= jim

	assert(bob['spam'] == 0.5)
	assert(bob['ham'] == 0.5)
	assert(bob.arg_max() in ('spam', 'ham'))
	assert(len(bob.keys()) == 2)
	
	del(half_spam)
	del(bob)
	del(jim)

	log_third_spam = Counter()
	log_third_spam['spam'] += log(1)
	log_third_spam['ham'] += log(2)

	log_third_spam.log_normalize()

	assert(log_third_spam['spam'] == log(1)-log(3))
	assert(log_third_spam['ham'] == log(2)-log(3))

	print "Log third spam: %s" % log_third_spam

	del(log_third_spam)

	amul = Counter()
	amul['bob'] = 2
	amul['jim'] = 2
	bmul = Counter()
	bmul['bob'] = 4

	amul *= bmul
	bmul *= amul

	assert(amul['bob'] == 8)
	assert(bmul['bob'] == 32)
	assert(amul['jim'] == 0)
	assert(bmul['jim'] == 0)

	del(amul)
	del(bmul)
	
	aadd = Counter()
	aadd['bob'] = 2
	badd = Counter()
	badd['bob'] = 4

	aadd += badd

	assert(aadd['bob'] == 6)
	assert(badd['bob'] == 4)
	

if __name__ == "__main__":
	test()
