from copy import copy
from nlp import counter
from math import log

def test():
	all_spam = counter()
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

	half_spam = counter()
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

	print "Half spam: %s" % half_spam

	half_spam = half_spam * 2.0

	print "Half spam: %s" % half_spam
	
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

	print "bob = half_spam * 2 done"
	
	print bob, "*",
	print half_spam, "=",
	jim = bob * half_spam
	print jim
	
	assert(jim['spam'] == 0.5)
	assert(jim['ham'] == 0.5)
	assert(jim.arg_max() in ('spam', 'ham'))
	assert(len(jim.keys()) == 2)

	print "jim = bob * half_spam done"

	print bob
	bob -= jim

	print "sub done"
	print jim
	
	assert(bob['spam'] == 0.5)
	assert(bob['ham'] == 0.5)
	assert(bob.arg_max() in ('spam', 'ham'))
	assert(len(bob.keys()) == 2)

	print "bob -= jim done"
	
	del(half_spam)
	del(bob)
	del(jim)

	log_third_spam = counter()
	log_third_spam['spam'] += log(1)
	log_third_spam['ham'] += log(2)

	log_third_spam.log_normalize()

	assert(log_third_spam['spam'] == log(1)-log(3))
	assert(log_third_spam['ham'] == log(2)-log(3))

	print "Log third spam: %s" % log_third_spam

	del(log_third_spam)

	amul = counter()
	amul['bob'] = 2
	amul['jim'] = 2
	bmul = counter()
	bmul['bob'] = 4

	amul *= bmul
	bmul *= amul

	assert(amul['bob'] == 8)
	assert(bmul['bob'] == 32)
	assert(amul['jim'] == 0)
	assert(bmul['jim'] == 0)

	del(amul)
	del(bmul)
	
	aadd = counter()
	aadd['bob'] = 2
	badd = counter()
	badd['bob'] = 4

	aadd += badd

	assert(aadd['bob'] == 6)
	assert(badd['bob'] == 4)

	base = counter()
	sub = counter()

	base['cat'] += 1
	base['dog'] += 1

	sub['cat'] = 0.001

	# Exercise the garbage collector - ideally we should be caching recently
	# released counters to optimize for this
	for i in xrange(10000):
		jim = base - sub

	print jim

	# Testing default values
	for operation in ("__add__", "__mul__"):
		print "Testing %s" % operation

		foo = counter()
		foo.default = float("-inf")
		foo['a'] = 1.0

		bar = counter()
		bar.default = float("-inf")
		bar['b'] = 2.0

		foofunc = getattr(foo, operation)
		barfunc = getattr(bar, operation)

		# Transitivity
		assert foofunc(bar) == barfunc(foo), "%s != %s" % (foofunc(bar), barfunc(foo))
		print "Transitivity passed"

		# Test that the values are correct
		bob = foofunc(bar)
		print bob
		assert bob['a'] == float("-inf")
		assert bob['b'] == float("-inf")
		if operation == "__add__": val = float("-inf")
 		elif operation == "__mul__": val = float("inf")
		assert bob['missing'] == val
		print "Values passed"

		# Verify that the originals are unchanged
		assert foo['a'] == 1.0 and foo['missing'] == float("-inf")
		assert bar['b'] == 2.0 and bar['missing'] == float("-inf")
		print "Changes passed"

	# Testing default values for in-place ops
	for operation in ("__iadd__", "__imul__"):
		print "Testing %s" % operation

		foo = counter()
		foo.default = float("-inf")
		foo['a'] = 1.0

		bar = counter()
		bar.default = float("-inf")
		bar['b'] = 2.0
		
		foofunc = getattr(foo, operation)
		barfunc = getattr(bar, operation)
		orig = copy(foo)
		orig.default = foo.default

		foofunc(bar)
		barfunc(orig)

		# Transitivity
		assert foo == bar, "%s != %s" % (foo, bar)
		print "Transitivity passed"

		# Test that the values are correct
		assert bar['a'] == float("-inf")
		assert bar['b'] == float("-inf")
		if operation == "__iadd__": val = float("-inf")
 		elif operation == "__imul__": val = float("inf")
		assert bar['missing'] == val
		print "Values passed"

if __name__ == "__main__":
	test()
