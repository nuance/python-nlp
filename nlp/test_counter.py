from copy import copy
from math import log
import unittest

from nlp import counter

class CounterTester(unittest.TestCase):
	def setUp(self):
		self.all_spam = counter()
		self.all_spam['spam'] = 2

		self.half_spam = counter()
		self.half_spam['spam'] += 1
		self.half_spam['ham'] += 1
	
	def test_single_key(self):
		self.assertEqual(self.all_spam.total_count(), 2)
		self.assertEqual(self.all_spam.arg_max(), 'spam')
	
		self.all_spam.normalize()

		self.assertEqual(self.all_spam['spam'], 1.0)
		self.failUnless(self.all_spam['ham'] == 0.0)
		self.failUnless(self.all_spam.arg_max() == 'spam')

		self.failUnless('spam' in self.all_spam.keys())
		self.failUnless('ham' in self.all_spam.keys())

		self.failUnless(len(self.all_spam.keys()) == 2)
		self.failUnless(len(self.all_spam.values()) == 2)


	def test_two_keys_normalize(self):
		self.failUnless(self.half_spam.total_count() == 2)
		self.failUnless(len(self.half_spam.keys()) == 2)
		self.failUnless(self.half_spam.arg_max() in ('spam', 'ham'))

		self.half_spam.normalize()

		self.failUnless(self.half_spam['spam'] == 0.5)
		self.failUnless(self.half_spam['ham'] == 0.5)
		self.failUnless(self.half_spam.arg_max() in ('spam', 'ham'))
		self.failUnless(len(self.half_spam.keys()) == 2)


	def test_scalar_multiplication(self):
		self.half_spam.normalize()

		self.half_spam = self.half_spam * 2.0

		self.failUnless(self.half_spam['spam'] == 1.0)
		self.failUnless(self.half_spam['ham'] == 1.0)
		self.failUnless(self.half_spam.arg_max() in ('spam', 'ham'))
		self.failUnless(len(self.half_spam.keys()) == 2)

		self.half_spam = self.half_spam * 0.5

		self.failUnless(self.half_spam['spam'] == 0.5)
		self.failUnless(self.half_spam['ham'] == 0.5)
		self.failUnless(self.half_spam.arg_max() in ('spam', 'ham'))
		self.failUnless(len(self.half_spam.keys()) == 2)

	def test_multiplication(self):
		bob = counter()
		bob['spam'] = 1.0
		bob['ham'] = 1.0
		self.half_spam = counter()
		self.half_spam['spam'] += 1
		self.half_spam['ham'] += 1
		self.half_spam.normalize()

		self.failUnless(bob['spam'] == 1.0)
		self.failUnless(bob['ham'] == 1.0)
		self.failUnless(bob.arg_max() in ('spam', 'ham'))
		self.failUnless(len(bob.keys()) == 2)

		jim = bob * self.half_spam
	
		self.failUnless(jim['spam'] == 0.5)
		self.failUnless(jim['ham'] == 0.5)
		self.failUnless(jim.arg_max() in ('spam', 'ham'))
		self.failUnless(len(jim.keys()) == 2)

	def test_subtraction(self):
		bob = counter()
		bob['spam'] = 1.0
		bob['ham'] = 1.0
		jim = counter()
		jim['spam'] = 0.5
		jim['ham'] = 0.5
		jim['tuna'] = 1.0

		bob -= jim

		self.failUnless(bob['spam'] == 0.5, self.half_spam)
		self.failUnless(bob['ham'] == 0.5)
		self.failUnless(bob.arg_max() in ('spam', 'ham'))
		self.failUnless(len(bob.keys()) == 3)
		
		foo = counter()
		foo['spam'] = 1.0
		foo['ham'] = 1.5
		foo['cheese'] = 3.5
		
		bar = foo - jim
		
		self.assertEqual(bar['spam'], 0.5)
		self.assertEqual(bar['ham'], 1.0)
		self.assertEqual(bar['cheese'], 3.5)
		self.assertEqual(bar['tuna'], -1.0)

		self.assertEqual(foo['spam'], 1.0)
		self.assertEqual(foo['ham'], 1.5)
		self.assertEqual(foo['cheese'], 3.5)
		self.failIf('tuna' in foo)

		self.assertEqual(jim['spam'], 0.5)
		self.assertEqual(jim['ham'], 0.5)
		self.assertEqual(jim['tuna'], 1.0)
		self.failIf('cheese' in jim)

	def test_log_normalize(self):
		log_third_spam = counter()
		log_third_spam['spam'] += log(1)
		log_third_spam['ham'] += log(2)

		log_third_spam.log_normalize()

		self.failUnless(log_third_spam['spam'] == log(1)-log(3))
		self.failUnless(log_third_spam['ham'] == log(2)-log(3))

	def test_in_place_multiply(self):
		amul = counter()
		amul['bob'] = 2
		amul['jim'] = 2
		bmul = counter()
		bmul['bob'] = 4

		amul *= bmul
		bmul *= amul

		self.failUnless(amul['bob'] == 8)
		self.failUnless(bmul['bob'] == 32)
		self.failUnless(amul['jim'] == 0)
		self.failUnless(bmul['jim'] == 0)

	def test_in_place_add(self):
		aadd = counter()
		aadd['bob'] = 2
		badd = counter()
		badd['bob'] = 4

		aadd += badd

		self.failUnless(aadd['bob'] == 6)
		self.failUnless(badd['bob'] == 4)

	def test_exercise_gc(self):
		base = counter()
		sub = counter()

		base['cat'] += 1
		base['dog'] += 1

		sub['cat'] = 0.001

		# Exercise the garbage collector - ideally we should be caching recently
		# released counters to optimize for this
		for i in xrange(10000):
			jim = base - sub

		self.assertEqual(jim['cat'], 0.999)
		self.assertEqual(jim['dog'], 1.0)
		self.assertEqual(len(jim.keys()), 2)

	def test_add_mul_comprehensive(self):
		# Testing default values
		for operation in ("__add__", "__mul__"):
			foo = counter()
			foo.default = float("-inf")
			foo['a'] = 1.0

			bar = counter()
			bar.default = float("-inf")
			bar['b'] = 2.0

			foofunc = getattr(foo, operation)
			barfunc = getattr(bar, operation)

			# Transitivity
			self.failUnless(foofunc(bar) == barfunc(foo), "%s != %s" % (foofunc(bar), barfunc(foo)))

			# Test that the values are correct
			bob = foofunc(bar)
			self.failUnless(bob['a'] == float("-inf"))
			self.failUnless(bob['b'] == float("-inf"))
			if operation == "__add__": val = float("-inf")
	 		elif operation == "__mul__": val = float("inf")
			self.failUnless(bob['missing'] == val)

			# Verify that the originals are unchanged
			self.failUnless(foo['a'] == 1.0 and foo['missing'] == float("-inf"))
			self.failUnless(bar['b'] == 2.0 and bar['missing'] == float("-inf"))

	def test_iadd_imul_comprehensive(self):
		# Testing default values for in-place ops
		for operation in ("__iadd__", "__imul__"):
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
			orig2 = copy(foo)
			orig2.default = foo.default

			foofunc(bar)
			barfunc(orig)
		
			# No side effects
			self.failUnless(orig == orig2)

			# Transitivity
			self.failUnless(foo == bar, "%s != %s" % (foo, bar))

			# Test that the values are correct
			self.failUnless(bar['a'] == float("-inf"))
			self.failUnless(bar['b'] == float("-inf"))
			if operation == "__iadd__": val = float("-inf")
	 		elif operation == "__imul__": val = float("inf")
			self.failUnless(bar['missing'] == val)

	def test_no_side_effects(self):
		foo = counter()
		bar = counter()
		foo['a'] += 2.0
		foo['b'] += 1.0

		foo2 = copy(foo)
		bar += foo
		self.failUnless(foo2 == foo)

if __name__ == "__main__":
	unittest.main()
