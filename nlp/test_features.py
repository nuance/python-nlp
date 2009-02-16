import unittest

import features

class NGramTest(unittest.TestCase):
	def test_three_grams(self):
		test_string = "hello"
		start = "<START>"
		stop = "<STOP>"

		test_features = set(tuple(x) for x in features.ngrams(test_string, 3, start, stop))
		expected_features = set(tuple(x) for x in ([start, start, 'h'], [start, 'h'], ['h'],
												   [start, 'h', 'e'], ['h', 'e'], ['e'],
												   ['h', 'e', 'l'], ['e', 'l'], ['l'],
												   ['e', 'l', 'l'], ['l', 'l'], ['l'],
												   ['l', 'l', 'o'], ['l', 'o'], ['o'],
												   ['l', 'o', stop], ['o', stop],
												   ['o', stop, stop]))

		# the equality test is split apart so it's easier to debug

		# verify we extracted all the features we wanted

	def test_one_gram(self):
		test_string = "hello"

		test_features = set(tuple(x) for x in features.ngrams(test_string, 1))
		expected_features = set(tuple(x) for x in "hello")

		for f in expected_features:
			self.assertTrue(f in test_features)
		for f in test_features:
			self.assertTrue(f in expected_features)


class ContextTest(unittest.TestCase):
	def test_three_context(self):
		test_string = "godspeed"
		test_features = set(tuple(x) for x in features.contexts(test_string, context_size=3))
		expected_features = set([(('g', 'o', 'd'), 's', ('p', 'e', 'e')),
								 (('o', 'd', 's'), 'p', ('e', 'e', 'd'))])
		self.assertEqual(test_features, expected_features)

	def test_one_context(self):
		test_string = "hello"
		test_features = set(tuple(x) for x in features.contexts(test_string, context_size=1))
		expected_features = set([(('h',), 'e', ('l',)), (('l',), 'l', ('o',)), (('e',), 'l', ('l',))])
		self.assertEqual(test_features, expected_features)


if __name__ == "__main__":
	unittest.main()
