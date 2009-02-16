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
		for f in expected_features:
			self.assertTrue(f in test_features)

		# verify we didn't extract any extras
		for f in test_features:
			self.assertTrue(f in expected_features)


if __name__ == "__main__":
	unittest.main()
