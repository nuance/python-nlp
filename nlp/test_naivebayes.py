import unittest
from math import log

from naivebayes import NaiveBayesClassifier
from counter import Counter

class NaiveBayesClassifierTest(unittest.TestCase):
	def test_single_training_data(self):
		classifier = NaiveBayesClassifier()
		classifier.train((('A', 'a'),))

		self.failUnless(classifier.label('a') == 'A')
		distribution = classifier.label_distribution('a')
		self.failUnlessEqual(len(distribution), 1)
		self.failUnless('A' in distribution)
		self.failUnless(distribution['A'] == 0.0, distribution)

	def test_single_class_training_data(self):
		classifier = NaiveBayesClassifier()
		classifier.train((('A', 'a'),('A', 'a'),('A', 'a')))

		self.failUnless(classifier.label('a') == 'A')
		distribution = classifier.label_distribution('a')
		self.failUnlessEqual(len(distribution), 1)
		self.failUnless('A' in distribution)
		self.failUnless(distribution['A'] == 0.0, distribution)

	def test_single_class_mixed_training_data(self):
		classifier = NaiveBayesClassifier()
		classifier.train((('A', 'a'),('A', 'a'),('B', 'a')))

		self.failUnless(classifier.label('a') == 'A')
		distribution = classifier.label_distribution('a')
		self.failUnlessEqual(len(distribution), 2)
		self.failUnless('A' in distribution)

		correct_distribution = Counter()
		correct_distribution['A'] = (2.0 / 3.0)**3
		correct_distribution['B'] = (1.0 / 3.0)**3
		correct_distribution.normalize()
		correct_distribution.log()

		self.failUnlessAlmostEqual(distribution['A'], correct_distribution['A'])
		self.failUnlessAlmostEqual(distribution['B'], correct_distribution['B'])

if __name__ == "__main__":
	unittest.main()
