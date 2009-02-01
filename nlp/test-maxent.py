from itertools import chain
from math import exp, log
import time

from counter import Counter
import maxent
import maximumentropy
from countermap import CounterMap

import unittest

class MaximumEntropyClassifierTestToyProblem(unittest.TestCase):
	def setUp(self):
		self.training_data = (('cat', Counter((key, 1.0) for key in ('fuzzy', 'claws', 'small'))),
							  ('bear', Counter((key, 1.0) for key in ('fuzzy', 'claws', 'big'))),
							  ('cat', Counter((key, 1.0) for key in ('claws', 'medium'))))
		self.test_data = (('cat', Counter((key, 1.0) for key in ('claws', 'small'))),
						  ('bear', Counter((key, 1.0) for key in ('fuzzy',))))

		self.classifier = maximumentropy.MaximumEntropyClassifier()
		self.classifier.labels = set(key for key, _ in self.training_data)
		self.classifier.features = set(chain(features.iterkeys() for _, features in self.training_data))

	def test_unsmoothed(self):
		self.classifier.train_with_features(self.training_data, sigma=0.0, quiet=True)
		maxent_log_probs = self.classifier.get_log_probabilities(self.test_data[0][1])
		self.assertAlmostEqual(exp(maxent_log_probs['cat']), 1.0, 2)
		self.assertAlmostEqual(exp(maxent_log_probs['bear']), 0.0, 2)

	def test_smoothed(self):
		self.classifier.train_with_features(self.training_data, sigma=1.0, quiet=True)
		maxent_log_probs = self.classifier.get_log_probabilities(self.test_data[0][1])
		self.assertAlmostEqual(exp(maxent_log_probs['cat']), 0.73, 2)
		self.assertAlmostEqual(exp(maxent_log_probs['bear']), 0.27, 2)
		
#		print "Weights: %s" % classifier.weights
#		for test_datum in test_data:
#			maxent_log_probs = classifier.get_log_probabilities(test_datum[1])
#			for label in ['cat', 'bear']:
#				print "P[%s | %s] = %f" % (label, test_datum[1].keys(), exp(maxent_log_probs[label]))

class MaximumEntropyExpectedCountsTest(unittest.TestCase):
	def setUp(self):
		self.labeled_extracted_features = (('cat', Counter((key, 1.0) for key in ('fuzzy', 'claws', 'small'))),
										   ('bear', Counter((key, 1.0) for key in ('fuzzy', 'claws', 'big'))),
										   ('cat', Counter((key, 1.0) for key in ('claws', 'medium'))))

		self.labels = set(label for label, _ in self.labeled_extracted_features)

	def test_fast_slow_equal(self):
		weights = CounterMap()
		weights['cat'] = Counter((key, 1.0) for key in ('fuzzy', 'claws', 'small', 'medium', 'large'))
		weights['bear'] = Counter((key, 1.0) for key in ('fuzzy', 'claws', 'small', 'medium', 'large'))

		log_probs = [maxent.get_log_probabilities(datum[1], weights, self.labels) for datum in self.labeled_extracted_features]

		slow_expectation = maximumentropy.slow_expected_counts(self.labeled_extracted_features, self.labels, log_probs)
		fast_expectation = maxent.get_expected_counts(self.labeled_extracted_features, self.labels, log_probs, CounterMap())

		self.assertEqual(slow_expectation, fast_expectation)

		# And try again with different weights
		weights['cat'] = Counter((key, 1.0) for key in ('fuzzy', 'claws', 'small', 'medium'))
		weights['bear'] = Counter((key, 1.0) for key in ('fuzzy', 'claws', 'big'))

		log_probs = [maxent.get_log_probabilities(datum[1], weights, self.labels) for datum in self.labeled_extracted_features]

		slow_expectation = maximumentropy.slow_expected_counts(self.labeled_extracted_features, self.labels, log_probs)
		fast_expectation = maxent.get_expected_counts(self.labeled_extracted_features, self.labels, log_probs, CounterMap())

		self.assertEqual(slow_expectation, fast_expectation)

class MaximumEntropyLogProbsTest(unittest.TestCase):
	def setUp(self):
		self.features = Counter((key, 1.0) for key in ['warm', 'fuzzy'])

		self.weights = CounterMap()
		self.weights['dog'] = Counter({'warm' : 2.0, 'fuzzy' : 0.5})
		self.weights['cat'] = Counter({'warm' : 0.5, 'fuzzy' : 2.0})

		self.labels = set(self.weights.iterkeys())
		self.logp = maxent.get_log_probabilities(self.features, self.weights, self.labels)

	def test_fast_slow_equal(self):
		slow_logp = maximumentropy.slow_log_probs(self.features, self.weights, self.labels)

		self.assertEqual(self.logp, slow_logp)

	def test_logp_is_probability_distribution(self):
		"""
		Verify that all log probs are <= 0 and total probability is 1.0
		"""
		self.assertTrue(max(self.logp.itervalues()) <= 0.0)
		self.assertAlmostEqual(sum(exp(val) for val in self.logp.itervalues()), 1.0)

	def test_basic_values(self):
		"""
		Are the log probs as expected?
		"""
		self.assertAlmostEqual(exp(self.logp['cat']), 0.5)
		self.assertAlmostEqual(exp(self.logp['dog']), 0.5)

	def test_single_label(self):
		weights = CounterMap()
		weights['dog'] = Counter({'warm' : 2.0, 'fuzzy' : 0.5})
		labels = set(weights.iterkeys())
		logp = maxent.get_log_probabilities(self.features, weights, labels)

		self.assertEqual(logp['dog'], 0.0)

	def test_extraneous_label(self):
		weights = CounterMap()
		weights['dog'] = Counter({'warm' : 2.0, 'fuzzy' : 0.5})
		labels = set(weights.iterkeys())
		logp = maxent.get_log_probabilities(self.features, weights, labels)

		self.assertEqual(logp['cat'], float('-inf'))

	def test_zero_weight(self):
		weights = CounterMap()
		weights['dog'] = Counter({'warm' : 2.0})
		labels = set(weights.iterkeys())
		logp = maxent.get_log_probabilities(self.features, weights, labels)

		self.assertEqual(logp['dog'], 0.0)
		
	def test_uneven_weights(self):
		weights = CounterMap()
		weights['dog'] = Counter({'warm' : 2.0, 'fuzzy' : 1.0})
		weights['cat'] = Counter({'warm' : 1.0, 'fuzzy' : 1.0})
		labels = set(weights.iterkeys())
		logp = maxent.get_log_probabilities(self.features, weights, labels)

		# construct scores
		scores = Counter()
		scores['dog'] = 2.0 * 1.0 + 1.0 * 1.0
		scores['cat'] = 1.0 * 1.0 + 1.0 * 1.0
		scores.log_normalize()

		# check scores explicitly
		self.assertAlmostEqual(scores['dog'], log(0.731), 3)
		self.assertAlmostEqual(scores['cat'], log(0.269), 3)

		# check that log probs is correct
		self.assertEqual(logp['dog'], scores['dog'])
		self.assertEqual(logp['cat'], scores['cat'])

	def test_performance(self):
		"""
		C api should be faster than python API (this is potentialy flakey, depending on system load patterns)
		"""
		start = time.time()
		for i in xrange(100000):
			test = maximumentropy.slow_log_probs(self.features, self.weights, self.labels)

		slow_time = time.time() - start

		start = time.time()
		for i in xrange(100000):
			test = maxent.get_log_probabilities(self.features, self.weights, self.labels)

		fast_time = time.time() - start

		self.assertTrue(fast_time < slow_time)

if __name__ == "__main__":
	unittest.main()
