import unittest

from counter import Counter
from crp import CRPGibbsSampler

class CRPGibbsSamplerTest(unittest.TestCase):
	def setUp(self):
		# two separate points, verify that the likelihoods are correct
		# and the sampling is sound

		self.points = dict()
		self.points[1] = Counter()
		self.points[1]['x'] = 1.0
		self.points[1]['y'] = 1.0
		self.points[2] = Counter()
		self.points[2]['x'] = 10.0
		self.points[2]['y'] = 10.0

		self.sampler = CRPGibbsSampler(self.points)

	def test_simple_sample_datum(self):
		self.sampler._sample_datum = lambda datum: 0
		self.sampler.log_likelihood = lambda: 0.0

		self.sampler.gibbs(1)

		self.assertTrue(all(v == 0 for v in self.sampler._datum_to_cluster.itervalues()))
		self.assertTrue(all(len(c) == 0 for idx, c in self.sampler._cluster_to_datum.iteritems() if idx != 0), self.sampler._cluster_to_datum)
		self.assertEqual(len(self.sampler._cluster_to_datum[0]), 2)

		self.sampler.gibbs(1)

		self.assertEqual(self.sampler._cluster_count[0], 1)
		self.assertEqual(self.sampler._iteration_likelihoods[0], 0.0)

	def test_two_cluster_sample_datum(self):
		self.sampler._sample_datum = lambda datum: 0 if datum == self.points[1] else 1
		self.sampler.log_likelihood = lambda: 0.0

		self.sampler.gibbs(1)
		self.assertEqual(self.sampler._datum_to_cluster[1], 0)
		self.assertEqual(self.sampler._datum_to_cluster[2], 1)

		self.assertEqual(len(self.sampler._cluster_to_datum[0]), 1)
		self.assertEqual(len(self.sampler._cluster_to_datum[1]), 1)

		# invert the sampler
		self.sampler._sample_datum = lambda datum: 1 if datum == self.points[1] else 0
		self.sampler.gibbs(1)

		self.assertEqual(self.sampler._cluster_count[0], 2)
		self.assertEqual(self.sampler._iteration_likelihoods[0], 0.0)

		# Assert they switched clusters
		self.assertEqual(self.sampler._datum_to_cluster[1], 1)
		self.assertEqual(self.sampler._datum_to_cluster[2], 0)

		self.assertEqual(len(self.sampler._cluster_to_datum[0]), 1)
		self.assertEqual(len(self.sampler._cluster_to_datum[1]), 1)

	def _test_add_self(self):
		# Shouldn't be initialized yet...
		self.assertEqual(self.sampler.log_likelihood(), 0.0)

		self.sampler._add_datum(1, self.points[1], 1)
		self.sampler._add_datum(2, self.points[2], 2)
		
		self.assertEqual(self.sampler.log_likelihood(), -(12.0625 * 2 + 306.25 * 2))
		self.assertEqual(len(self.sampler._cluster_to_datum), 2)
		self.assertEqual(self.sampler._cluster_to_datum[1], [self.points[1]])
		self.assertEqual(self.sampler._cluster_to_datum[2], [self.points[2]])

		self.sampler._remove_datum(1, self.points[1])
		self.sampler._add_datum(1, self.points[1], 2)

		self.assertEqual(self.sampler.log_likelihood(), -306.625)
		self.assertEqual(len([c for c, v in self.sampler._cluster_to_datum.iteritems() if v]), 1)
		self.assertEqual(sorted(self.sampler._cluster_to_datum[2]), sorted(self.points.itervalues()))

if __name__ == "__main__":
	unittest.main()
