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

		self.sampler = CRPGibbsSampler(self.points, mh_mean=Counter((('x', 5.0), ('y', 5.0))), mh_precision=1.0, cluster_precision=0.25)

	def test_add_self(self):
		# Shouldn't be initialized yet...
		self.assertEqual(self.sampler.log_likelihood(), 0.0)

		self.sampler._add_datum(1, self.points[1], 1)
		self.sampler._add_datum(2, self.points[2], 2)
		
		self.assertEqual(self.sampler.log_likelihood(), 12.0625 * 2 + 306.25 * 2)
		self.assertEqual(len(self.sampler._cluster_to_datum), 2)
		self.assertEqual(self.sampler._cluster_to_datum[1], [self.points[1]])
		self.assertEqual(self.sampler._cluster_to_datum[2], [self.points[2]])

		self.sampler._remove_datum(1, self.points[1])
		self.sampler._add_datum(1, self.points[1], 2)

		self.assertEqual(self.sampler.log_likelihood(), 306.625)
		self.assertEqual(len([c for c, v in self.sampler._cluster_to_datum.iteritems() if v]), 1)
		self.assertEqual(sorted(self.sampler._cluster_to_datum[2]), sorted(self.points.itervalues()))

if __name__ == "__main__":
	unittest.main()
