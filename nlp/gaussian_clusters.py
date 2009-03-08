from collections import defaultdict
from random import Random
import sys

from counter import Counter
from crp import CRPGibbsSampler

class GaussianClusterer(object):
	def train(self):
		rand = Random()
		rand.seed()
		clusters = 2
		dims = 2
		points = 100
		data = []
		data_to_cluster = dict()

#		means = [tuple(rand.uniform(0, 100) for _ in xrange(dims)) for
#		_ in xrange(clusters)]
		means = [(10.0, 10.0), (90.0, 90.0)]
		mean_counters = [Counter((('x', x), ('y', y))) for (x, y) in means]

		cluster_mean = sum(mean_counters, Counter()) / len(means)
		lm = len(means) - 1
		cluster_precision = Counter((k, lm / v) for k,v in sum(((m - cluster_mean) * (m - cluster_mean) for m in mean_counters), Counter()).iteritems())

		cluster_to_data = defaultdict(list)
		for _ in xrange(points):
			cluster = rand.sample(means, 1)[0]
			point = Counter()
			point['x'] = rand.gauss(cluster[0], 4.0)
			point['y'] = rand.gauss(cluster[1], 4.0)
			data.append(point)
			data_to_cluster[tuple(point.values())] = cluster
			cluster_to_data[cluster].append(point)

		for cluster, cdata in cluster_to_data.iteritems():
			print "Cluster (size %d): %s" % (len(cdata), sum(cdata, Counter()) / len(cdata))
		data = dict(enumerate(data))
		# and hand over work to the sampler
		sampler = CRPGibbsSampler(data, mh_mean=cluster_mean, mh_precision=cluster_precision)

		sampler.gibbs(int(sys.argv[1]))

	def run(self):
		# generate random means and sample points from them
		self.train()

if __name__ == "__main__":
	problem = GaussianClusterer()
	problem.run()
