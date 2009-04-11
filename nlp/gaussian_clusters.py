from collections import defaultdict
from math import log, pi, sqrt
from random import Random
import sys

from counter import Counter, counter_map
from countermap import CounterMap
from crp import CRPGibbsSampler
from distributions import Gaussian

class GaussianClusterer(CRPGibbsSampler):
	def _sample_datum(self, datum):
		likelihoods = CounterMap(float("-inf"))
		priors = CounterMap(float("-inf"))
		posteriors = CounterMap(float("-inf"))
		sizes = Counter()

		for c_idx, cluster in self._cluster_to_datum.iteritems():
			if not cluster:
				continue
			sizes[c_idx] = len(cluster)
			cluster_mean = sum(cluster) / float(sizes[c_idx])

			# the updated mean
			new_mean = (cluster_mean * sizes[c_idx] + datum) / (sizes[c_idx] + 1)

			posterior_precision = self._mh_tau + self._cluster_tau
#			raise Exception(self._mh_tau, self._cluster_tau, posterior_precision)
			# convex combination for mean
			posterior_mean = self._mh_mean * self._mh_tau
			posterior_mean += cluster_mean * self._cluster_tau
			posterior_mean /= posterior_precision

			posteriors[c_idx] = GaussianDistribution.log_prob(new_mean, posterior_mean, posterior_precision)
			# prior is keyed on the (potentially) updated params
			priors[c_idx] = GaussianDistribution.log_prob(new_mean, self._mh_mean, self._mh_tau)
			likelihoods[c_idx] = GaussianDistribution.log_prob(datum, new_mean, self._cluster_tau)

		# Now generate probs for the new cluster
		# prefer to reuse an old cluster # if possible
		new_cluster = min([c for c, d in self._cluster_to_datum.iteritems() if not d], len(self._cluster_to_datum))
#		print " New cluster: %d" % (new_cluster)

		posterior_precision = self._mh_tau + self._cluster_tau
		posterior_mean = self._mh_mean * self._mh_tau
		posterior_mean += datum * self._cluster_tau
		posterior_mean /= posterior_precision

		posteriors[new_cluster] = GaussianDistribution.log_prob(datum, posterior_mean, posterior_precision)
		priors[new_cluster] = GaussianDistribution.log_prob(datum, self._mh_mean, self._mh_tau)
		likelihoods[new_cluster] = GaussianDistribution.log_prob(datum, datum, self._cluster_tau)
		sizes[new_cluster] = self._concentration

		for dist in priors, likelihoods, posteriors:
			if not all(all(v <= 0.0 for v in scores.itervalues()) for scores in dist.itervalues()):
				print "Not a log distribution: %s" % dist
				print "(new cluster %d)" % new_cluster
				print datum
				for k, scores in dist.iteritems():
					if all(v <= 0.0 for v in scores.itervalues()): continue
					print "error on cluster %d" % k
					print "posteriors: %r" % posteriors[k]
					print "priors: %r" % priors[k]
					print "likelihoods: %r" % likelihoods[k]
					print "sizes: %r" % sizes[k]
				raise Exception()

		probs = likelihoods + priors - posteriors
#		print " Total probs: %s" % probs
		probs.exp()
		probs.normalize()

		probs = Counter((k, v.total_count()) for k, v in probs.iteritems())
		probs *= sizes
		probs.normalize()
#		print " Total probs: %s" % probs

		assert all(0.0 <= p <= 1.0 for p in probs.itervalues()), "Not a distribution: %s" % probs
		return probs.sample()

	def log_likelihood(self):
		# evaluate the likelihood of the labelling (which is,
		# conveniently, just the likelihood of the current mixture
		# model)

		# FIXME: This should really be cached for the last invocation
		score = Counter()
		for c_idx, cluster in self._cluster_to_datum.iteritems():
			if not cluster: continue
			# Evaluate the likelihood of each individual cluster
			cluster_size = len(cluster)
			# The mean of the data points belonging to this cluster
			cluster_datum_mean = sum(cluster) / cluster_size

			# p(c)
			score += GaussianDistribution.log_prob(cluster_datum_mean, self._mh_mean, self._mh_tau)
			# p(x|c)
			score += sum(GaussianDistribution.log_prob(datum, cluster_datum_mean, self._cluster_tau) for datum in cluster)

		# for the gaussian the dimensions are independent so we should
		# just be able to combine them directly
		return score.total_count()

	def __init__(self):
		rand = Random()
		rand.seed()
		clusters = 3
		dims = 2
		points = 1000
		data = []
		data_to_cluster = dict()

#		means = [tuple(rand.uniform(0, 100) for _ in xrange(dims)) for
#		_ in xrange(clusters)]
		means = [(50.0, 60.0), (90.0, 90.0), (70.0, 40.0)]
		mean_counters = [Counter((('x', x), ('y', y))) for (x, y) in means]

		cluster_mean = sum(mean_counters) / len(means)
		lm = len(means) - 1
		cluster_precision = lm / sum(((m - cluster_mean) ** 2 for m in mean_counters)) / 15

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
			print "Cluster (size %d): %s" % (len(cdata), sum(cdata) / len(cdata))
			
		data = dict(enumerate(data))
		# and hand over work to the sampler
		super(GaussianClusterer, self).__init__(data, mh_mean=cluster_mean, mh_precision=cluster_precision)

	def run(self, iterations):
		# generate random means and sample points from them
		self.gibbs(iterations)
		self.plot(iterations)

if __name__ == "__main__":
	problem = GaussianClusterer()
	problem.run(int(sys.argv[1]))
