from collections import defaultdict
from math import exp
from random import Random
import sys

from counter import Counter
from crp import CRPGibbsSampler

class GaussianClusterer(CRPGibbsSampler):
	def _sample_datum(self, datum):
		likelihoods = Counter(float("-inf"))
		priors = Counter(float("-inf"))
		posteriors = Counter(float("-inf"))
		sizes = Counter()

		print "*** start: %s ***" % repr(self._cluster_to_datum.items())
		for c_idx, cluster in self._cluster_to_datum.iteritems():
			if not cluster:
				continue
			sizes[c_idx] = len(cluster)
			cluster_mean = sum(cluster, Counter()) / float(sizes[c_idx])

			mean = self._mh_mean * self._mh_tau
			mean += cluster_mean * self._cluster_tau
			mean /= (self._mh_tau + sizes[c_idx] * self._cluster_tau)
			precision = self._mh_tau + sizes[c_idx] * self._cluster_tau

			diff = datum - mean
			point_score = diff * diff
			point_score *= precision * -0.5
			posteriors[c_idx] = point_score.total_count()

			# prior is keyed on the (potentially) updated params
			prior = self._mh_tau * -0.5
			prior_cluster_mean = cluster_mean + (datum / sizes[c_idx])
			prior *= (prior_cluster_mean - self._mh_mean)
			prior *= (prior_cluster_mean - self._mh_mean)
			priors[c_idx] = sum(prior.itervalues())

			likelihoods[c_idx] = -0.5 * self._cluster_tau
			likelihoods[c_idx] *= sum(((datum - cluster_mean) * (datum - cluster_mean)).itervalues())

			sizes[c_idx] = sizes[c_idx]

		# Now generate probs for the new cluster
		# prefer to reuse an old cluster # if possible
		new_cluster = min([c for c, d in self._cluster_to_datum.iteritems() if not d], len(self._cluster_to_datum))

		mean = self._mh_mean * self._mh_tau
		mean += datum * self._cluster_tau
		mean /= (self._mh_tau + self._cluster_tau)
		precision = self._cluster_tau
		diff = datum - mean
		point_score = diff * diff
		point_score *= precision * -0.5
		posteriors[new_cluster] = point_score.total_count()
		print " New cluster (%d) posterior: %s" % (new_cluster, posteriors[new_cluster])

		priors[new_cluster] = sum((self._mh_tau * (datum - self._mh_mean) * (datum - self._mh_mean) * -0.5).itervalues())
		likelihoods[new_cluster] = 0.0
		sizes[new_cluster] = self._concentration

		likelihoods.log_normalize()
		print "  Likelihoods", likelihoods
		priors.log_normalize()
		print "  Priors", priors
		posteriors.log_normalize()
		print "  Posteriors:", posteriors
		probs = likelihoods + priors - posteriors
		print " Total probs: %s" % probs
		probs.exp()
		probs.normalize()

		probs *= sizes
		print " Total probs: %s" % probs

		return probs.sample()

	def log_likelihood(self):
		# evaluate the likelihood of the labelling (which is,
		# conveniently, just the likelihood of the current mixture
		# model)

		# FIXME: This should really be cached for the last invocation
		score = Counter(default=0.0)
		for c_idx, cluster in self._cluster_to_datum.iteritems():
			if not cluster: continue
			# Evaluate the likelihood of each individual cluster
			cluster_size = len(cluster)
			# The mean of the data points belonging to this cluster
			cluster_datum_mean = sum(cluster) / cluster_size
			# The MLE of the mean of the cluster given the data points and the prior
			cluster_mle_mean = cluster_datum_mean

			likelihood = cluster_mle_mean * cluster_mle_mean
			x = (cluster_size * self._cluster_tau + self._mh_tau)
			likelihood *= x * x
			score -= likelihood

			likelihood = cluster_size * self._cluster_tau * cluster_datum_mean
			likelihood += self._mh_tau * self._mh_mean
			likelihood *= 2 * cluster_mle_mean
			score -= likelihood

		# for the gaussian the dimensions are independent so we should
		# just be able to combine them directly
		return sum(score.itervalues())

	def __init__(self):
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
		cluster_precision = Counter((k, lm / v) for k, v in sum(((m - cluster_mean) * (m - cluster_mean) for m in mean_counters), Counter()).iteritems())

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
		super(GaussianClusterer, self).__init__(data, mh_mean=cluster_mean, mh_precision=cluster_precision)

	def run(self, iterations):
		# generate random means and sample points from them
		self.gibbs(iterations)

if __name__ == "__main__":
	problem = GaussianClusterer()
	problem.run(int(sys.argv[1]))
