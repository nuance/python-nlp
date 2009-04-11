from collections import defaultdict
import itertools
from random import Random
import sys

import rpy2.robjects as robjects

from counter import Counter
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

			posterior_precision = self._prior_precision + self._cluster_precision
#			raise Exception(self._prior_precision, self._cluster_precision, posterior_precision)
			# convex combination for mean
			posterior_mean = self._prior_mean * self._prior_precision
			posterior_mean += cluster_mean * self._cluster_precision
			posterior_mean /= posterior_precision

			posteriors[c_idx] = Gaussian.log_prob(new_mean, posterior_mean, posterior_precision)
			# prior is keyed on the (potentially) updated params
			priors[c_idx] = Gaussian.log_prob(new_mean, self._prior_mean, self._prior_precision)
			likelihoods[c_idx] = Gaussian.log_prob(datum, new_mean, self._cluster_precision)

		# Now generate probs for the new cluster
		# prefer to reuse an old cluster # if possible
		new_cluster = min([c for c, d in self._cluster_to_datum.iteritems() if not d], len(self._cluster_to_datum))
#		print " New cluster: %d" % (new_cluster)

		posterior_precision = self._prior_precision + self._cluster_precision
		posterior_mean = self._prior_mean * self._prior_precision
		posterior_mean += datum * self._cluster_precision
		posterior_mean /= posterior_precision

		posteriors[new_cluster] = Gaussian.log_prob(datum, posterior_mean, posterior_precision)
		priors[new_cluster] = Gaussian.log_prob(datum, self._prior_mean, self._prior_precision)
		likelihoods[new_cluster] = Gaussian.log_prob(datum, datum, self._cluster_precision)
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
			score += Gaussian.log_prob(cluster_datum_mean, self._prior_mean, self._prior_precision)
			# p(x|c)
			score += sum(Gaussian.log_prob(datum, cluster_datum_mean, self._cluster_precision) for datum in cluster)

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

		self._prior_mean = sum(mean_counters) / len(mean_counters)
		lm = len(means) - 1
		self._prior_precision = lm / sum(((m - self._prior_mean) ** 2 for m in mean_counters)) / 15
		# fixed variance
		variance = 4.0
		self._cluster_precision = Counter(1.0 / variance ** 2)

		cluster_to_data = defaultdict(list)
		for _ in xrange(points):
			cluster = rand.sample(means, 1)[0]
			point = Counter()
			point['x'] = rand.gauss(cluster[0], variance)
			point['y'] = rand.gauss(cluster[1], variance)
			data.append(point)
			data_to_cluster[tuple(point.values())] = cluster
			cluster_to_data[cluster].append(point)

		for cluster, cdata in cluster_to_data.iteritems():
			print "Cluster (size %d): %s" % (len(cdata), sum(cdata) / len(cdata))
			
		data = dict(enumerate(data))

		self._max_x = max(v['x'] for v in data.itervalues())
		self._min_x = min(v['x'] for v in data.itervalues())
		self._max_y = max(v['y'] for v in data.itervalues())
		self._min_y = min(v['y'] for v in data.itervalues())

		# and hand over work to the sampler
		super(GaussianClusterer, self).__init__(data)

	def run(self, iterations):
		# generate random means and sample points from them
		self.gibbs(iterations)
		self.plot(iterations)

	def plot(self, iteration):
		r = robjects.r
		r.png("likelihood-%d.png" % iteration)
		r.plot(robjects.IntVector(range(1, len(self._iteration_likelihoods) + 1)), robjects.FloatVector(self._iteration_likelihoods), xlab="iteration", ylab="likelihood")
		r['dev.off']()

		r = robjects.r
		r.png("cluster-count-%d.png" % iteration)
		r.plot(robjects.IntVector(range(1, len(self._cluster_count) + 1)), robjects.FloatVector(self._cluster_count), xlab="iteration", ylab="# clusters")
		r['dev.off']()

		r.png("test-%d.png" % iteration)
		r.plot([self._min_x - 1.0, self._max_x + 1.0],
			   [self._min_y - 1.0, self._max_y + 1.0],
			   xlab="x", ylab="y", col="white")

		self._cluster_to_datum = dict((cluster, data) for cluster, data in self._cluster_to_datum.iteritems() if data)

		colors = itertools.cycle(("red", "green", "blue", "black", "purple", "orange"))
		for (cluster, cdata), color in zip(self._cluster_to_datum.iteritems(), colors):
			points_x = robjects.FloatVector([point['x'] for point in cdata])
			points_y = robjects.FloatVector([point['y'] for point in cdata])

			print "Cluster (size %d): %s" % (len(cdata), sum(cdata) / len(cdata))
			print color
			r.points(points_x, points_y, col=color)

			cmean = sum(cdata) / len(cdata)
			r.points(cmean['x'], cmean['y'], pch=21, cex=4.0, col=color)

		r['dev.off']()

if __name__ == "__main__":
	problem = GaussianClusterer()
	problem.run(int(sys.argv[1]))
