import itertools
import random
import sys

import rpy2.robjects as robjects

from counter import Counter
from crp import CRPGibbsSampler
from distributions import Gaussian

class GaussianClusterer(CRPGibbsSampler):
	def _sample_datum(self, datum):
		likelihoods = Counter(float("-inf"))
		priors = Counter(float("-inf"))
		posteriors = Counter(float("-inf"))
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

		sizes[new_cluster] = self._concentration

		posterior_precision = self._prior_precision + self._cluster_precision
		posterior_mean = self._prior_mean * self._prior_precision
		posterior_mean += datum * self._cluster_precision
		posterior_mean /= posterior_precision

		posteriors[new_cluster] = Gaussian.log_prob(datum, posterior_mean, posterior_precision)
		priors[new_cluster] = Gaussian.log_prob(datum, self._prior_mean, self._prior_precision)
		likelihoods[new_cluster] = Gaussian.log_prob(datum, datum, self._cluster_precision)

		for dist in priors, likelihoods, posteriors:
			if not all(v <= 0.0 for v in dist.itervalues()):
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

	def __init__(self, data, cluster_precision, prior_mean, prior_precision):
		data = dict(enumerate(data))

		self._cluster_precision = cluster_precision
		self._prior_mean = prior_mean
		self._prior_precision = prior_precision

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

	def plot(self, iteration, cluster_only=False):
		r = robjects.r

		if not cluster_only:
			# 		r.png("likelihood-%d.png" % iteration)
			# 		r.plot(robjects.IntVector(range(1, len(self._iteration_likelihoods) + 1)),
			# 			   robjects.FloatVector(self._iteration_likelihoods),
			# 			   xlab="iteration", ylab="likelihood")
			# 		r['dev.off']()

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

#			print "Cluster (size %d): %s" % (len(cdata), sum(cdata) / len(cdata))
#			print color
			r.points(points_x, points_y, col=color)

			cmean = sum(cdata) / len(cdata)
			r.points(cmean['x'], cmean['y'], pch=21, cex=4.0, col=color)

		r['dev.off']()


def points(means, std_dev, prior_mean, prior_precision, num_points=1000):
	points = []
	for _ in xrange(num_points):
		cluster = random.sample(means, 1)[0]
		point = Counter()
		point['x'] = random.gauss(cluster[0], std_dev)
		point['y'] = random.gauss(cluster[1], std_dev)
		points.append(point)
		
	return points

if __name__ == "__main__":
	cluster = 3
	dims = 2
#	means = [tuple(rand.uniform(0, 100) for _ in xrange(dims)) for
#	_ in xrange(clusters)]
	means = [(10.0, 10.0), (10.0, 180.0), (105.0, 45.0)]
	mean_counters = [Counter((('x', x), ('y', y))) for (x, y) in means]

	prior_mean = sum(mean_counters) / len(mean_counters)
	prior_precision = len(means) / sum(((m - prior_mean) ** 2 for m in mean_counters))
	# fixed variance
	std_dev = 5.0
	cluster_precision = Counter(1.0 / std_dev**2) / 100

	problem = GaussianClusterer(points(means, std_dev, prior_mean, prior_precision, num_points=1000), cluster_precision, prior_mean, prior_precision)
	problem.run(int(sys.argv[1]))

