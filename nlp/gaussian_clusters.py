import itertools
import random
import sys

import rpy2.robjects as robjects

from counter import Counter
from countermap import CounterMap, outer_product
from crp import CRPGibbsSampler
from distributions import Gaussian

class GaussianClusterer(CRPGibbsSampler):
	def _cluster_log_probs(self, cluster, cluster_size, cluster_mean, cluster_covariance, new_point):
		""" Return the posterior, prior, and likelihood of new_point
		being in the cluster of size cluster_size centered at cluster_mean
		"""
		# the updated mean
		new_mean = (cluster_mean * cluster_size + new_point) / (cluster_size + 1)

		posterior_precision = self._prior_precision + self._cluster_precision
		# convex combination for mean
		posterior_mean = self._prior_mean * self._prior_precision
		posterior_mean += cluster_mean * self._cluster_precision
		posterior_mean /= posterior_precision

		posterior = Gaussian.log_prob(new_mean, posterior_mean, posterior_precision)
		# prior is keyed on the (potentially) updated params
		prior = Gaussian.log_prob(new_mean, self._prior_mean, self._prior_precision)
		likelihood = Gaussian.log_prob(new_point, new_mean, self._cluster_precision)

		return posterior, prior, likelihood

	def _sample_datum(self, datum):
		likelihoods = Counter(float("-inf"))
		priors = Counter(float("-inf"))
		posteriors = Counter(float("-inf"))
		sizes = Counter()

		# Regenerate all the cluster params (should be caching this,
		# not doing it inline)
		for c_idx, cluster in self._cluster_to_datum.iteritems():
			if not cluster:
				continue
			sizes[c_idx] = len(cluster)
			cluster_mean = sum(cluster) / float(sizes[c_idx])
			cluster_covariance = 1.0 / float(len(cluster) + 1) * sum(outer_product((pt - cluster_mean), (pt - cluster_mean)) for pt in cluster)

			posteriors[c_idx], priors[c_idx], likelihoods[c_idx] = self._cluster_log_probs(cluster, sizes[c_idx], cluster_mean, cluster_covariance, datum)

			if all(prob == float("-inf") for prob in (priors[c_idx], likelihoods[c_idx], posteriors[c_idx])):
				del priors[c_idx]
				del likelihoods[c_idx]
				del posteriors[c_idx]
				del sizes[c_idx]
				continue

		# Now generate probs for the new cluster
		# prefer to reuse an old cluster # if possible
		new_cluster = min([c for c, d in self._cluster_to_datum.iteritems() if not d], len(self._cluster_to_datum))

		sizes[new_cluster] = self._concentration

		# build a really lame covariance matrix for single points
		covariance = CounterMap()
		for axis in datum:
			covariance[axis] = 1.0

		posteriors[new_cluster], priors[new_cluster], likelihoods[new_cluster] = self._cluster_log_probs([], sizes[new_cluster], datum, covariance, datum)

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
		probs.exp()
		probs *= sizes

		# filter out nan
		for k, v in probs.items():
			if v != v:
				del probs[k]

		probs.normalize()

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

			# score => p(x, c)

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

	def __draw_cluster(self, r, points, color):
		# Draw the 95% interval as a circle around the center
		cmean = sum(points) / len(points)
		cdeviation = (sum((data - cmean)**2 for data in points) / (len(points))).total_count() / 2

		r.points(cmean['x'], cmean['y'], pch=21, cex=cdeviation, col=color)

	def plot(self, iteration, cluster_only=False):
		r = robjects.r

		if not cluster_only:
			r.png("likelihood-%d.png" % iteration)
			r.plot(robjects.IntVector(range(1, len(self._iteration_likelihoods) + 1)),
			robjects.FloatVector(self._iteration_likelihoods),
							     xlab="iteration", ylab="likelihood")
			r['dev.off']()

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

			self.__draw_cluster(r, cdata, color)
			
		r['dev.off']()


def points(means, std_dev, num_points=1000):
	points = []
	for _ in xrange(num_points):
		cluster = random.sample(means, 1)[0]
		point = Counter()
		point['x'] = random.gauss(cluster[0], std_dev)
		point['y'] = random.gauss(cluster[1], std_dev)
		points.append(point)
		
	return points

def xy_cnt(points):
	return [Counter(zip(('x', 'y'), p)) for p in points]

if __name__ == "__main__":
	clusters = 3

	# Prior params
	prior_mean = (50.0, 50.0)
	prior_std_dev = 250.0
	prior_precision = 1.0 / (prior_std_dev**2)

	# Draw the means from the prior params
	#means = [(random.gauss(prior_mean[0], prior_std_dev), random.gauss(prior_mean[1], prior_std_dev)) for _ in xrange(clusters)]
	means = [(70.0, 10.0), (70.0, 90.0), (10.0, 50.0)]

	prior_mean = xy_cnt([prior_mean])[0]
	prior_precision = Counter(prior_precision)
	mean_counters = xy_cnt(means)

	print "Cluster means: %s" % mean_counters

	# fixed variance
	std_dev = 10.0
	cluster_precision = Counter(1.0 / std_dev**2)

	problem = GaussianClusterer(points(means, std_dev/2.0, num_points=100), cluster_precision, prior_mean, prior_precision)
	problem.run(int(sys.argv[1]))

