from collections import defaultdict
import datetime
import itertools

import rpy2.robjects as robjects

from countermap import CounterMap
from counter import Counter

class CRPGibbsSampler(object):
	def __init__(self, data, gibbs_iterations=1, cluster_precision=0.25, mh_mean=Counter(default=1.0), mh_precision=1.0):
		"""
		data: for now, counters of score-for-context (HUGE cardinality)
		gibbs_iterations: should be a number >= 1, large enough to
		ensure the chain is converged given updated params
		"""
		self._gibbs_iterations = gibbs_iterations
		self._data = data
		self._max_x = max(v['x'] for v in data.itervalues())
		self._min_x = min(v['x'] for v in data.itervalues())
		self._max_y = max(v['y'] for v in data.itervalues())
		self._min_y = min(v['y'] for v in data.itervalues())

		self._concentration = 0.9

		# fixed variance
		self._cluster_tau = cluster_precision

		# hyper-params for the mean
		self._mh_tau = mh_precision
		self._mh_mean = mh_mean

		# These will be learned during sampling
		self._datum_to_cluster = dict()
		self._cluster_to_datum = defaultdict(list)
		self._cluster_counts = CounterMap(default=float("-inf"))

		self._iteration_likelihoods = []
		self._cluster_count = []

	def _sample_datum(self, datum):
		probs = Counter()

		for cluster in self._cluster_to_datum:
			cluster_size = len(self._cluster_to_datum[cluster])
			cluster_mean = sum(self._cluster_to_datum[cluster], Counter()) / cluster_size
			mean = self._mh_mean * self._mh_tau
			mean += cluster_mean * self._cluster_tau
			mean /= (self._mh_tau + cluster_size * self._cluster_tau)
			precision = self._mh_tau + cluster_size * self._cluster_tau

			diff = datum - mean
			point_score = diff * diff
			point_score *= 0.5 * precision
			probs[cluster] = point_score.total_count()

		probs[len(self._cluster_to_datum)] = self._concentration
		probs.normalize()

		return probs.sample()

	def _add_datum(self, name, datum, cluster):
		self._datum_to_cluster[name] = cluster
		self._cluster_to_datum[cluster].append(datum)

	def _remove_datum(self, name, datum):
		cluster = self._datum_to_cluster.get(name)
		if cluster == None: return

		cluster = self._cluster_to_datum[cluster].remove(datum)
		del self._datum_to_cluster[name]

	def gibbs(self, iterations=None):
		# use gibbs sampling to find a sufficiently good labelling
		# starting with the current parameters and iterate
		if not iterations:
			iterations = self._gibbs_iterations

		for iteration in xrange(iterations):
			if iteration % 1000 == 0:
				print "*** Iteration %d starting (%s) ***" % (iteration, datetime.datetime.now())

			if self._cluster_to_datum:
				self._iteration_likelihoods.append(self.log_likelihood())
				self._cluster_count.append(len([c for c, v in self._cluster_to_datum.iteritems() if v]))
				if iteration % 1000 == 0:
					print "    Clusters: %d" % self._cluster_count[-1]
					print "    Likelihood: %f" % self._iteration_likelihoods[-1]
					self.plot(iteration)
			for name, datum in self._data.iteritems():
				# resample cluster for this data, given all other data
				# as fixed

				# first, remove this point from it's current cluster
				self._remove_datum(name, datum)
				# then find a new cluster for it to live in
				cluster = self._sample_datum(datum)
				# and, finally, add it back in
				self._add_datum(name, datum, cluster)

		print "Finished Gibbs with likelihood: %f" % self.log_likelihood()

	def log_likelihood(self):
		# evaluate the likelihood of the labelling (which is,
		# conveniently, just the likelihood of the current mixture
		# model)

		# FIXME: This should really be cached for the last invocation
		score = Counter(default=0.0)
		for cluster in self._cluster_to_datum:
			# Evaluate the likelihood of each individual cluster
			cluster_size = len(self._cluster_to_datum[cluster])
			if not cluster_size: continue
			# The mean of the data points belonging to this cluster
			cluster_datum_mean = sum(self._cluster_to_datum[cluster]) / cluster_size
			# The MLE of the mean of the cluster given the data points and the prior
			cluster_mle_mean = cluster_datum_mean

			likelihood = cluster_mle_mean * cluster_mle_mean
			likelihood *= (cluster_size * self._cluster_tau + self._mh_tau) ** 2
			score += likelihood

			likelihood = cluster_size * self._cluster_tau * cluster_datum_mean
			likelihood += self._mh_tau * self._mh_mean
			likelihood *= 2 * cluster_mle_mean
			score += likelihood

		# for the gaussian the dimensions are independent so we should
		# just be able to combine them directly
		return sum(score.itervalues())

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

		colors = itertools.cycle(("red", "green", "blue", "black", "purple", "orange"))
		for (cluster, cdata), color in zip(self._cluster_to_datum.iteritems(), colors):
			points_x = robjects.FloatVector([point['x'] for point in cdata])
			points_y = robjects.FloatVector([point['y'] for point in cdata])

			if not len(cdata): continue
			
			print "Cluster (size %d): %s" % (len(cdata), sum(cdata) / len(cdata))
			print color
			r.points(points_x, points_y, col=color)

			cmean = sum(cdata) / len(cdata)
			r.points(cmean['x'], cmean['y'], pch=21, cex=4.0, col=color)

		r['dev.off']()
