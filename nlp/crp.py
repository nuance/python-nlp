from collections import defaultdict
import datetime

from countermap import CounterMap
from counter import Counter

class CRPGibbsSampler(object):
	def __init__(self, data, burn_in_iterations=1, gibbs_iterations=1):
		"""
		data: for now, counters of score-for-context (HUGE cardinality)
		burn_in_iterations: # of times to run gibbs before we do
		anything else
		gibbs_iterations: should be a number >= 1, large enough to
		ensure the chain is converged given updated params
		"""
		self._gibbs_iterations = gibbs_iterations
		self._data = data

		self._concentration = 0.2

		# fixed variance
		self._cluster_tau = 1.0 / (2.0**2)

		# hyper-params for the mean
		self._mh_tau = 1.0 / (1.0**2)
		self._mh_mean = Counter(default=1.0)

		# These will be learned during the initial burn-in
		self._datum_to_cluster = dict()
		self._cluster_to_datum = defaultdict(list)
		self._cluster_counts = CounterMap(default=float("-inf"))

		# and run the burn in...
		self.gibbs(burn_in_iterations)

	def _sample_datum(self, datum):
		probs = Counter()

		for cluster in self._cluster_to_datum:
			# TODO: actually do this
			probs[cluster] = len(self._cluster_to_datum)

		# TODO: do this
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
			print "*** Iteration %d starting (%s) ***" % (iteration, datetime.datetime.now())

			if self._cluster_to_datum:
				print "    Clusters: %d" % len([c for c, v in self._cluster_to_datum.iteritems() if v])
				print "    Likelihood: %f" % self.log_likelihood()
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
		score = 0.0
		for cluster in self._cluster_to_datum:
			cluster_size = len(self._cluster_to_datum[cluster])
			cluster_mean = sum(self._cluster_to_datum[cluster], Counter()) / cluster_size
			mean = self._mh_mean * self._mh_tau
			mean += cluster_mean * self._cluster_tau
			mean /= (self._mh_tau + cluster_size * self._cluster_tau)
			precision= self._mh_tau + cluster_size * self._cluster_tau

			for point in self._cluster_to_datum[cluster]:
				diff = point - mean
				point_score = diff * diff
				point_score *= 0.5 * precision
				score -= point_score.total_count()

		return score
