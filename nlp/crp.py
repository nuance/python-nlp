import datetime

class CRPGibbsSampler(object):
	def __init__(self, data, gibbs_iterations=1):
		"""
		data: for now, counters of score-for-context (HUGE cardinality)
		gibbs_iterations: should be a number >= 1, large enough to
		ensure the chain is converged given updated params
		"""
		self._gibbs_iterations = gibbs_iterations
		self._data = data
		self._concentration = 5.0

		# These will be learned during sampling
		self._datum_to_cluster = dict()
		self._cluster_to_datum = dict()

		self._iteration_likelihoods = []
		self._cluster_count = []

	def _sample_datum(self, datum):
		raise Exception("Not implemented")

	def _add_datum(self, name, datum, cluster):
		self._datum_to_cluster[name] = cluster
		self._cluster_to_datum.setdefault(cluster, []).append(datum)

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
			if iteration % 1 == 0:
				print "*** Iteration %d starting (%s) ***" % (iteration, datetime.datetime.now())

			if self._cluster_to_datum:
				self._iteration_likelihoods.append(self.log_likelihood())
				self._cluster_count.append(len([c for c, v in self._cluster_to_datum.iteritems() if v]))
				print "    Clusters: %d" % self._cluster_count[-1]
				print "    Likelihood: %f" % self._iteration_likelihoods[-1]
				if iteration % 10 == 0:
					self.plot(iteration, cluster_only=True)
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

	def plot(self, iteration, cluster_only=False):
		print "Not implemented"

	def log_likelihood(self):
		raise Exception("NotImplemented")

