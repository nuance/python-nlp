from __future__ import with_statement

import datetime
from itertools import chain
import sys

from counter import Counter
from countermap import CounterMap
import features

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

		self._concentration = 1.0

		# These will be learned during the initial burn-in
		self._clusters = []
		self._datum_to_cluster = dict()
		self._cluster_counts = CounterMap(default=float("-inf"))

		# and run the burn in...
		self.gibbs(burn_in_iterations)

	def _update_cluster_params(self, cluster, added=None, removed=None):
		assert (added or removed) and not (added and removed)

		if added:
			if cluster >= len(self._clusters):
				cluster_params = [cluster, None, 1]
				self._clusters.append(cluster_params)
			else:
				self._clusters[cluster][2] += 1
		else:
			assert cluster < len(self._clusters)
			self._clusters[cluster][2] -= 1

	def _remove_datum(self, datum):
		cluster = self._datum_to_cluster.get(datum)
		if not cluster: return
		self._cluster_counts[cluster] -= self._data[datum]
		self._update_cluster_params(cluster, removed=datum)

		del self._datum_to_cluster[datum]

	def _sample_datum(self, datum):
		probs = Counter()

		for cluster, cluster_params, cluster_size in self._clusters:
			# TODO: actually do this
			probs[cluster] = cluster_size

		probs[len(self._clusters)] = self._concentration
		probs.normalize()

		return probs.sample()

	def _add_datum(self, datum, cluster):
		self._datum_to_cluster[datum] = cluster
		self._cluster_counts[cluster] += self._data[datum]
		self._update_cluster_params(cluster, added=datum)

	def gibbs(self, iterations=None):
		# use gibbs sampling to find a sufficiently good labelling
		# starting with the current parameters and iterate
		if not iterations:
			iterations = self._gibbs_iterations

		for iteration in xrange(iterations):
			print "*** Iteration %d starting (%s) ***" % (iteration, datetime.datetime.now())
			print self._datum_to_cluster

			if self._clusters:
				print "    Likelihood: %f" % self.log_likelihood()
			for datum in self._data:
				# resample cluster for this data, given all other data
				# as fixed

				# first, remove this point from it's current cluster
				self._remove_datum(datum)
				# then find a new cluster for it to live in
				cluster = self._sample_datum(datum)
				# and, finally, add it back in
				self._add_datum(datum, cluster)

		print "Finished Gibbs with likelihood: %f" % self.log_likelihood()

	def log_likelihood(self):
		# evaluate the likelihood of the labelling (which is,
		# conveniently, just the likelihood of the current mixture
		# model)

		# FIXME: This should really be cached for the last invocation
		# TODO: implement this
		return -1.0

class SynonymLearner(object):
	def __init__(self):
		super(SynonymLearner, self).__init__()

	def _file_triples(self, lines):
		for line in lines:
			for triple in features.contexts(line.rstrip().split(), context_size=1):
				yield triple

	def _gather_colocation_counts(self, files):
		files = [open(path) for path in files]
		triples = chain(*[self._file_triples(file) for file in files])

		pre_counts = CounterMap()
		post_counts = CounterMap()
		full_counts = CounterMap()

		for pre, word, post in triples:
			full_context = '::'.join(pre + post)
			pre_context = '::'.join(pre)
			post_context = '::'.join(post)

			pre_counts[word][pre_context] += 1
			post_counts[word][post_context] += 1
			full_counts[word][full_context] += 1

		for file in files:
			file.close()

		return pre_counts, post_counts, full_counts

	def train(self, paths):
		pre_counts, post_counts, full_counts = self._gather_colocation_counts(paths)

		full_counts += pre_counts
		full_counts += post_counts

		print full_counts

		# and hand over work to the sampler
		sampler = CRPGibbsSampler(full_counts, burn_in_iterations=5)

	def run(self, args):
		self.train(args)

if __name__ == "__main__":
	problem = SynonymLearner()
	problem.run(sys.argv[1:])
