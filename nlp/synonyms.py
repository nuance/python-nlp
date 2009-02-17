from __future__ import with_statement

import datetime
from itertools import chain
import sys

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

		# These will be learned during the initial burn-in
		self._cluster_params = None
		self._datum_to_cluster = dict()
		self._cluster_counts = CounterMap(default=float("-inf"))

		# and run the burn in...
		self.gibbs(burn_in_iterations)

	def _remove_datum(self, datum):
		cluster = self._datum_to_cluster[datum]
		self._cluster_counts -= self._data[datum]
		self._update_cluster_params(cluster)

		del self._datum_to_cluster[datum]

	def _sample_datum(self, datum):
		# TODO: implement this		
		return None

	def _add_datum(self, datum, cluster):
		self._datum_to_cluster[datum] = cluster
		self._cluster_counts += self._data[datum]
		self._update_cluster_params(cluster)

	def gibbs(self, iterations=None):
		# use gibbs sampling to find a sufficiently good labelling
		# starting with the current parameters and iterate
		if not iterations:
			iterations = self._gibbs_iterations

		for iteration in xrange(iterations):
			print "*** Iteration %d starting (%s) ***" % (iteration, datetime.now())
			if self._cluster_params:
				print "    Likelihood: %f" % self.likelihood()
			for datum in self._data:
				# resample cluster for this data, given all other data
				# as fixed

				# first, remove this point from it's current cluster
				self._remove_datum(datum)
				# then find a new cluster for it to live in
				cluster = self._sample_datum(datum)
				# and, finally, add it back in
				self._add_datum(datum, cluster)

		print "Finished Gibbs with likelihood: %f" % self.likelihood()

	def likelihood(self):
		# evaluate the likelihood of the labelling (which is,
		# conveniently, just the likelihood of the current mixture
		# model)

		# FIXME: This should really be cached for the last invocation
		# TODO: implement this
		pass

class SynonymLearner(object):
	def __init__(self):
		super(SynonymLearner, self).__init__()

	def _file_triples(self, lines):
		for line in lines:
			for triple in features.contexts(line.rstrip().split(), context_size=2):
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

		# and hand over work to the sampler

if __name__ == "__main__":
	problem = SynonymLearner()
	problem.run(sys.argv[1:])
