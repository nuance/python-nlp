from __future__ import with_statement

from itertools import chain
import sys

from countermap import CounterMap
from crp import CRPGibbsSampler
import features

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

		# and hand over work to the sampler
		print len(full_counts)
		sampler = CRPGibbsSampler(full_counts, burn_in_iterations=20)
		print len(full_counts)
		print sampler._datum_to_cluster

	def run(self, args):
		self.train(args)

if __name__ == "__main__":
	problem = SynonymLearner()
	problem.run(sys.argv[1:])
