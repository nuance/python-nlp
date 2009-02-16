from __future__ import with_statement
from itertools import chain
import sys

from countermap import CounterMap
import features

class CoOccurenceProblem(object):
	@classmethod
	def gather_cooccurrence_counts(cls, triples):
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

		return pre_counts, post_counts, full_counts

	def __init__(self):
		super(CoOccurenceProblem, self).__init__()

	def file_triples(self, lines):
		for line in lines:
			for triple in featurers.contexts(line.rstrip().split(), context_size=2):
				yield triple

	def run(self, paths):
		files = [open(path) for path in paths]

		pre_counts, post_counts, full_counts = self.gather_cooccurrence_counts(chain(*[self.file_triples(file) for file in files]))

		full_counts += pre_counts
		full_counts += post_counts

		print full_counts

		for file in files:
			file.close()

		# now cluster...

if __name__ == "__main__":
	problem = CoOccurenceProblem()
	problem.run(sys.argv[1:])
