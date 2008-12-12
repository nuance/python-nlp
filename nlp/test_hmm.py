from math import log
import unittest

from hmm import HiddenMarkovModel, START_LABEL, STOP_LABEL

class SimpleHMMTest(unittest.TestCase):

	def set_defaults(self, model):
		for state in model.labels:
			model.transition[state].default = float("-inf")
			model.reverse_transition[state].default = float("-inf")
			model.emission[state].default = float("-inf")
			model.label_emissions[state].default = float("-inf")

	def uniform_transitions(self, model):
		for start in ('A', 'B'):
			model.transition[start].update(self.defaults)
			model.reverse_transition[start].update(self.defaults)
			for finish in ('A', 'B', STOP_LABEL):
				model.transition[start][finish] = log(1.0 / 3.0)
				model.reverse_transition[finish][start] = log(1.0 / 3.0)
			model.transition[START_LABEL][start] = log(0.5)
			model.reverse_transition[start][START_LABEL] = log(0.5)

	def biased_transitions(self, model):
		for start in ('A', 'B'):
			model.transition[start].update(self.defaults)
			model.reverse_transition[start].update(self.defaults)
			for finish in ('A', 'B', STOP_LABEL):
				if start == finish:
					model.transition[start][finish] = log(1.0 / 2.0)
					model.reverse_transition[finish][start] = log(1.0 / 2.0)
				else:
					model.transition[start][finish] = log(1.0 / 4.0)
					model.reverse_transition[finish][start] = log(1.0 / 4.0)
			model.transition[START_LABEL][start] = log(0.5)
			model.reverse_transition[start][START_LABEL] = log(0.5)

	def identity_emissions(self, model):
		for label in model.labels:
			for emission in model.labels:
				if label == emission:
					model.emission[label][emission] = log(1.0)
					model.label_emissions[emission][label] = log(1.0)
				else:
					model.emission[label][emission] = float("-inf")
					model.label_emissions[emission][label] = float("-inf")

	def biased_emissions(self, model):
		for label in model.labels:
			for emission in model.labels:
				if label == emission:
					model.emission[label][emission] = log(2.0 / 3.0)
					model.label_emissions[emission][label] = log(2.0 / 3.0)
				else:
					model.emission[label][emission] = log(1.0 / (3.0 * float(len(model.labels)-1)))
					model.label_emissions[emission][label] = log(1.0 / (3.0 * float(len(model.labels)-1)))

	def _test_label(self, model, emissions, score, labels=None, debug=False):
		if debug: print
		if not labels: labels = emissions

		if debug: print "Emission-Labels: %s" % zip(emissions, labels)
		guessed_labels, labelling_score = model.label(emissions, debug=debug, return_score=True)
		if debug: print "Guessed labels: %s" % guessed_labels
		self.assertEqual(sum(label == emission for label, emission in zip(guessed_labels, labels)), len(emissions))
		
		if debug: print "Score: %f" % score
		guessed_score = model.score(zip(guessed_labels, emissions), debug=debug)
		if debug: print "Guessed score: %f" % guessed_score
		self.assertAlmostEqual(guessed_score, score, 4)
		self.assertAlmostEqual(score, labelling_score, 4)
		
	def setUp(self):
		self.defaults = {START_LABEL : float("-inf"), STOP_LABEL : float("-inf")}

	def test_identity_emission_uniform_transitions(self):
#		print "Testing emission == state w/ uniform transitions chain: ",

		model = HiddenMarkovModel()
		model.labels = ('A', 'B', START_LABEL, STOP_LABEL)

		self.set_defaults(model)
		self.uniform_transitions(model)
		self.identity_emissions(model)

		tests = [['A', 'A', 'A', 'A'], ['B', 'B', 'B', 'B'], ['A', 'A', 'B', 'B'], ['B', 'A', 'B', 'B']]

		for test in tests:
			self._test_label(model, test, log(1.0 / 2.0) + log(1.0 / 3.0) * 4)

#		print "ok"

	def test_identity_emissions_non_uniform_transitions(self):
#		print "Testing emissions == labels with non-uniform transitions chain: ",

		model = HiddenMarkovModel()
		model.labels = ('A', 'B', START_LABEL, STOP_LABEL)

		self.set_defaults(model)
		self.biased_transitions(model)
		self.identity_emissions(model)

		tests = [['A', 'A', 'A', 'A'], ['B', 'B', 'B', 'B'], ['A', 'A', 'B', 'B'], ['B', 'A', 'B', 'B']]
		scores = [log(0.5) * 4 + log(0.25), log(0.5) * 4 + log(0.25), log(0.5)*3 + log(0.25)*2, log(0.5)*2 + log(0.25)*3]
		scored_tests = zip(tests, scores)

		for test, score in scored_tests:
			self._test_label(model, test, score)

#		print "ok"

	def test_biased_emissions_uniform_transitions(self):
#		print "Testing uniform transitions with self-biased emissions: ",

		model = HiddenMarkovModel()
		model.labels = ('A', 'B', START_LABEL, STOP_LABEL)

		self.set_defaults(model)
		self.uniform_transitions(model)
		self.biased_emissions(model)

		tests = [['A', 'A', 'A', 'A'], ['B', 'B', 'B', 'B'], ['A', 'A', 'B', 'B'], ['B', 'A', 'B', 'B']]
		scores = [log(0.5) + log(1.0 / 3.0) * 4.0 + 6.0 * log(2.0 / 3.0) for i in xrange(4)]
		scored_tests = zip(tests, scores)

		for test, score in scored_tests:
			self._test_label(model, test, score)

#		print "ok"

	def test_biased_emissions_biased_transitions(self):
#		print "Testing self-biased transitions with self-biased emissions: ",

		model = HiddenMarkovModel()
		model.labels = ('A', 'B', START_LABEL, STOP_LABEL)

		self.set_defaults(model)
		self.biased_transitions(model)
		self.biased_emissions(model)

		tests = [['A', 'A', 'A', 'A'], ['B', 'B', 'B', 'B'], ['A', 'A', 'B', 'B'], ['B', 'A', 'B', 'B']]
		scores = [log(0.5) * 4 + log(0.25), log(0.5) * 4 + log(0.25), log(0.5)*3 + log(0.25)*2, log(0.5)*2 + log(0.25)*3]
		scores = [6.0 * log(2.0 / 3.0) + score for score in scores]
		scored_tests = zip(tests, scores)

		for test, score in scored_tests:
			self._test_label(model, test, score)

#		print "ok"

	def test_unk_emission(self):
#		print "Testing UNK emission with emission == label and self-biased transitions: ",

		model = HiddenMarkovModel()
		model.labels = ('A', 'B', START_LABEL, STOP_LABEL)

		self.set_defaults(model)
		self.identity_emissions(model)
		self.biased_transitions(model)

		emissions = ['A', 'C', 'A', 'B', 'B']
		labels = ['A', 'A', 'A', 'B', 'B']
		score = log(0.5) * 2 + log(0.25) + log(0.5) + log(0.25) + log(0.5) + log(0.25)

		self._test_label(model, emissions, score, labels=labels)

		emissions = ['A', 'C', 'C', 'B', 'B']
		labels = [['A', 'A', 'A', 'B', 'B'], ['A', 'A', 'B', 'B', 'B'], ['A', 'B', 'B', 'B', 'B']]

		score = None
		for label in labels:
			new_score = model.score(zip(label, emissions))
			if score: self.assertAlmostEqual(score, new_score, 5)#, "score(%s) (%f) bad" % (label, new_score)
			score = new_score

		emissions = ['A', 'C', 'C', 'B', 'B']
		labels = [['A', 'A', 'A', 'B', 'B'], ['A', 'A', 'B', 'B', 'B'], ['A', 'B', 'B', 'B', 'B']]

		score = None
		for label in labels:
			new_score = model.score(zip(label, emissions))
			if score: self.assertAlmostEqual(score, new_score, 5)#, "score(%s) (%f) bad" % (label, new_score)
			score = new_score

#		print "ok"

class HMMUtilityTest(unittest.TestCase):
	def test_extend_labels_simple(self):
		stream = (('1', 1), ('2', 2), ('3', 3))
		two_extended = [('1', ('', '<START>'), 1), ('2', ('', '1'), 2), ('3', ('', '2'), 3)]

		self.assertEquals(HiddenMarkovModel._extend_labels(stream, 2), two_extended)

	def test_extend_labels_multiple_sentences(self):
		stream = (('1', 1), ('2', 2), ('<STOP>', '<STOP>'), ('<START>', '<START>'), ('1', 1))
		two_extended = [('1', ('', '<START>'), 1), ('2', ('', '1'), 2), ('<STOP>', ('', '2'), '<STOP>'), ('<START>', ('', '<START>'), '<START>'), ('1', ('', '<START>'), 1)]

		self.assertEquals(HiddenMarkovModel._extend_labels(stream, 2), two_extended)

	def test_extend_labels_one_history(self):
		stream = (('1', 1), ('2', 2), ('3', 3))
		one_extended = [('1', ('',), 1), ('2', ('',), 2), ('3', ('',), 3)]

		self.assertEquals(HiddenMarkovModel._extend_labels(stream, 1), one_extended)
		
	def test_extend_labels_three_history(self):
		stream = (('1', 1), ('2', 2), ('3', 3))
		three_extended = [('1', ('', '<START>', '<START>::<START>'), 1), ('2', ('', '1', '<START>::1'), 2), ('3', ('', '2', '1::2'), 3)]

		self.assertEquals(HiddenMarkovModel._extend_labels(stream, 3), three_extended)

	def test_extend_labels_longer_history_than_sentence(self):
		stream = (('1', 1), ('2', 2))
		five_extended = [('1', ('', '<START>', '<START>::<START>', '<START>::<START>::<START>', '<START>::<START>::<START>::<START>'), 1),
						 ('2', ('', '1', '<START>::1', '<START>::<START>::1', '<START>::<START>::<START>::1'), 2)]

		self.assertEquals(HiddenMarkovModel._extend_labels(stream, 5), five_extended)

if __name__ == "__main__":
	unittest.main()
