from math import log

from hmm import HiddenMarkovModel, START_LABEL, STOP_LABEL

def test_problem():
	defaults = {START_LABEL : float("-inf"), STOP_LABEL : float("-inf")}

	def set_defaults(model):
		for state in model.labels:
			model.transition[state].default = float("-inf")
			model.reverse_transition[state].default = float("-inf")
			model.emission[state].default = float("-inf")
			model.label_emissions[state].default = float("-inf")

	def uniform_transitions(model):
		for start in ('A', 'B'):
			model.transition[start].update(defaults)
			model.reverse_transition[start].update(defaults)
			for finish in ('A', 'B', STOP_LABEL):
				model.transition[start][finish] = log(1.0 / 3.0)
				model.reverse_transition[finish][start] = log(1.0 / 3.0)
			model.transition[START_LABEL][start] = log(0.5)
			model.reverse_transition[start][START_LABEL] = log(0.5)

	def self_biased_transitions(model):
		for start in ('A', 'B'):
			model.transition[start].update(defaults)
			model.reverse_transition[start].update(defaults)
			for finish in ('A', 'B', STOP_LABEL):
				if start == finish:
					model.transition[start][finish] = log(1.0 / 2.0)
					model.reverse_transition[finish][start] = log(1.0 / 2.0)
				else:
					model.transition[start][finish] = log(1.0 / 4.0)
					model.reverse_transition[finish][start] = log(1.0 / 4.0)
			model.transition[START_LABEL][start] = log(0.5)
			model.reverse_transition[start][START_LABEL] = log(0.5)

	def identity_emissions(model):
		for label in model.labels:
			for emission in model.labels:
				if label == emission:
					model.emission[label][emission] = log(1.0)
					model.label_emissions[emission][label] = log(1.0)
				else:
					model.emission[label][emission] = float("-inf")
					model.label_emissions[emission][label] = float("-inf")

	def self_biased_emissions(model):
		for label in model.labels:
			for emission in model.labels:
				if label == emission:
					model.emission[label][emission] = log(2.0 / 3.0)
					model.label_emissions[emission][label] = log(2.0 / 3.0)
				else:
					model.emission[label][emission] = log(1.0 / (3.0 * float(len(model.labels)-1)))
					model.label_emissions[emission][label] = log(1.0 / (3.0 * float(len(model.labels)-1)))

	def test_label(model, emissions, score, labels=None, debug=False):
		if debug: print
		if not labels: labels = emissions

		if debug: print "Emission-Labels: %s" % zip(emissions, labels)
		guessed_labels = model.label(emissions, debug=debug)
		if debug: print "Guessed labels: %s" % guessed_labels
		assert sum(label == emission for label, emission in zip(guessed_labels, labels)) == len(emissions)
		
		if debug: print "Score: %f" % score
		guessed_score = model.score(zip(guessed_labels, emissions), debug=debug)
		if debug: print "Guessed score: %f" % guessed_score
		assert abs(guessed_score - score) < 0.0001, score

	print "Testing emission == state w/ uniform transitions chain: ",

	model = HiddenMarkovModel()
	model.labels = ('A', 'B', START_LABEL, STOP_LABEL)

	set_defaults(model)
	uniform_transitions(model)
	identity_emissions(model)

	tests = [['A', 'A', 'A', 'A'], ['B', 'B', 'B', 'B'], ['A', 'A', 'B', 'B'], ['B', 'A', 'B', 'B']]

	for test in tests:
		test_label(model, test, log(1.0 / 2.0) + log(1.0 / 3.0) * 4)

	print "ok"

	print "Testing emissions == labels with non-uniform transitions chain: ",

	model = HiddenMarkovModel()
	model.labels = ('A', 'B', START_LABEL, STOP_LABEL)

	set_defaults(model)
	self_biased_transitions(model)
	identity_emissions(model)

	scores = [log(0.5) * 4 + log(0.25), log(0.5) * 4 + log(0.25), log(0.5)*3 + log(0.25)*2, log(0.5)*2 + log(0.25)*3]
	scored_tests = zip(tests, scores)

	for test, score in scored_tests:
		test_label(model, test, score)

	print "ok"

	print "Testing uniform transitions with self-biased emissions: ",

	model = HiddenMarkovModel()
	model.labels = ('A', 'B', START_LABEL, STOP_LABEL)

	set_defaults(model)
	uniform_transitions(model)
	self_biased_emissions(model)

	scores = [log(0.5) + log(1.0 / 3.0) * 4.0 + 4.0 * log(2.0 / 3.0) for i in xrange(4)]
	scored_tests = zip(tests, scores)

	for test, score in scored_tests:
		test_label(model, test, score, debug=True)

	print "ok"

	print "Testing self-biased transitions with self-biased emissions: ",

	model = HiddenMarkovModel()
	model.labels = ('A', 'B', START_LABEL, STOP_LABEL)

	set_defaults(model)
	self_biased_transitions(model)
	self_biased_emissions(model)

	scores = [log(0.5) * 4 + log(0.25), log(0.5) * 4 + log(0.25), log(0.5)*3 + log(0.25)*2, log(0.5)*2 + log(0.25)*3]
	scores = [4.0 * log(2.0 / 3.0) + score for score in scores]
	scored_tests = zip(tests, scores)

	for test, score in scored_tests:
		test_label(model, test, score)

	print "ok"

	print "Testing UNK emission with emission == label and self-biased transitions: ",

	model = HiddenMarkovModel()
	model.labels = ('A', 'B', START_LABEL, STOP_LABEL)

	set_defaults(model)
	identity_emissions(model)
	self_biased_transitions(model)

	emissions = ['A', 'C', 'A', 'B', 'B']
	labels = ['A', 'A', 'A', 'B', 'B']
	score = log(0.5) * 2 + log(0.25) + log(0.5) + log(0.25) + log(0.5) + log(0.25)

	test_label(model, emissions, score, labels=labels)

	emissions = ['A', 'C', 'C', 'B', 'B']
	labels = [['A', 'A', 'A', 'B', 'B'], ['A', 'A', 'B', 'B', 'B'], ['A', 'B', 'B', 'B', 'B']]

	score = None
	for label in labels:
		new_score = model.score(zip(label, emissions))
		if score: assert score == new_score, "score(%s) (%f) bad" % (label, new_score)
		score = new_score

	emissions = ['A', 'C', 'C', 'B', 'B']
	labels = [['A', 'A', 'A', 'B', 'B'], ['A', 'A', 'B', 'B', 'B'], ['A', 'B', 'B', 'B', 'B']]

	score = None
	for label in labels:
		new_score = model.score(zip(label, emissions))
		if score: assert score == new_score, "score(%s) (%f) bad" % (label, new_score)
		score = new_score

	print "ok"

if __name__ == "__main__":
	test_problem()
