# Simple HMM implementation. Test code focuses on discrete signal reconstruction.

import sys
import random
from itertools import izip, islice
from time import time
from math import log, exp

from countermap import CounterMap
from nlp import counter as Counter

START_LABEL = "<START>"
STOP_LABEL = "<STOP>"

class HiddenMarkovModel:
	# Distribution over next state given current state
	labels = list()
	transition = CounterMap()
	reverse_transition = CounterMap() # same as transitions but indexed in reverse (useful for decoding)

	# Multinomial distribution over emissions given label
	emission = CounterMap()
	# p(label | emission)
	label_emissions = CounterMap()

	def __pad_sequence(self, sequence, pairs=False):
		if pairs: padding = [(START_LABEL, START_LABEL),]
		else: padding = [START_LABEL,]
		padding.extend(sequence)
		if pairs: padding.append((STOP_LABEL, STOP_LABEL))
		else: padding.append(STOP_LABEL)

		return padding

	def train(self, labeled_sequence):
		label_counts = Counter()
		# Currently this assumes the HMM is multinomial
		last_label = None

		labeled_sequence = self.__pad_sequence(labeled_sequence, pairs=True)

		# Transitions
		for label, emission in labeled_sequence:
			label_counts[label] += 1.0
			self.emission[label][emission] += 1.0
			self.label_emissions[emission][label] += 1.0
			if last_label:
				self.transition[last_label][label] += 1.0
			last_label = label

		self.label_emissions.normalize()
		self.transition.normalize()
		self.emission.normalize()
		self.labels = self.emission.keys()

		# Convert to log score counters
		for counter_map in (self.label_emissions, self.transition, self.emission):
			for sub_counter in counter_map.itervalues():
				sub_counter.default = float("-inf")
				for sub_key, score in sub_counter.iteritems():
					sub_counter[sub_key] = log(score)

		# Construct reverse transition probabilities
		for label, counter in self.transition.iteritems():
			for sublabel, score in counter.iteritems():
				self.reverse_transition[sublabel][label] = score
				self.reverse_transition[sublabel].default = float("-inf")

	def fallback_probs(self, emission):
		fallback = Counter()
		uniform = log(1.0 / len(self.labels))
		for label in self.labels: fallback[label] = uniform

		return fallback

	def score(self, labeled_sequence, debug=False):
		score = 0.0
		last_label = START_LABEL

		if debug: print "*** SCORE (%s) ***" % labeled_sequence

		for pos, (label, emission) in enumerate(labeled_sequence):
			if emission in self.emission[label]:
				score += self.emission[label][emission]
			else:
				score += self.fallback_probs(emission)[label]
			score += self.transition[last_label][label]
			last_label = label
			if debug: print "  SCP %d score after label %s emits %s: %s" % (pos, label, emission, score)

		score += self.transition[last_label][STOP_LABEL]

		if debug: print "*** SCORE => %f ***" % score
		
		return score

	def label(self, emission_sequence, debug=False, return_score=False):
		# This needs to perform viterbi decoding on the the emission sequence
		emission_sequence = self.__pad_sequence(emission_sequence)

		# Backtracking pointers - backtrack[position] = {state : prev, ...}
		backtrack = [dict() for state in emission_sequence]
		scores = list()

		for pos, (emission, backpointers) in enumerate(izip(emission_sequence, backtrack)):
			if debug: print "** ENTERING POS %d      :: %s" % (pos, emission)
			curr_scores = Counter()
			curr_scores.default = float("-inf")

			if pos == 0:
				curr_scores[START_LABEL] = 0.0
			else:
				# Transition probs (prob of arriving in this state)
				prev_scores = scores[pos-1]
				for label in self.labels:
					transition_scores = prev_scores + self.reverse_transition[label]
#					if debug: print "  Label %s :: %s" % (label, [i for i in transition_scores.iteritems() if i[1] != float("-inf")])
					last = transition_scores.arg_max()
					curr_score = transition_scores[last]
					if curr_score > float("-inf"):
						backpointers[label] = last
						curr_scores[label] = curr_score

#						if debug: print "          :: %f => %s" % (curr_scores[label], backpointers[label])

				if debug: 
					print " ++ TRANSITIONS        ::",
					if self.label_emissions[emission]: print ["%s => %s :: %f" % (backpointers[label], label, score) for label, score in curr_scores.iteritems() if label in self.label_emissions[emission]]
					else: print ["%s => %s :: %f" % (backpointers[label], label, score) for label, score in curr_scores.iteritems()]

				# Emission probs (prob. of emitting `emission`)
				if self.label_emissions.get(emission, None): curr_scores += self.label_emissions[emission]
				else: curr_scores += self.fallback_probs(emission)
				
				if debug:
					if self.label_emissions[emission]: print " ++ EMISSIONS          :: %s" % self.label_emissions[emission].items()
					else: print " ++ EMISSIONS FALLBACK :: %s" % [(label, score) for label, score in self.fallback_probs(emission).iteritems() if label in curr_scores]

			if debug: print "=> EXITING WITH SCORES :: %s" % [item for item in curr_scores.iteritems() if item[1] != float("-inf")]
			scores.append(curr_scores)

		# Now decode
		states = list()
		current = STOP_LABEL
		for pos in xrange(len(backtrack)-1, 0, -1):
			if current not in backtrack[pos]:
				current = STOP_LABEL
				states.append(current)
				continue
			if debug: print "Pos %d :: %s => %s" % (pos, current, backtrack[pos][current])
			current = backtrack[pos][current]
			states.append(current)

		states.pop()
		states.reverse()

		if return_score:
			return states, scores[-1][STOP_LABEL]
		return states

	def __sample_transition(self, label):
		sample = random.random()

		for next, prob in self.transition[label].iteritems():
			sample -= exp(prob)
			if sample <= 0.0: return next

		assert False, "Should have returned a next state"

	def __sample_emission(self, label):
		if label in [START_LABEL, STOP_LABEL]: return label
		sample = random.random()

		for next, prob in self.emission[label].iteritems():
			sample -= exp(prob)
			if sample <= 0.0: return next

		assert False, "Should have returned an emission"

	def sample(self, start=None):
		"""Returns a generator yielding a sequence of (state, emission) pairs
		generated by the modeled sequence"""
		state = start
		if not state:
			state = random.choice(self.transition.keys())
			for i in xrange(1000): state = self.__sample_transition(state)

		while True:
			yield (state, self.__sample_emission(state))
			state = self.__sample_transition(state)

def debug_problem(args):
	# Very simple chain for debugging purposes
	states = ['1', '1', '1', '2', '3', '3', '3', '3', STOP_LABEL, START_LABEL, '2', '3', '3']
	emissions = ['y', 'm', 'y', 'm', 'n', 'm', 'n', 'm', STOP_LABEL, START_LABEL, 'm', 'n', 'n']

	test_emissions = [['y', 'y', 'y', 'm', 'n', 'm', 'n', 'm'], ['y', 'm', 'n'], ['m', 'n', 'n', 'n']]
	test_labels = [['1', '1', '1', '2', '3', '3', '3', '3'], ['1', '2', '3'], ['2', '3', '3', '3']]

#	test_emissions, test_labels = ([['m', 'n', 'n', 'n']], [['2', '3', '3', '3']])

	chain = HiddenMarkovModel()
	chain.train(zip(states, emissions))

	print "Label"
	for emissions, labels in zip(test_emissions, test_labels):
		guessed_labels = chain.label(emissions)
		if guessed_labels != labels:
			print "Guessed: %s (score %f)" % (guessed_labels, chain.score(zip(guessed_labels, emissions)))
			print "Correct: %s (score %f)" % (labels, chain.score(zip(labels, emissions)))
			assert chain.score(zip(guessed_labels, emissions)) > chain.score(zip(labels, emissions)), "Decoder sub-optimality"
	print "Transition"
	print chain.transition
	print "Emission"
	print chain.emission

	sample = [val for _, val in izip(xrange(10), chain.sample())]
	print [label for label, _ in sample]
	print [emission for _, emission in sample]

def toy_problem(args):
	# Simulate a 3 state markov chain with transition matrix (given states in row vector):
	#  (destination)
	#   1    2    3
	# 1 0.7  0.3  0
	# 2 0.05 0.4  0.55
	# 3 0.25 0.25 0.5
	transitions = CounterMap()

	transitions['1']['1'] = 0.7
	transitions['1']['2'] = 0.3
	transitions['1']['3'] = 0.0

	transitions['2']['1'] = 0.05
	transitions['2']['2'] = 0.4
	transitions['2']['3'] = 0.55

	transitions['3']['1'] = 0.25
	transitions['3']['2'] = 0.25
	transitions['3']['3'] = 0.5

	def sample_transition(label):
		sample = random.random()

		for next, prob in transitions[label].iteritems():
			sample -= prob
			if sample <= 0.0: return next

		assert False, "Should have returned a next state"

	# And emissions (state, (counter distribution)): {1 : (yes : 0.5, sure : 0.5), 2 : (maybe : 0.75, who_knows : 0.25), 3 : (no : 1)}
	emissions = {'1' : {'yes' : 0.5, 'sure' : 0.5}, '2' : {'maybe' : 0.75, 'who_knows' : 0.25}, '3' : {'no' : 1.0}}

	def sample_emission(label):
		if label in [START_LABEL, STOP_LABEL]: return label
		choice = random.random()

		for emission, prob in emissions[label].iteritems():
			choice -= prob
			if choice <= 0.0: return emission

		assert False, "Should have returned an emission"
	
	# Create the training/test data
	states = ['1', '2', '3']
	start = random.choice(states)

	# Burn-in (easier than hand-calculating stationary distribution & sampling)
	for i in xrange(10000):	start = sample_transition(start)

	def label_generator(start_label):
		next = start_label
		while True:
			yield next
			next = sample_transition(next)

	training_labels = [val for _, val in izip(xrange(1000), label_generator('1'))]
	training_labels.extend((START_LABEL, STOP_LABEL))
	training_labels.extend([val for _, val in izip(xrange(1000), label_generator('2'))])
	training_labels.extend((START_LABEL, STOP_LABEL))
	training_labels.extend([val for _, val in izip(xrange(1000), label_generator('3'))])

	training_emissions = [sample_emission(label) for label in training_labels]

	training_signal = zip(training_labels, training_emissions)

	# Training phase
	signal_decoder = HiddenMarkovModel()
	signal_decoder.train(training_signal)

	# Labeling phase: given a set of emissions, guess the correct states
	start = random.choice(states)
	for i in xrange(10000):	start = sample_transition(start)
	test_labels = [val for _, val in izip(xrange(500), label_generator(start))]
	test_emissions = [sample_emission(label) for label in training_labels]

	guessed_labels = signal_decoder.label(test_emissions)
	correct = sum(1 for guessed, correct in izip(guessed_labels, test_labels) if guessed == correct)

	print "%d labels recovered correctly (%.2f%% correct out of %d)" % (correct, 100.0 * float(correct) / float(len(test_labels)), len(test_labels))

def main(args):
	if args[0] == 'toy': toy_problem(args[1:])
	elif args[0] == 'debug': debug_problem(args[1:])

if __name__ == "__main__":
	main(sys.argv[1:])
