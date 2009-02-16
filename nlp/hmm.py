# Simple HMM implementation. Test code focuses on discrete signal reconstruction.

__using_cython_viterbi__ = True

from itertools import izip, islice, repeat
from math import log, exp
import cPickle as pickle
from pprint import pformat
import random
import sys
from time import time

from countermap import CounterMap
from counter import Counter
import cyhmm
from utilities import permutations, memoized

START_LABEL = "<START>"
STOP_LABEL = "<STOP>"
UNK_LABEL = "<UNK>"

class HiddenMarkovModel:
	def __init__(self, label_history_size=2):
		# Distribution over next state given current state
		self.labels = list()
		self.label_history_size = label_history_size
		self.transition = CounterMap()
		self.reverse_transition = CounterMap() # same as transitions but indexed in reverse (useful for decoding)

		self.fallback_emissions_model = None
		self.fallback_transition = None
		self.fallback_reverse_transition = None

		# Multinomial distribution over emissions given label
		self.emission = CounterMap()
		# p(label | emission)
		self.label_emissions = CounterMap()

	def _pad_sequence(self, sequence, pairs=False):
		if pairs: yield (START_LABEL, START_LABEL)
		else: yield START_LABEL

		for item in sequence: yield item

		# Pad the end so we'll decode the whole thing
		for _ in xrange(self.label_history_size):
			if pairs: yield (STOP_LABEL, STOP_LABEL)
			else: yield STOP_LABEL

	@classmethod
	def _extend_labels(cls, sequence, label_history_size):
		'''
		>>> foo = HiddenMarkovModel()
		>>> foo._extend_labels((('A', 3), ('B', 4), ('C', 5)), 1)
		[('A', (), 3), ('B', (), 4), ('C', (), 5)]
		>>> foo._extend_labels((('A', 3), ('B', 4), ('C', 5)), 2)
		[('A', ('<START>',), 3),
		 ('B', ('A',), 4),
		 ('C', ('B',), 5)]
		'''
		last_labels = [START_LABEL for _ in xrange(label_history_size)]

		for label, emission in sequence:
			last_labels.append(label)
			last_labels.pop(0)

			if label == START_LABEL:
				last_labels = [START_LABEL for _ in xrange(label_history_size)]

			all_labels = ('::'.join(last_labels[label_history_size-length-2:-1])
						  for length in xrange(label_history_size-1))
			yield (label, tuple(all_labels), emission)

	@property
	def start_label(self):
		return '::'.join(repeat(START_LABEL, self.label_history_size))

	@property
	def stop_label(self):
		return '::'.join(repeat(STOP_LABEL, self.label_history_size))

	def push_label(self, history, label):
		return '::'.join(history.split('::')[1:] + [label,])

	@classmethod
	def _linear_smooth(cls, labels, fallback_transition, label_history_size):
		transition = CounterMap()
		linear_smoothing_weights = [1.0 - 0.1 * (label_history_size-1)]
		linear_smoothing_weights.extend(0.1 for _ in xrange(label_history_size-1))

		# This is super inefficient - it should be caching smoothings involving the less-specific counters
		# e.g. smoothed['NN']['CD'] = cnter['NN']['CD'] * \lambda * smoothed['NN'] and so on
		all_label_histories = set(permutations(labels, label_history_size-1))
		for label_history in all_label_histories:
			histories = [history for history in (label_history[i:] for i in xrange(label_history_size))]
			# >>> label_history = ('WDT', 'RBR')
			# histories = [('WDT', 'RBR'), ('RBR')]

			history_strings = ['::'.join(history) for history in histories]
			history_scores = [fallback_transition[len(history)][history_string] for history, history_string in izip(histories, history_strings)]

			transition[history_strings[0]] = Counter()
			for smoothing, history_score in izip(linear_smoothing_weights, history_scores):
				transition[history_strings[0]] += history_score * smoothing

		transition.normalize()

		return transition

	def train(self, labeled_sequence, fallback_model=None, fallback_training_limit=None, use_linear_smoothing=True):
		label_counts = [Counter() for _ in xrange(self.label_history_size)]
		self.fallback_transition = [CounterMap() for _ in xrange(self.label_history_size)]
		self.fallback_reverse_transition = [CounterMap() for _ in xrange(self.label_history_size)]

		labeled_sequence = self._pad_sequence(labeled_sequence, pairs=True)
		labeled_sequence = list(HiddenMarkovModel._extend_labels(labeled_sequence, self.label_history_size+1))

		# Load emission and transition counters from the raw data
		for label, label_histories, emission in labeled_sequence:
			full_label = self.push_label(label_histories[-1], label)

			self.emission[full_label][emission] += 1.0
			self.label_emissions[emission][full_label] += 1.0

			for history_size, label_history in enumerate(label_histories):
				label_counts[history_size][label_history] += 1.0
				self.fallback_transition[history_size][label_history][full_label] += 1.0

		# Make the counters distributions
		for transition in self.fallback_transition:	transition.normalize()
		self.label_emissions.normalize()
		self.emission.normalize()
		self.labels = self.emission.keys()

		# Smooth transitions using fallback data
		# Doesn't work with label history size 1!
		if use_linear_smoothing and self.label_history_size > 1:
			self.transition = \
				HiddenMarkovModel._linear_smooth(self.labels,
												 self.fallback_transition,
												 self.label_history_size)
		else:
			self.transition = self.fallback_transition[-1]

		# Convert to log score counters
		self.transition.log()
		self.label_emissions.log()
		self.emission.log()

		self.reverse_transition = self.transition.inverted()

		# Train the fallback model on the label-emission pairs
		if fallback_model:
			try:
				start = time()
				pickle_file = open("fallback_model.pickle")
				self.fallback_emissions_model, training_pairs_length = pickle.load(pickle_file)
				pickle_file.close()

				if fallback_training_limit and fallback_training_limit != training_pairs_length:
					raise IOError()
				elif not fallback_training_limit and len(labeled_sequence) != training_pairs_length:
					raise IOError()

				print "Unpickling fallback model: %f" % (time() - start)
			except (IOError, EOFError), e:
				print "Training fallback model"
				self.fallback_emissions_model = fallback_model()

				emissions_training_pairs = [(emission_history[-1] + '::' + label, emission) for label, emission_history, emission in labeled_sequence if label != START_LABEL and label != STOP_LABEL]

				if fallback_training_limit:
					emissions_training_pairs = islice(emissions_training_pairs, fallback_training_limit)

				self.fallback_emissions_model.train(emissions_training_pairs)

				serialized = (self.fallback_emissions_model, len(labeled_sequence))
				pickle_file = open("fallback_model.pickle", "w")
				pickle.dump(serialized, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
				pickle_file.close()

		self._post_training()

	def _post_training(self):
		# Build the cython backing model
		if __using_cython_viterbi__:
			self.cyhmm = cyhmm.CyHMM(self.labels, self.reverse_transition)

	def emission_fallback_probs(self, emission):
		if self.fallback_emissions_model:
			return self.fallback_emissions_model.label_distribution(emission)

		fallback = Counter()
		uniform = log(1.0 / len(self.labels))
		for label in self.labels: fallback[label] = uniform

		return fallback


	@memoized
	def emission_scores(self, emission):
		"""
		Returns a counter of P(state | emission)
		"""

		# Emission probs (prob. of emitting `emission`) are P(label | emission),
		# so we need to convert the single labels to reduced history
		# e.g. say that P(DT | "A") = log(0.75), then we assume that
		# P(NNP::DT | "A") = P(DT::DT | "A") = log(0.75)

		if self.label_emissions.get(emission):
			return self.label_emissions[emission]
		else:
			return self.emission_fallback_probs(emission)

	def transition_scores(self, label):
		"""
		Returns a counter of s(label | previous state)
		"""

		return self.reverse_transition[label]

	def score(self, labeled_sequence, debug=False):
		score = 0.0
		last_score = 0.0
		last_labels = self.start_label

		if debug: print "*** SCORE (%s) ***" % labeled_sequence

		# Start with the probability of emitting the start emission
		score += self.emission_scores(START_LABEL)[START_LABEL]

		for pos, (label, emission) in enumerate(labeled_sequence):
			# Transition
			score += self.transition_scores(label)[last_labels]
			if debug: print " ++ TRANSITION (%s => %s): %f" % (last_labels, label, score - last_score)
			t_score = score

			# Emission
			score += self.emission_scores(emission)[label]
			if debug: print " ++ EMISSION: %f" % (score - t_score)

			if debug: print "  @ %d ::  score after label %s emits %s: %f (change %f)" % (pos, label, emission, score, score - last_score)

			# Bookkeeping
			last_labels = self.push_label(last_labels, label)
			last_score = score

		# Add in the probability of transitioning to the stop state
		score += self.transition_scores(STOP_LABEL)[last_labels]

		# And the normalizing probability of emitting the stop emission
		score += self.emission_scores(STOP_LABEL)[STOP_LABEL]

		if debug: print "*** SCORE => %f ***" % score
		
		return score

	def label(self, emission_sequence, debug=False, return_score=False):
		if __using_cython_viterbi__:
			labelling = self.cyhmm.label(self, emission_sequence, debug=debug)

			if return_score:
				score = self.score(zip(labelling, emission_sequence))
				return labelling, score

			return labelling
		else:
			return self._label(emission_sequence, debug=debug, return_score=return_score)

	def _label(self, emission_sequence, debug=False, return_score=False):
		# This needs to perform viterbi decoding on the the emission sequence
		emission_length = len(emission_sequence)
		emission_sequence = list(self._pad_sequence(emission_sequence))

		# Backtracking pointers - backtrack[position] = {state : prev, ...}
		backtrack = [dict() for state in emission_sequence]
		# Scores should contain counters with reduced histories as keys (so if
		# history is STATE1::STATE2::STATE3, key is STATE2::STATE3)
		scores = list()

		for pos, (emission, backpointers) in enumerate(izip(emission_sequence, backtrack)):
			curr_scores = Counter()
			curr_scores.default = float("-inf")

			if pos == 0:
				# Pack curr_scores with just the reduced start history
				curr_scores[self.start_label] = 0.0
			else:
				# Transition probs (prob of arriving in this state)
				prev_scores = scores[pos-1]

				for label in self.labels:
					transition_scores = prev_scores + self.transition_scores(label)

					last = transition_scores.arg_max()
					curr_score = transition_scores[last]

					if curr_score > float("-inf"):
						backpointers[label] = last
						curr_scores[label] = curr_score

			curr_scores += self.emission_scores(emission)
			scores.append(curr_scores)

		if debug:
			print
			print pformat(zip(enumerate(emission_sequence), backtrack))

		# Now decode
		states = list()
		current = self.stop_label

		for pos in xrange(len(backtrack)-1, 0, -1):
			if current not in backtrack[pos]:
  				if debug:
  					print "Pos %d :: %s =>" % (pos, current),
  					print scores[pos].arg_max(),
  					print "(no backpointer)"
  				current = scores[pos].arg_max()
			else:
				if debug: print "Pos %d :: %s => %s" % (pos, current, backtrack[pos][current])
				current = backtrack[pos][current]

			if not current:
				current = UNK_LABEL

			states.append(current.split('::')[-1])

		# Pop all the extra start & stop states
		states.reverse()
		if debug: print states
		states = states[1:]
		states = states[:emission_length]
		if debug: print states

		if return_score:
			return states, scores[-1][STOP_LABEL]
		return states

	def __sample_transition(self, label):
		sample = random.random()
		print "Label: \"" + label + "\""
		print "Transition: \"" + str(self.transition) + "\""

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

	def sample(self):
		"""Returns a generator yielding a sequence of (state, emission) pairs
		generated by the modeled sequence"""
		state = self.start_label

		while True:
			yield (state, self.__sample_emission(state))
			state = self.__sample_transition(state)

def debug_problem(args):
	#pragma: no cover
	# Very simple chain for debugging purposes
	states = ['1', '1', '1', '2', '3', '3', '3', '3', STOP_LABEL, START_LABEL, '2', '3', '3']
	emissions = ['y', 'm', 'y', 'm', 'n', 'm', 'n', 'm', STOP_LABEL, START_LABEL, 'm', 'n', 'n']

	test_emissions = [['y', 'y', 'y', 'm', 'n', 'm', 'n', 'm'], ['y', 'm', 'n'], ['m', 'n', 'n', 'n']]
	test_labels = [['1', '1', '1', '2', '3', '3', '3', '3'], ['1', '2', '3'], ['2', '3', '3', '3']]

#	test_emissions, test_labels = ([['m', 'n', 'n', 'n']], [['2', '3', '3', '3']])

	chain = HiddenMarkovModel(label_history_size=1)

	chain.train(zip(states, emissions), use_linear_smoothing=False)

	print "Label"
	for emissions, labels in zip(test_emissions, test_labels):
		guessed_labels = chain.label(emissions)
		if guessed_labels != labels:
			print "Guessed: %s (score %f)" % (guessed_labels, chain.score(zip(guessed_labels, emissions)))
			print "Correct: %s (score %f)" % (labels, chain.score(zip(labels, emissions)))
			assert chain.score(zip(guessed_labels, emissions)) >= chain.score(zip(labels, emissions)), "Decoder sub-optimality"
#	print "Transition"
#	print pformat(dict(chain.transition.items()))
#	print "Emission"
#	print pformat(dict(chain.emission.items()))

	sample = [val for _, val in izip(xrange(10), chain.sample())]
	print [label for label, _ in sample]
	print [emission for _, emission in sample]

def toy_problem(args):
	#pragma: no cover
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
	signal_decoder = HiddenMarkovModel(label_history_size=1)
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
