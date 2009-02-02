# cython viterbi decoding & scoring
from itertools import izip
from counter import Counter

include "stdlib.pxi"

cdef enum bool_:
	false = 0
	true = 1

# class cyHMM:
# 	# encoding / decoding dictionaries
# 	cdef object label_idx
# 	cdef object idx_label[]
# 	cdef object emission_idx
# 	cdef object idx_emission[]

# 	# label transitions (null terminated)
# 	cdef bool_ label_transitions[][]

# 	# scores, indexed by encoded label / emission
# 	cdef double emission_scores[][]
# 	cdef double transition_scores[][]

cdef class CyHMM:
	cdef readonly object label_idx, idx_label
	cdef readonly bool_ **label_follows
	cdef readonly double **transition_idx_scores

	def __init__(labels, label_transitions, emissions, transition_scores):
	#	state = cyHMM()
		self.label_idx = dict()
		self.idx_label = list()

		for idx, label in enumerate(labels):
			label_idx[label] = idx
			idx_label[idx] = label

		label_follows = <bool_ **>malloc(len(labels) * sizeof(bool_*))
		for label in label_transitions:
			lbl = label_idx[label]
			label_follows[lbl] = <bool_ *>malloc(len(labels) * sizeof(bool_))

			for l in range(len(labels)):
				ls = idx_label[l]
				if ls in label_transitions[label]:
					label_follows[lbl][l] = true
				else:
					label_follows[lbl][l] = false

		transition_idx_scores = <double **>malloc(len(labels) * sizeof(double*))
		for transition in transition_scores:
			t_idx = label_idx[transition]
			transition_idx_scores[t_idx] = <double *>malloc(len(labels) * sizeof(double))

			for next in transition_scores[transition]:
				n_idx = label_idx[next]
				transition_idx_scores[t_idx][n_idx] = transition_scores[transition][next]

		return label_idx, emission_idx, label_follows, transition_idx_scores

	cdef object forward(self, object hmm, object emission_sequence):
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
				curr_scores[hmm.start_label] = 0.0
			else:
				# Transition probs (prob of arriving in this state)
				prev_scores = scores[pos-1]

				for label in hmm.labels:
					transition_scores = prev_scores + hmm.transition_scores(label)

					last = transition_scores.arg_max()
					curr_score = transition_scores[last]

					if curr_score > float("-inf"):
						backpointers[label] = last
						curr_scores[label] = curr_score

			curr_scores += hmm.emission_scores(emission)
			scores.append(curr_scores)

		return scores, backtrack

	def label(self, hmm, emission_sequence, debug=False, return_score=False):
		# This needs to perform viterbi decoding on the the emission sequence
		emission_length = len(emission_sequence)
		emission_sequence = list(hmm._pad_sequence(emission_sequence))

		scores, backtrack = forward(hmm, emission_sequence)

		# Now decode
		states = list()
		current = hmm.stop_label
		for pos in xrange(len(backtrack)-1, 0, -1):
			if current not in backtrack[pos]:
				current = scores[pos].arg_max()
			else:
				current = backtrack[pos][current]

			states.append(current.split('::')[-1])

		# Pop all the extra stop states
		states.pop()
		states.reverse()
		states = states[:emission_length]

		if return_score:
			return states, scores[-1]
		return states
