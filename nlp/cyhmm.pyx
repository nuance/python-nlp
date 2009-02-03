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
			self.label_idx[label] = idx
			self.idx_label[idx] = label

		self.label_follows = <bool_ **>malloc(len(labels) * sizeof(bool_*))
		for label in label_transitions:
			lbl = label_idx[label]
			self.label_follows[lbl] = <bool_ *>malloc(len(labels) * sizeof(bool_))

			for l in range(len(labels)):
				ls = idx_label[l]
				self.label_follows[lbl][l] = int(bool(ls in label_transitions[label]))

		self.transition_idx_scores = <double **>malloc(len(labels) * sizeof(double*))
		for transition in transition_scores:
			t_idx = label_idx[transition]
			self.transition_idx_scores[t_idx] = <double *>malloc(len(labels) * sizeof(double))

			for next in transition_scores[transition]:
				n_idx = label_idx[next]
				self.transition_idx_scores[t_idx][n_idx] = transition_scores[transition][next]

	cdef void add_score_vectors(double *dst, double *a, double *b, int length):
		for i in range(length):
			dst[i] = a[i] + b[i]

	cdef int** forward(self, object hmm, int[] emission_sequence, int emissions):
		# Backtracking pointers - backtrack[position] = {state : prev, ...}
		cdef int **backpointers = <int**>malloc(emissions * sizeof(int*))
		cdef size_t scores_len = self.label_count * sizeof(double)

		# These two should really be outside of the forward def'n
		cdef double *zero_scores = <double*>malloc(scores_len)
		cdef double ninf = log(0)

		for i in range(scores_len):
			zero_scores[i] = ninf

		cdef double *curr_scores, *prev_scores
		memcpy(prev_scores, zero_scores, scores_len)

		cdef double *label_scores = <double*> malloc(scores_len)

		# Manually unroll first iteration so we don't risk branch mispredict
		# (indented to signify it really belongs below)
			# Pack curr_scores with just the reduced start history
			prev_scores[self.label_idx[hmm.start_label]] = 0.0

		for pos in range(1, emissions):
			# loop vars
			backtrack = backpointers[pos]
			emission = emission_sequence[pos]

			# Wipe out scores
			memcpy(curr_scores, zero_scores, scores_len)

			for label in range(self.label_count):
				# Transition scores
				add_score_vectors(label_scores, prev_scores, self.transition_idx_scores[label], self.label_count)

				# Pick max / argmax
				cdef int last_label
				cdef double score = ninf

				for i in range(self.label_count):
					if label_scores[i] > score:
						last_label = i
						score = label_scores[i]

				backtrack[label] = last_label
				curr_scores[label] = score

			emission_scores = hmm.emission_scores(emission)
			for label in emission_scores:
				label_idx = self.label_idx[label]
				score = emission_scores[label]
				curr_scores[label_idx] += score

			# And set up for the next iteration
			swap = prev_scores
			prev_scores = curr_scores
			curr_scores = swap # will be obliterated in the beginning of the loop

		free(curr_scores)
		free(prev_scores)
		free(zero_scores)
		free(label_scores)

		return backpointers

	def label(self, hmm, emission_sequence, debug=False, return_score=False):
		# This needs to perform viterbi decoding on the the emission sequence
		emission_length = len(emission_sequence)
		emission_sequence = list(hmm._pad_sequence(emission_sequence))

		backtrack = forward(hmm, emission_sequence)

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
