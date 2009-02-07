# cython viterbi decoding & scoring
from itertools import izip
from counter import Counter

include "stdlib.pxi"
include "math.pxi"

cdef enum bool_:
	false = 0
	true = 1

# class cyHMM:
# 	# encoding / decoding dictionaries
# 	cdef object label_idx
# 	cdef object idx_label[]

# 	# scores, indexed by encoded label / emission
# 	cdef double emission_scores[][]
# 	cdef double transition_scores[][]

cdef class CyHMM:
	cdef readonly object label_idx, idx_label
	cdef double **transition_idx_scores
	cdef int label_count

	def __init__(self, labels, transition_scores):
	#	state = cyHMM()
		self.label_idx = dict()
		self.idx_label = list()

		for idx, label in enumerate(labels):
			self.label_idx[label] = idx
			self.idx_label.append(label)

		self.label_count = len(labels)

		self.transition_idx_scores = <double **>malloc(len(labels) * sizeof(double*))
		for transition in labels:
			t_idx = self.label_idx[transition]
			self.transition_idx_scores[t_idx] = <double *>malloc(len(labels) * sizeof(double))

			for next in labels:
				n_idx = self.label_idx[next]
				self.transition_idx_scores[t_idx][n_idx] = transition_scores[transition][next]

	cdef void add_score_vectors(CyHMM self, double *dst, double *a, double *b, int length):
		print "hello"
		for i in range(length):
			print "hi"
#			print "a[i] = %f" % a[i]
#			print "b[i] = %f" % b[i]
#			print "dst[i] = %f" % dst[i]
			dst[i] = a[i] + b[i]
#			print "dst[i] = %f" % dst[i]

	cdef int** forward(CyHMM self, object hmm, object emission_sequence):
		# Backtracking pointers - backtrack[position] = {state : prev, ...}
		cdef int **backpointers = <int**>malloc(len(emission_sequence) * sizeof(int*))
		cdef size_t scores_len = self.label_count * sizeof(double)

		# These two should really be outside of the forward def'n
		cdef double *zero_scores = <double*>malloc(scores_len * sizeof(double))
		cdef double ninf = log(0)

		for i in range(scores_len):
			zero_scores[i] = ninf

		cdef double *curr_scores, *prev_scores, *swap

		curr_scores = <double*>malloc(scores_len * sizeof(double))
		prev_scores = <double*>malloc(scores_len * sizeof(double))
		memcpy(prev_scores, zero_scores, scores_len * sizeof(double))

		cdef double *label_scores = <double*> malloc(scores_len * sizeof(double))

		# Manually unroll first iteration so we don't risk branch mispredict
		# (indented to signify it really belongs below)
			# Pack curr_scores with just the reduced start history
		cdef int start_label = self.label_idx[hmm.start_label]
		prev_scores[start_label] = 0.0
		backpointers[0] = <int*>malloc(self.label_count * sizeof(int))

		# loop vars
		cdef int last_label
		cdef int *backtrack
		cdef double score = ninf

		for pos in range(1, len(emission_sequence)):
			# loop vars
			backpointers[pos] = <int*>malloc(self.label_count * sizeof(int))
			backtrack = backpointers[pos]
			emission = emission_sequence[pos]

			# Wipe out scores
			memcpy(curr_scores, zero_scores, scores_len)

			for label in range(self.label_count):
				# Transition scores
				self.add_score_vectors(label_scores, prev_scores, self.transition_idx_scores[label], self.label_count)

				# Pick max / argmax
				last_label = 0
				score = ninf

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
		emission_sequence = list(hmm._pad_sequence(emission_sequence))

		cdef int **backtrack = self.forward(hmm, emission_sequence)

		# Now decode
		states = list()
		current = hmm.stop_label
		current_idx = self.label_idx[current]

		for pos in xrange(len(emission_sequence)-1, 0, -1):
			current_idx = backtrack[pos][current_idx]
			current = self.idx_label[current_idx]
			states.append(current.split('::')[-1])
			free(backtrack[pos])

		free(backtrack)

		# Pop all the extra stop states
		states.pop()
		states.reverse()
		states = states[:len(emission_sequence)]

		return states
