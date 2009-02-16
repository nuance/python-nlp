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

	cdef double *zero_scores

	def __init__(self, labels, transition_scores):
	#	state = cyHMM()
		self.label_idx = dict()
		self.idx_label = list()

		for idx, label in enumerate(labels):
			self.label_idx[label] = idx
			self.idx_label.append(label)

		self.label_count = len(labels)

		cdef int i
		cdef double ninf = log(0)
		self.zero_scores = <double*>malloc(self.label_count * sizeof(double))

		for i in range(self.label_count):
			self.zero_scores[i] = ninf

		self.transition_idx_scores = <double **>malloc(len(labels) * sizeof(double*))
		for transition in labels:
			t_idx = self.label_idx[transition]
			self.transition_idx_scores[t_idx] = <double *>malloc(len(labels) * sizeof(double))

			for next in labels:
				n_idx = self.label_idx[next]
				if transition in transition_scores and next in transition_scores[transition]:
					self.transition_idx_scores[t_idx][n_idx] = transition_scores[transition][next]
				else:
					self.transition_idx_scores[t_idx][n_idx] = float("-inf")

	cdef void add_score_vectors(CyHMM self, double *dst, double *a, double *b, int length):
		cdef int i
		for i in range(length):
			dst[i] = a[i] + b[i]

	cdef int** forward(CyHMM self, object hmm, object emission_sequence, int *arg_maxes, bool_ debug) except *:
		# Backtracking pointers - backtrack[position] = {state : prev, ...}
		cdef int **backpointers = <int**>malloc(len(emission_sequence) * sizeof(int*))
		cdef size_t scores_len = self.label_count * sizeof(double)
		cdef int pos, i, label_idx

		# These two should really be outside of the forward def'n
		cdef double ninf = log(0)
		cdef double *curr_scores, *prev_scores, *swap

		curr_scores = <double*>malloc(scores_len * sizeof(double))
		prev_scores = <double*>malloc(scores_len * sizeof(double))
		memcpy(prev_scores, self.zero_scores, scores_len * sizeof(double))

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
		cdef double label_score = ninf
		cdef double top_score = ninf

		if debug == true:
			print "LABEL :: %s" % emission_sequence

		for pos in range(1, len(emission_sequence)):
			# loop vars
			backpointers[pos] = <int*>malloc(self.label_count * sizeof(int))
			backtrack = backpointers[pos]
			emission = emission_sequence[pos]
			top_score = ninf

			if debug == true:
				print "** ENTERING POS %d     :: %s" % (pos, emission)

			# Wipe out scores
			memcpy(curr_scores, self.zero_scores, scores_len)
			
			if debug == true:
				print " >> PREVIOUS SCORES    :: %s" % [(self.idx_label[history], prev_scores[history]) for history in range(self.label_count) if prev_scores[history] > float("-inf")]

			for label_idx in range(self.label_count):
# 				if debug == true:
# 					print "    ++ LABEL          :: %s" % self.idx_label[label_idx]
				# Pick max / argmax of sums
				last_label = self.label_count + 1
				score = ninf

				for i in range(self.label_count):
					label_score = prev_scores[i] + self.transition_idx_scores[label_idx][i]
					if label_score > score:
						last_label = i
						score = label_score

				backtrack[label_idx] = last_label
				curr_scores[label_idx] = score

			if debug == true:
				print " >> PREVIOUS           :: %s" % [(self.idx_label[label_idx], self.idx_label[backtrack[label_idx]], prev_scores[backtrack[label_idx]]) for label_idx in range(self.label_count) if backtrack[label_idx] < self.label_count]
				print " ++ TRANSITIONS        ::",
				if hmm.label_emissions.get(emission):
					print ["%s => %s :: %f" % (self.idx_label[backtrack[label_idx]], self.idx_label[label_idx], curr_scores[label_idx]) for label_idx in range(self.label_count) if self.idx_label[label_idx] in hmm.label_emissions[emission] and backtrack[label_idx] < self.label_count]
				else:
					print ["%s => %s :: %f" % (self.idx_label[backtrack[label_idx]], self.idx_label[label_idx], curr_scores[label_idx]) for label_idx in range(self.label_count) if backtrack[label_idx] < self.label_count]

			emission_scores = hmm.emission_scores(emission)
			arg_maxes[pos] = 0
			for label_idx in range(self.label_count):
				label = self.idx_label[label_idx]
				score = emission_scores[label]
				curr_scores[label_idx] += score

				if score > top_score:
					top_score = score
					arg_maxes[pos] = label_idx

			if debug == true:
				print " ++ EMISSION SCORES    :: %s" % emission_scores.items()

			if debug == true: print "=> EXITING WITH SCORES :: %s" % [(self.idx_label[label], curr_scores[label]) for label in range(self.label_count) if curr_scores[label] > ninf]

			# And set up for the next iteration
			swap = prev_scores
			prev_scores = curr_scores
			curr_scores = swap # will be obliterated in the beginning of the loop

		free(curr_scores)
		free(prev_scores)

		return backpointers

	def label(self, hmm, emission_sequence, debug=False, return_score=False):
		# This needs to perform viterbi decoding on the the emission sequence
		emission_length = len(emission_sequence)
		emission_sequence = list(hmm._pad_sequence(emission_sequence))

		cdef bool_ c_debug = false
		if debug:
			c_debug = true

		cdef int *arg_maxes = <int*>malloc(len(emission_sequence) * sizeof(int))
		cdef int **backtrack = self.forward(hmm, emission_sequence, arg_maxes, c_debug)
		cdef int pos, current_idx

		# Now decode
		states = list()
		current = hmm.stop_label
		current_idx = self.label_idx[current]
		if debug:
			print "Starting at stop label %s (idx %d)" % (current, current_idx)

		for pos in range(len(emission_sequence)-1, 0, -1):
			current_idx = backtrack[pos][current_idx]
			free(backtrack[pos])

			if current_idx >= self.label_count:
				current_idx = arg_maxes[pos]

			current = self.idx_label[current_idx]

			if debug:
				print "pos %d => %s (idx %d)" % (pos, current, current_idx)
			states.append(current.split('::')[-1])

		free(backtrack)
		free(arg_maxes)

		# Pop all the extra stop states
		states.reverse()
		states = states[1:]
		states = states[:emission_length]

		return states
