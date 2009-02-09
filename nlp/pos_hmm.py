from itertools import islice, izip
import cPickle as pickle
import sys
from sys import stdout
from time import time

from hmm import HiddenMarkovModel, START_LABEL, STOP_LABEL
from penntreebankreader import PennTreebankReader
from naivebayes import NaiveBayesClassifier
from maximumentropy import MaximumEntropyClassifier

def merge_stream(stream):
	# Combine sentences into one long string, with each sentence start with <START> and ending with <STOP>
	# [1:-2] cuts the leading STOP_LABEL and the trailing START_LABEL
	sentences = []
	tag_stream = []
	for tags, sentence in stream:
		sentences.append(START_LABEL)
		tag_stream.append(START_LABEL)
		for word in sentence:
			sentences.append(word)
		for tag in tags:
			tag_stream.append(tag)
		sentences.append(STOP_LABEL)
		tag_stream.append(STOP_LABEL)

	return zip(tag_stream, sentences)

def pos_problem(arguments, fallback_model=None, fallback_training_limit=None):
	dataset_size = None
	if len(arguments) >= 2: dataset_size = int(arguments[1])
	if len(arguments) >= 3: fallback_training_limit = int(arguments[2])

	try:
		start = time()
		pickle_file = open("pos_hmm.pickle")
		request_size, training_stream, validation_stream, testing_sentences = pickle.load(pickle_file)
		pickle_file.close()

		if request_size != dataset_size: raise IOError()

		print "Unpickling: %f" % (time() - start)
	except (IOError, EOFError):
		# Load the dataset
		print "Loading dataset"
		start = time()
		if dataset_size: tagged_sentences = list(islice(PennTreebankReader.read_pos_tags_from_directory("data/wsj"), dataset_size))
		else: tagged_sentences = list(PennTreebankReader.read_pos_tags_from_directory("data/wsj"))
		stop = time()
		print "Reading: %f" % (stop-start)

		print "Creating streams"
		start = time()
		training_sentences = tagged_sentences[0:len(tagged_sentences)*4/5]
		validation_sentences = tagged_sentences[len(tagged_sentences)*8/10+1:len(tagged_sentences)*9/10]
		testing_sentences = tagged_sentences[len(tagged_sentences)*9/10+1:]

		print "Training: %d" % len(training_sentences)
		print "Validation: %d" % len(validation_sentences)
		print "Testing: %d" % len(testing_sentences)

		training_stream, validation_stream = map(merge_stream, (training_sentences, validation_sentences))
		stop = time()
		print "Streaming: %f" % (stop-start)

		serialized = (dataset_size, training_stream, validation_stream, testing_sentences)
		pickle_file = open("pos_hmm.pickle", "w")
		pickle.dump(serialized, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
		pickle_file.close()

	print "Training"
	start = time()
	pos_tagger = HiddenMarkovModel(label_history_size=2)
	pos_tagger.train(training_stream[1:-2], fallback_model=fallback_model, fallback_training_limit=fallback_training_limit)
	stop = time()
	print "Training: %f" % (stop-start)

	print "Testing on %d sentences" % len(testing_sentences)
	start = time()

	num_correct = 0
	num_incorrect = 0

	for correct_labels, emissions in testing_sentences:
		guessed_labels = pos_tagger.label(emissions, debug=False)
#		print "SENTENCE: %s" % emissions
#		print "CORRECT: %s" % correct_labels
#		print "GUESSED: %s" % guessed_labels
		for correct, guessed in izip(correct_labels, guessed_labels):
			if correct == START_LABEL or correct == STOP_LABEL: continue
			if correct == guessed:
				num_correct += 1
			else:
				num_incorrect += 1

		if correct_labels != guessed_labels:
			guessed_score = pos_tagger.score(zip(guessed_labels, emissions))
			correct_score = pos_tagger.score(zip(correct_labels, emissions))

			if guessed_score < correct_score: print "%d Guessed: %f, Correct: %f" % (len(emissions), guessed_score, correct_score)

			debug_label = lambda: pos_tagger.label(emissions, debug=True)
			debug_score = lambda labels: pos_tagger.score(zip(labels, emissions), debug=False)
			assert guessed_score >= correct_score or len(emissions) > 23, "Decoder sub-optimality (%f for guess, %f for correct)\n%s vs. %s" % (debug_score(guessed_labels), debug_score(correct_labels), debug_label(), correct_labels)

			stdout.write("%1.0f" % (sum(1 for guess, correct in zip(guessed_labels, correct_labels) if guess == correct) / len(correct_labels) * 10))
			stdout.flush()
		else:
			stdout.write(".")
			stdout.flush()
	stdout.write("\n")

	stop = time()
	print "Testing: %f" % (stop-start)

	print "%d correct (%.3f%% of %d)" % (num_correct, 100.0 * float(num_correct) / float(num_correct + num_incorrect), num_correct + num_incorrect)

if __name__ == "__main__":
	pos_problem(sys.argv, fallback_model=NaiveBayesClassifier)
