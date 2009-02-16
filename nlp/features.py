from copy import copy

def ngrams(datum, size, start_token=None, stop_token=None):
	"""
	pull apart datum into component chunks
	"""
	if start_token:
		histories = [[start_token for _ in xrange(sub_size)] for sub_size in xrange(1, size+1)]
	else:
		histories = [[] for _ in xrange(1, size+1)]

	for chunk in datum:
		for history in histories:
			if len(history): history.pop(0)
			history.append(chunk)
			yield copy(history)

	if stop_token:
		for min_size in xrange(size+1):
			for history in histories:
				if len(history) <= min_size + 1:
					# size = 3, on sub_size 1, don't return '<STOP>' 3 times
					continue
				if len(history): history.pop(0)
				history.append(stop_token)
				yield copy(history)

def contexts(datum, context_size=2):
	buffer = []
	# this could be just .__iter__(), but f*cking strings are a special, non-standard case that have to work
	datum = (i for i in datum)

	# Fill up the buffer
	for item in datum:
		buffer.append(item)
		if len(buffer) == context_size * 2 + 1: break

	for item in datum:
		yield (tuple(buffer[:context_size]), buffer[context_size], tuple(buffer[context_size+1:]))
		buffer.pop(0)
		buffer.append(item)

	if len(buffer) == context_size * 2 + 1:
		yield (tuple(buffer[:context_size]), buffer[context_size], tuple(buffer[context_size+1:]))
