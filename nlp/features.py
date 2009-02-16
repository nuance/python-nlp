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
			history.pop(0)
			history.append(chunk)
			yield copy(history)

	if stop_token:
		for min_size in xrange(size+1):
			for history in histories:
				if len(history) <= min_size + 1:
					# size = 3, on sub_size 1, don't return '<STOP>' 3 times
					continue
				history.pop(0)
				history.append(stop_token)
				yield copy(history)
