from collections import defaultdict

class Counter(defaultdict):

	def __init__(self):
		super(Counter, self).__init__(lambda:0.0)

	def __init__(self, *args):
		super(Counter, self).__init__(lambda:0.0, *args)

	# I feel like there's a better way to do this...
	def arg_max(self):
		(max_key, max_value) = (None, None)

		for (key, value) in self.iteritems():
			if not max_value or value > max_value:
				(max_key, max_value) = (key, value)

		return max_key

	def total_count(self):
		return sum(self.itervalues())

	def normalize(self):
		sum = self.total_count()

		for (key, value) in self.items():
			self[key] = value / sum

	def __str__(self):
		return "[%s]" % (" ".join(["%s : %f," % (key, value) for (key, value) in self.iteritems()]))

	def __imul__(self, other):
		for (key, value) in other.items():
			self[key] *= value

		return self

	def __iadd__(self, other):
		for (key, value) in other.items():
			self[key] += value

		return self

def test():
	all_spam = Counter()
	all_spam['spam'] = 2

	assert(all_spam.total_count() == 2)
	assert(all_spam.arg_max() == 'spam')
	
	all_spam.normalize()

	assert(all_spam['spam'] == 1.0)
	assert(all_spam['ham'] == 0.0)
	assert(all_spam.arg_max() == 'spam')

	print "All spam: %s" % all_spam
	
	del(all_spam)

	half_spam = Counter()
	half_spam['spam'] += 1
	half_spam['ham'] += 1

	assert(half_spam.total_count() == 2)
	assert(half_spam.arg_max() in ('spam', 'ham'))

	half_spam.normalize()

	assert(half_spam['spam'] == 0.5)
	assert(half_spam['ham'] == 0.5)
	assert(half_spam.arg_max() in ('spam', 'ham'))

	print "Half spam: %s" % half_spam

	del(half_spam)

	amul = Counter()
	amul['bob'] = 2
	bmul = Counter()
	bmul['bob'] = 4

	amul *= bmul

	assert(amul['bob'] == 8)
	assert(bmul['bob'] == 4)

	del(amul)
	del(bmul)
	
	aadd = Counter()
	aadd['bob'] = 2
	badd = Counter()
	badd['bob'] = 4

	aadd += badd

	assert(aadd['bob'] == 6)
	assert(badd['bob'] == 4)
	

if __name__ == "__main__":
	test()
