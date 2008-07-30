from collections import defaultdict
from counter import Counter

class CounterMap(defaultdict):
	def __init__(self):
		super(CounterMap, self).__init__(lambda:Counter())

	def normalize(self):
		for key in self.iterkeys():
			self[key].normalize()

	def __str__(self):
		string = ""
		for (key, counter) in self.iteritems():
			string += "%s : %s\n" % (key, counter)
		return string.rstrip()

def test():
	one_all_spam = CounterMap()
	one_all_spam['xxx']['spam'] = 2

	assert(one_all_spam['xxx'].total_count() == 2)
	assert(one_all_spam['xxx'].arg_max() == 'spam')
	
	one_all_spam.normalize()

	assert(one_all_spam['xxx']['spam'] == 1.0)
	assert(one_all_spam['xxx']['ham'] == 0.0)
	assert(one_all_spam['cheese']['ham'] == 0.0)

	print "All spam: %s" % one_all_spam
	
	del(one_all_spam)

	half_spam = CounterMap()
	half_spam['xxx']['spam'] += 1
	half_spam['xxx']['ham'] += 1
	half_spam['cheese']['spam'] += 1
	half_spam['cheese']['ham'] += 1

	half_spam.normalize()

	assert(half_spam['xxx']['spam'] == 0.5)
	assert(half_spam['cheese']['spam'] == 0.5)

	print "Half spam: %s" % half_spam

	del(half_spam)

if __name__ == "__main__":
	test()
