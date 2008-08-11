from nlp import counter

def test():
	all_spam = counter()
	all_spam['spam'] = 2

	assert(all_spam.total_count() == 2)
	assert(all_spam.arg_max() == 'spam')
	
	all_spam.normalize()

	assert(all_spam['spam'] == 1.0)
	assert(all_spam['ham'] == 0.0)
	assert(all_spam.arg_max() == 'spam')

	assert('spam' in all_spam.keys())
	assert('ham' in all_spam.keys())

	assert(len(all_spam.keys()) == 2)
	assert(len(all_spam.values()) == 2)

	print "All spam: %s" % all_spam
	
	del(all_spam)

	half_spam = counter()
	half_spam['spam'] += 1
	half_spam['ham'] += 1

	assert(half_spam.total_count() == 2)
	assert(len(half_spam.keys()) == 2)
	assert(half_spam.arg_max() in ('spam', 'ham'))

	half_spam.normalize()

	assert(half_spam['spam'] == 0.5)
	assert(half_spam['ham'] == 0.5)
	assert(half_spam.arg_max() in ('spam', 'ham'))
	assert(len(half_spam.keys()) == 2)

	print "Half spam: %s" % half_spam

	del(half_spam)

	amul = counter()
	amul['bob'] = 2
	amul['jim'] = 2
	bmul = counter()
	bmul['bob'] = 4

	amul *= bmul
	bmul *= amul

	assert(amul['bob'] == 8)
	assert(bmul['bob'] == 32)
	assert(amul['jim'] == 0)
	assert(bmul['jim'] == 0)

	del(amul)
	del(bmul)
	
	aadd = counter()
	aadd['bob'] = 2
	badd = counter()
	badd['bob'] = 4

	aadd += badd

	assert(aadd['bob'] == 6)
	assert(badd['bob'] == 4)
	

if __name__ == "__main__":
	test()
