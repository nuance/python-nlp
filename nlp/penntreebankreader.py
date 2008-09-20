from os import walk
from os.path import join
import re

class PennTreebankReader:
	@classmethod
	def read_pos_tags_from_directory(cls, path):
		for root, dirs, files in walk(path):
			for file in files:
				trees = cls.load_pos_tags(join(root, file))
				for tree in trees: yield tree

	@classmethod
	def load_pos_tags(cls, file_path):
		tree_file = open(file_path)

		tree_split_re = re.compile("\n\(")
		tags_re = re.compile("\([^\(\)]+\)")
		trees = tree_split_re.split(tree_file.read())

		raw_tags = []
		for tree in trees:
			if len(tree) == 0: continue
			tree = "".join(line.rstrip() for line in tree.split("\n"))
			raw_tags.append(tags_re.findall(tree))

		sentences = []
		for sentence in raw_tags:
			tags, words = [], []
			for pair in sentence:
				tag, word = pair.strip("()").split()
				tags.append(tag)
				words.append(word)
			if len(tags) > 0:
				sentences.append((tags, words))

		tree_file.close()

		return sentences
