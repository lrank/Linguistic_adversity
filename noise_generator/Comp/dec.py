import os
# from nltk.parse.stanford import GenericStanfordParser, StanfordDependencyParser, StanfordParser
# from nltk.parse.stanford import StanfordNeuralDependencyParser
from nltk.parse.stanford import StanfordParser
from nltk.tree import Tree
from collections import defaultdict
# from nltk.tokenize import StanfordTokenizer
from nltk import word_tokenize
import re
import numpy as np

C = defaultdict(dict)
# T = defaultdict(dict)

def load_model():
	text = list(open("written.model").readlines())
	text = [s.strip() for s in text]

	for s in text:
		l = s.split(" ")
		s1 = l[0]
		s2 = l[1]
		c = float(l[2]) / float(l[3])
		C[s1][s2] = C.get(s1, {}).get(s2, c)

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def decoder(r, f, k):

	if (type(r) != Tree):
		print r,
		return

	thresh = C.get(f, {}).get(r.label(), -1)
	p = np.random.random_sample()
	# print p, thresh
	if thresh != -1 and p < thresh:
		return

	for i in range(0, len(r)):
		decoder(r[i], r.label(), k + 1)

dep_parser = StanfordParser(path_to_jar="./stanford-parser.jar", path_to_models_jar="./stanford-models.jar")

load_model()

import sys
filename = sys.argv[1]
text = list(open(filename).readlines())
text = [s.strip() for s in text]

for i in range(len(text)):

	s1 = clean_str(text[i])
	if s1 == "":
		continue

	print 201
	print text[i]

	a = list(dep_parser.raw_parse(s1))
	for _ in range(200):
		decoder(a[0], "root", 0)
		print
