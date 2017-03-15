import nltk
from nltk.tag import map_tag
import re
from nltk.tokenize import word_tokenize
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import lin_thesaurus as linthes

from nltk.tag import StanfordNERTagger
import kenlm

import logging

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# Tag		Meaning					English Examples
# ADJ		adjective				new, good, high, special, big, local
# ADP		adposition				on, of, at, with, by, into, under
# ADV		adverb					really, already, still, early, now
# CONJ		conjunction				and, or, but, if, while, although
# DET		determiner, article		the, a, some, most, every, no, which
# NOUN		noun					year, home, costs, time, Africa
# NUM		numeral					twenty-four, fourth, 1991, 14:24
# PRT		particle				at, on, out, over per, that, up, with
# PRON		pronoun					he, their, her, its, my, I, us
# VERB		verb					is, say, told, given, playing, would
# .			punctuation marks		. , ; !
# X			other					ersatz, esprit, dunno, gr8, univeristy
def tag_filter(tag):
	if tag == "CONJ":
		return False
	elif tag == "ADP":
		return False
	elif tag == "DET":
		return False
	elif tag == "NUM":
		return False
	elif tag == "PRT":
		return False
	elif tag == ".":
		return False
	elif tag == "X":
		return False
	return True

def tag_wn(tag):
	if tag == 'n':
		return 'NOUN'
	elif tag == 'v':
		return 'VERB'
	elif tag == 'a' or tag == 's':
		return 'ADJ'
	elif tag == 'r':
		return 'ADV'
	else:
		return 'NONE'


import sys
filename = sys.argv[1]
lm_threshold = float( sys.argv[2] )

text = list(open(filename).readlines())
text = [s.strip() for s in text]

vocab = {}
for sentence in text:
	s = word_tokenize(clean_str(sentence))
	for w in s:
		vocab[w] = vocab.get(w, 0) + 1

st = StanfordNERTagger('/stanford-ner-2015-12-09/english.all.3class.nodistsim.crf.ser.gz',
	'/stanford-ner-2015-12-09/stanford-ner.jar')

lm = kenlm.Model('/Data/en-70k-0.2.lm')



def sample_by_lm(rep, post_s, index):
	word = post_s[index][0]

	a = max(index - 2, 0)
	b = min(index + 3, len(post_s))
	li = []
	for i in range(a, b):
		li.append( post_s[i][0] )
	t1 = " ".join(li)
	score = []
	candi = []

	for w in rep:
			li[index - a] = w
			t2 = " ".join(li)
			s2 = lm.score( t2 )
			score.append(s2)
			candi.append(w)

	score = np.array( score )
	score = np.exp( score / 2  - 5)
	t_score = np.sum( score )
	score = score / t_score
	
	s = np.random.random_sample()
	i = 0
	while s - score[i] > 0:
		s -= score[i]
		i += 1
	return candi[i]
	

for sentence in text:
	
	s = word_tokenize(clean_str(sentence))
	if s == ['']:
		continue

	print(201)
	print(sentence)
	ner_tag = st.tag(s)
	post = nltk.pos_tag(s)
	
	for _ in range(200):
		index = 0
		for (x, t) in post:
			t = map_tag('en-ptb', 'universal', t)

			synlist = [x]
			if tag_filter(t) and ner_tag[index][1] == 'O':
				for syn in wn.synsets(x):
					sss = syn.name().split('.')[0]
					ttt = syn.name().split('.')[1]
					if vocab.get(sss, 0) != 0 and t == tag_wn(ttt):
						synlist += [ str(sss) ]
				synlist = list(set(synlist))
			if np.random.random_sample() <= lm_threshold:
				print( sample_by_lm( synlist, post, index) ),
			else:
				print( x ),
			index += 1
	
		print
	