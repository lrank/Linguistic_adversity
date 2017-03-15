import numpy as np
import re
from nltk.tag import StanfordNERTagger
st = StanfordNERTagger('/stanford-ner-2015-12-09/english.all.3class.nodistsim.crf.ser.gz', '/stanford-ner-2015-12-09/stanford-ner.jar')

import kenlm
lm = kenlm.Model('/Data/en-70k-0.2.lm')

# np.random.seed(1001003)

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

text = list(open("overfitting.dict").readlines())
text = [s.strip() for s in text]

a = dict()

for i in range(1, len(text), 2):
	s = text[i]
	t = text[i + 1].split("\t")
	a[s] = t[0:5]

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

    #LM
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

def proc(filename, lm_threshold = 0.5):
	text = list(open(filename).readlines())
	text = [s.strip() for s in text]

	for t in text:
		index = 0
		t = clean_str(t)
		if t == '':
			continue

		print(201)
		print(t)

		sent = t.split(" ")
		ner_tag = st.tag(sent)
		for _ in range(200):
			index = 0
			for k in sent:
				if a.get(k, 0) != 0 and ner_tag[index][1] == 'O':
					if np.random.random_sample() <= lm_threshold:
						print sample_by_lm(a[k], sent, index),
					else:
						print k,
				else:
					print k,
				index += 1
			print

import sys
filename = sys.argv[1]
lm_threshold = float( sys.argv[2] )
proc(filename = filename, lm_threshold = lm_threshold)
