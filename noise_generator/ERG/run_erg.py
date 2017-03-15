import os
from nltk.tokenize import word_tokenize
import re

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\']", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def run(filename):
	text = list(open(filename).readlines())
	text = [s.strip() for s in text]

	count = 0
	os.system("rm -f " + filename + ".res")
	for t in text:
		t = clean_str(t)
		if t == "":
			continue
		print count, '\t', t
		count += 1


		sent = word_tokenize( t )
		# print sent
		t = ""
		for i in sent:
			t = t + i + " "

		os.system("echo \"" + t + "\" | ./ace -g erg-1214-x86-64-0.9.23.dat -1T 2>/dev/null | python mapf.py | python mapb.py >>" + filename + ".res")


import sys
filename = sys.argv[1]

run(filename = filename)
