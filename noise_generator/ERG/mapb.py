from nltk.tokenize import word_tokenize
from random import randint
import random
from datetime import datetime

random.seed(datetime.now())

# f = open("map.info.tmp")
n = input()
# print n
d = {}
# text= list(f.readlines())
# text = [s.strip() for s in text]

for i in range(n):
	t = raw_input()
	t = t.split(" ")
	d[t[0]] = t[1]
# f.close()

s = []

while True:
	try:
		sent = raw_input()
		s.append(str(sent))

	except (EOFError):
		break

# print len(s)
print len(s)
for index in range( len(s) ):
# index = 0
# if len(s) != 1:
# 	index = randint(1, len(s) - 1)
# print index

	sent = s[index].lower()
	# sent = sent.split(" ")
	sent = word_tokenize(sent.decode('ascii', 'ignore'))
	for token in sent:
		if token in d:
			print d[token],
		else:
			print token,
	print
