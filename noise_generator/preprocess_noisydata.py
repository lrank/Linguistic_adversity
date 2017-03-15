import numpy as np
import re
import sys


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
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


def load_data(data_filename):
    text = list(open(data_filename, "r").readlines())
    text = [s.strip() for s in text]
    # print text[0]

    i = 0
    res = []
    while i < len(text):
        doc = []
        # print i
        for j in range( int(text[i]) ):
            doc.append( text[i + j + 1] )
        i += int(text[i]) + 1


        if len(doc) > 200:
        	tmp = []
        	tmp.append( doc[ 0 ] )
        	for j in range(200):
        		tmp.append( doc[np.random.randint(1, len(doc))] )
        	doc = tmp
        
        doc = [clean_str(sent) for sent in doc]
        res.append( doc )
    return res

output_file = sys.argv[1]
text = load_data(output_file)
# print len(text)
for doc in text:
	print len(doc)
	for s in doc:
		print s
