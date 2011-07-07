import sys
import os
import cPickle
from string import punctuation, digits
from collections import defaultdict
from modelStateClass import state


#this script reads data, removes punctuation and caps, removes words on
#a provided stopword list, and creates and pickles a state object.

#takes command line arguments:
#1) directory containing the rep-files
#2) file containing stopword list
#3) name of output pickle file
#4) optional: # of words to snip from the front of each doc


dir = sys.argv[1]
stopwords = sys.argv[2]
out = sys.argv[3]
if len(sys.argv)>4:
	snip = int(sys.argv[4])
else:
	snip = 0

swFile = open(stopwords,'r')
sw = swFile.read().split()

repDirs = os.listdir(dir)

pAndD = punctuation+digits

docIndex = 0
typeIndex = 0
indexToDoc = {}
indexToType = {}
typeToIndex = {}
docByType = [] #this will be an array of dictionaries to keep things sparse
docTokens = []
docLength = []

for rd in repDirs:
	if os.path.isdir(dir+rd):
		print rd
		repFiles = os.listdir(dir+rd)
		for file in repFiles:
			if os.path.isfile(dir+rd+"/"+file):
				indexToDoc[docIndex] = file
				docByType.append(defaultdict(int))
				docLength.append(0)
				docTokens.append([])
				f=open(dir+rd+"/"+file,"r")
				text = f.read().split()
				text = text[snip:]
				for token in text:
					token = token.lower()
					#remove punctuation and digits:
					token = ''.join(c for c in token if
						c not in pAndD)
					if (token not in sw) and token != '':
						if token in typeToIndex:
							ti = typeToIndex[token]
						else:
							ti = typeIndex
							typeIndex += 1
							typeToIndex[token] = ti
							indexToType[ti] = token
						docByType[docIndex][ti]+=1
						docLength[docIndex]+=1
						docTokens[docIndex].append(ti)
				docIndex += 1

modelState = state(indexToDoc,typeToIndex,indexToType,docLength,docTokens)

picklefile = open(out,'w')
cPickle.dump(modelState,picklefile,2)
picklefile.close()
