import sys
import os
from modelStateClass import *
from cPickle import dump

#inputs
#1) input file (mallet state)
#2) name of output pickle file


inputfilename = sys.argv[1]
outputfilename = sys.argv[2]

indexToDoc = {}
typeToIndex = {}
indexToType = {}
docTokens = []
docLengths = []
assignedTops = []
maxTop=0

f = open(inputfilename)
linenum = 0
prevDocNum = -1
missingDocs=0
for line in f:
	if linenum>2:
		s=line.split()
		docNum=int(s[0])-missingDocs
		docName=s[1]
		typeInd=int(s[3])
		type=s[4]
		top=int(s[5])
		if docNum not in indexToDoc:
			if prevDocNum!=docNum-1:
				missingDocs+=docNum-prevDocNum
				docNum-=docNum-prevDocNum
			prevDocNum=docNum
			indexToDoc[docNum] = docName
			docLengths.append(0)
			docTokens.append([])
			assignedTops.append([])
		if typeInd not in indexToType:
			indexToType[typeInd] = type
			typeToIndex[type] = typeInd
		docTokens[docNum].append(typeInd)
		docLengths[docNum] += 1
		assignedTops[docNum].append(top)
		if top>maxTop:
			maxTop=top
	linenum+=1
f.close()

modelState = state(indexToDoc,typeToIndex,indexToType,docLengths,docTokens)
modelState.processInitializedTopics(maxTop+1,assignedTops)
outfile=open(outputfilename,"w")
dump(modelState,outfile)
outfile.close()


