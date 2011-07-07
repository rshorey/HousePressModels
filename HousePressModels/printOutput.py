import sys
from modelStateClass import *
from numpy import *
import cPickle


#takes a state object and prints topic keywords, docByTop, docsInCluster,
#and clusterByTop distribution, all distributions unsmoothed.
#command line arguments:
#1) input pickled file
#2) output topic keywords file
#3) output docByTop file
#4) output docsInCluster file
#5) output clustByTop file

input = sys.argv[1]
topKeys = sys.argv[2]
docTop = sys.argv[3]
clustMember = sys.argv[4]
clustTop = sys.argv[5]

f = open(input,"r")
modelState = cPickle.load(f)
f.close()


#finds the indices of the top n elements in an array
def getTopN(counts,n):
	indices = []
	currentNum = len(indices)
	m = max(counts)
	while currentNum<n and m>1:
		if shape(where(counts==m)[0])[0]>0:
			newIndex = where(counts==m)[0][0]
			indices.append(newIndex)
			counts[newIndex] = 0
			currentNum += 1
		else:
			m -= 1

	return indices


typeByTop = modelState.typeByTop
indexToType = modelState.indexToType
numTops = modelState.numTops
tokensInTop = modelState.tokensInTop
topTokenNorm = sum(tokensInTop)
topPcts = tokensInTop/topTokenNorm


#write topic file
tkFile = open(topKeys,"w")
tkFile.write("top\tpct\tkey\n")

for t in xrange(numTops):
	print t
	topRow = typeByTop[:,t]
	topIndices = getTopN(topRow,20)
	tkFile.write(str(t))
	tkFile.write("\t")
	tkFile.write(str(topPcts[t]))
	tkFile.write("\t")
	for i in topIndices:
		type = indexToType[i]
		tkFile.write(type)
		tkFile.write(" ")
	tkFile.write("\n")
tkFile.close()

docByTop = modelState.docByTop
indexToDoc = modelState.indexToDoc
numDocs = modelState.numDocs
docLength = modelState.tokensInDoc

#write doc-topic file
dtFile = open(docTop,"w")
dtFile.write("doc")
for t in xrange(numTops):
	dtFile.write(",")
	dtFile.write(str(t))
dtFile.write("\n")

for d in xrange(numDocs):
	if d%10==0:
		print d
	lenD = docLength[d]*1.0
	docName = indexToDoc[d]
	dtFile.write(docName)
	for t in xrange(numTops):
		dtFile.write(",")
		num = docByTop[d][t]*1.0
		pct = num/lenD
		dtFile.write(str(pct))
	dtFile.write("\n")
dtFile.close()



assignedClusts = modelState.assignedClusts
clustDict = {}
numClusts = modelState.numClusts

#write cluster membership file
cmFile = open(clustMember,"w")
cmFile.write("Cluster Number\tPct\tDocuments\n")

for d in xrange(numDocs):
	currentClust = assignedClusts[d]
	if currentClust not in clustDict:
		clustDict[currentClust] = []
	clustDict[currentClust].append(d)
for c in xrange(numClusts):
	print c
	clustDocs = len(clustDict[c])*1.0
	pct = clustDocs/numDocs
	cmFile.write(str(c))
	cmFile.write("\t")
	cmFile.write(str(pct))
	cmFile.write("\t")
	for d in clustDict[c]:
		cmFile.write(indexToDoc[d])
		cmFile.write(" ")
	cmFile.write("\n")
cmFile.close()



#write clustByTop file
clustTops=zeros((numClusts,numTops))
for d in xrange(0,numDocs):
	currentClust = assignedClusts[d]
	clustTops[currentClust] += docByTop[d]
norms = sum(clustTops,axis = 1)

ctFile = open(clustTop,"w")
for c in xrange(-1,numClusts):
	if c == -1:
		ctFile.write("Clust")
	else:
		ctFile.write(str(c))
		clustTops[c]/=norms[c]
	for t in xrange(numTops):
		ctFile.write(",")
		if c == -1:
			ctFile.write(str(t))
		else:
			ctFile.write(str(clustTops[c][t]))
	ctFile.write("\n")


ctFile.close()


	
