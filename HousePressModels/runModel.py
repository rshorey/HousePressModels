import cPickle
from numpy import *
import modelStateClass
import sys

#takes the following command line arguments:
#1) filename of "pickled" file of model state with data read in but
	#topics/clusts not yet initialized
#2-6) initial values for alpha0, alpha1, alpha, beta, zeta
#7) number of topics
#8) number of iterations
#9) filename for output pickle file of state.

def getState(input):
	#unpickles the state that is saved in the file called "input"
	f=open(input,"r")
	modelState = cPickle.load(f)
	f.close()
	return modelState

def initialize(modelState,zeta,numTops):
	#makes initial cluster and topic assignments
	modelState.assignTopics(numTops)
	modelState.assignClusters(zeta)
	return modelState

def gibbs(modelState,alpha0,alpha1,alpha,beta,zeta):
	typeByTop = modelState.typeByTop
	tokensInTop = modelState.tokensInTop
	docByTop = modelState.docByTop
	tokensInDoc = modelState.tokensInDoc #observed
	docTokens = modelState.docTokens #observed
	clustDocsUsingTop = modelState.clustDocsUsingTop
	docsInClust = modelState.docsInClust
	clustsUsingTop = modelState.clustsUsingTop
	assignedTops = modelState.assignedTops
	assignedClusts = modelState.assignedClusts
	numClusts = modelState.numClusts
	numDocs = modelState.numDocs
	numTops = modelState.numTops
	numTypes = modelState.numTypes


	#assign a new topic to each word
	for d in xrange(numDocs):
		currentClust = assignedClusts[d]
		numT = tokensInDoc[d]
		recomputePerDoc = 0

		#this will nearly always be the same for each word in a doc.
		#we'll only recompute it if necessary
		docTopFrac = clustsUsingTop+alpha0/numTops
		docTopFrac /= (numClusts + alpha0)

		#also frequently unchanged
		docMidFrac = clustDocsUsingTop[currentClust]+alpha1*docTopFrac
		docMidFrac /= docsInClust[currentClust] + alpha1

		for i in xrange(numT):
			currentType = docTokens[d][i]
			currentTop = assignedTops[d][i]


			#remove current token:

			#remove one from clustDocsUsingTop if there is
			#only one word in the current doc and the current top
			if docByTop[d][currentTop] ==  1:
				recomputePerDoc += 1
				#remove one from clustsUsingTop if there is
				#also only one document using the current topic
				#in this cluster
				if (clustDocsUsingTop[currentClust][currentTop]
					 == 1):
					clustsUsingTop[currentTop] -= 1
					#recompute the top fraction if anything
					#has changed
					docTopFrac = (clustsUsingTop
						+alpha0/numTops)
					docTopFrac /= (numClusts + alpha0)

				clustDocsUsingTop[currentClust][currentTop] -= 1
				#recompute the middle frac if anything changed
				docMidFrac = (clustDocsUsingTop[currentClust]+
					alpha1*docTopFrac)
				docMidFrac /= docsInClust[currentClust] + alpha1

			typeByTop[currentType][currentTop] -= 1
			tokensInTop[currentTop] -= 1
			docByTop[d][currentTop] -= 1

			#type contrib:
			topicProbs = log(typeByTop[currentType]+beta/numTypes)
			topicProbs -= log(tokensInTop+beta)

			#doc contrib:
			topicProbs += log(docByTop[d] + alpha*docMidFrac)
			topicProbs -= log(tokensInDoc[d] + alpha)



			#normalize and exponentiate topic probs
			m = max(topicProbs)
			topicProbs -= m
			topicProbs = exp(topicProbs)
			norm = sum(topicProbs)
			topicProbs /= norm
			sample = random.multinomial(1,topicProbs)
			newTop = where(sample==1)[0][0]

			#assign everything to the appropriate topic
			if docByTop[d][newTop] == 0:
				recomputePerDoc += 1
				if clustDocsUsingTop[currentClust][newTop] == 0:
					clustsUsingTop[newTop] += 1
					#recompute the top fraction if anything
					#has changed
					docTopFrac = (clustsUsingTop
						+alpha0/numTops)
					docTopFrac /= (numClusts + alpha0)
				clustDocsUsingTop[currentClust][newTop] += 1
				docMidFrac = (clustDocsUsingTop[currentClust]+
					alpha1*docTopFrac)
				docMidFrac /= docsInClust[currentClust] + alpha1

			typeByTop[currentType][newTop] += 1
			tokensInTop[newTop] += 1
			docByTop[d][newTop] += 1
			assignedTops[d][i] = newTop
		#print i
		#print recomputePerDoc
	
	print "topics sampled"

	print numClusts
	#print clustDocsUsingTop
	#assign a new cluster to each document
	for d in xrange(numDocs):
		currentClust = assignedClusts[d]
		#remove doc from cluster:
		docsInClust[currentClust] -= 1
		#this is the # we'll want to subtract:
		#it'll be 1 if there's only one document in the topic
		#and zero otherwise
		onlyDoc = where(clustDocsUsingTop[currentClust] == 1,1,0)
		#here's what we actually subtract
		subtract = onlyDoc[assignedTops[d]]
		clustsUsingTop[assignedTops[d]] -= subtract
		#the following will be 1 if the doc contains the top, 0 o'wise
		docsUsingTop = where(docByTop[d] > 0,1,0)
		sub2 = docsUsingTop[assignedTops[d]]
		clustDocsUsingTop[currentClust][assignedTops[d]] -= sub2
		
		#remove rows from empty cluster:
		if docsInClust[currentClust] == 0:
			#print "cluster removed"
			deadClust = currentClust
			docsInClust = delete(docsInClust,deadClust)
			clustDocsUsingTop = delete(clustDocsUsingTop,
				deadClust,axis=0)
			numClusts -= 1
			for tmpd in xrange(numDocs):
				if assignedClusts[tmpd]>=deadClust:
					assignedClusts[tmpd] -= 1 

		#compute probabilities
		#clust contrib:
		clustProbs = log(docsInClust)
		#print shape(clustDocsUsingTop)
		#print shape(docsInClust)

		tokenTops = assignedTops[d]	


		#doc contrib:

		topFrac = (clustsUsingTop[tokenTops] + alpha0/numTops)	
		topFrac /= (numClusts + alpha0)


		midFrac = transpose(clustDocsUsingTop[:,tokenTops]
			+ alpha1*topFrac)
		midFrac /= (docsInClust  + alpha1)


		tokenProbs = log(docByTop[d][tokenTops] +
			alpha*transpose(midFrac))
		tokenProbs -= log(tokensInDoc[d] + alpha)
		#print sum(tokenProbs,axis=1)
		#print clustProbs			

		clustProbs += sum(tokenProbs,axis=1)

		#add on prob for "new cluster"
		topFrac = (clustsUsingTop[tokenTops] + 
			 alpha0/numTops)	
		topFrac /= (numClusts  + alpha0)


		midFrac = alpha1*topFrac
		midFrac /= (alpha1)

		tokenProbs = log(docByTop[d][tokenTops]+alpha*midFrac)
		tokenProbs -= log(tokensInDoc[d] + alpha)

		clustProbs = append(clustProbs,sum(tokenProbs)+log(zeta))

		#sample new cluster
		m = max(clustProbs)
		clustProbs -= m
		clustProbs = exp(clustProbs)
		norm = sum(clustProbs)
		clustProbs /= norm
		sample = random.multinomial(1,clustProbs)
		newClust = where(sample==1)[0][0]
		#set up rows for new cluster
		if newClust == numClusts:
			#print "new cluster created"
			numClusts += 1
			docsInClust = append(docsInClust,0)
			clustDocsUsingTop = vstack([clustDocsUsingTop,
				[0]*numTops])		
		

		#add doc to new cluster
		docsInClust[newClust] += 1
		onlyDoc = where(clustDocsUsingTop[newClust] == 0,1,0)
		add = onlyDoc[assignedTops[d]]
		clustsUsingTop[assignedTops[d]] += add

		docsUsingTop = where(docByTop[d] > 0,1,0)
		add2 = docsUsingTop[assignedTops[d]]
		clustDocsUsingTop[newClust][assignedTops[d]] += add2
		assignedClusts[d] = newClust



	modelState.docsInClust = docsInClust
	modelState.clustDocsUsingTop = clustDocsUsingTop
	modelState.numClusts = numClusts
	modelState.assignedClusts = assignedClusts
		
	print "clusters sampled"

	return modelState



def iterate(modelState,alpha0,alpha1,alpha,beta,zeta,numIts):
	i=0
	while i<numIts:
		print i
		modelState.numIts = i
		if i%10 == 0:
			writeState(modelState,"/m/canvas1/rshorey/Grimmer/test_checkpoint.dat")
		modelState = gibbs(modelState,alpha0,alpha1,alpha,beta,zeta)
		i += 1
	return modelState	

def writeState(modelState,output):
	f=open(output,"w")
	cPickle.dump(modelState,f)
	f.close()

#main
input = sys.argv[1]
alpha0 = float(sys.argv[2])
alpha1 = float(sys.argv[3])
alpha = float(sys.argv[4])
beta = float(sys.argv[5])
zeta = float(sys.argv[6])
numTops = int(sys.argv[7])
numIts = int(sys.argv[8])
output = sys.argv[9]

print("classes loaded")
modelState = getState(input)
print("data loaded")
modelState = initialize(modelState,zeta,numTops)
print("topics and clusters initialized")
modelState = iterate(modelState,alpha0,alpha1,alpha,beta,zeta,numIts)
writeState(modelState,output)


