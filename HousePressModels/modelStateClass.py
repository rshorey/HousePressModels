from numpy import *

class state:
	def __init__(self,indexToDoc,typeToIndex,indexToType,
			docLength,docTokens):
		self.indexToDoc = indexToDoc #dict from indices to doc names
		self.typeToIndex = typeToIndex #dict from word types to indices
		self.indexToType = indexToType #dict from indices to word types
		self.tokensInDoc = array(docLength) #list of document lengths
		self.numTokens = sum(self.tokensInDoc)
		self.docTokens = docTokens #list of lists of the types of the
					#tokens (in order) in a given document
		self.numTypes = len(self.typeToIndex)
		self.numDocs = len(self.indexToDoc)
		self.numIts = 0

	def assignTopics(self, numTops):
		self.numTops = numTops
		self.assignedTops = []
		self.typeByTop = zeros((self.numTypes,self.numTops))
		self.tokensInTop = zeros(numTops)
		self.docByTop = zeros((self.numDocs,numTops))
		for d in xrange(0,self.numDocs):
			self.assignedTops.append([])
			num = self.tokensInDoc[d]
			for i in xrange(0,num):
				currentType = self.docTokens[d][i]
				t = random.randint(numTops)
				self.assignedTops[d].append(t)
				self.typeByTop[currentType][t] += 1
				self.tokensInTop[t] += 1
				self.docByTop[d][t] += 1
			

	def assignClusters(self, zeta):
		numClusts = 0
		self.assignedClusts = []
		self.docsInClust = zeros(numClusts)
		for d in xrange(self.numDocs):
			probs = copy(self.docsInClust)
			probs = append(probs,zeta)
			probs /= sum(probs)
			c = where(random.multinomial(1,probs)==1)[0][0]
			if c==numClusts:
				self.docsInClust = append(self.docsInClust,0)
				numClusts += 1
			self.docsInClust[c] += 1
			self.assignedClusts.append(c)
		self.clustDocsUsingTop = zeros((numClusts,self.numTops))
		self.clustsUsingTop = zeros(self.numTops)
		self.numClusts = numClusts
		for tp in xrange(self.numTops):
			for dc in xrange(self.numDocs):
				currentClust = self.assignedClusts[dc]
				if self.docByTop[dc][tp] > 0:
					self.clustDocsUsingTop[currentClust][tp]+=1
			self.topClustDocs = self.clustDocsUsingTop[:,tp]
			self.clustsUsingTop[tp] = sum(where(
				self.topClustDocs==0,0,1))


