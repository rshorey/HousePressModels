from numpy import *
from numpy import random


#params is a numpy array of the hyperparameters alpha0,alpha1,alpha

def alphaLikelihood(params,modelState):
	logProb = 0
	assignedClusts = modelState.assignedClusts
	numClusts = modelState.numClusts
	numTops = modelState.numTops
	newNumClusts = 0
	docsInClust = modelState.docsInClust
	newDocsInClust = zeros_like(docsInClust)
	docByTop = modelState.docByTop
	newDocByTop = zeros_like(docByTop)
	newClustDocsUsingTop = zeros((numClusts,numTops))
	newClustsUsingTop = zeros(numTops)
	tokensInDoc = modelState.tokensInDoc
	newTokensInDoc = zeros_like(tokensInDoc)
	a0 = params[0]*1.0
	a1 = params[1]*1.0
	a = params[2]*1.0
	tf = 0
	mf = 0
	for c in xrange(numClusts):
		clustArray = array(assignedClusts)
		newNumClusts += 1
		docs  = where(clustArray == c)[0]
		for d in docs:
			newDocsInClust[c] += 1
			for t in xrange(numTops):
				numTopTokens = int(docByTop[d][t])
				for i in xrange(numTopTokens):
					newTokensInDoc[d] += 1
					if newClustDocsUsingTop[c][t]==0:
						newClustsUsingTop[t]+=1
						tf = (newClustsUsingTop[t] + 
							a0/numTops)
						tf /= (newNumClusts + a0)
					if i==0:
						newClustDocsUsingTop[c][t] += 1
						mf=(newClustDocsUsingTop[c][t]
							+a1*tf)
						mf /= (newDocsInClust[c] + a1)
					newDocByTop[d][t] += 1
					logProb += log(newDocByTop[d][t] + a*mf)
					logProb -= log(newTokensInDoc[d] + a)

	#print logProb
	return logProb


def betaLikelihood(beta,modelState):
	numTypes = modelState.numTypes
	numTops = modelState.numTops
	typeByTop = modelState.typeByTop
	tokensInTop = modelState.tokensInTop
	newTypeByTop = zeros_like(typeByTop)
	newTokensInTop = zeros_like(tokensInTop)

	logprob = 0
	for type in xrange(numTypes):
		for top in xrange(numTops):
			numTopTokens = int(typeByTop[type][top])
			for i in xrange(numTopTokens):
				newTypeByTop[type][top] += 1
				newTokensInTop[top] += 1
				logprob += log(newTypeByTop[type][top]
					+ beta*1.0/numTypes)
				logprob -= log(newTokensInTop[top] + beta)
	#print logprob
	return logprob

def zetaLikelihood(zeta,modelState):
	numClusts = modelState.numClusts
	numDocs = modelState.numDocs
	docsInClust = modelState.docsInClust
	logprob = log(zeta)*numClusts
	for c in xrange(numClusts):
		#numpy doesn't have a factorial, but this'll work
		logprob += sum(log(arange(1,docsInClust[c])))
	docRange = arange(1,numDocs+1)
	logprob -= sum(log(zeta+docRange-1))
	#print logprob
	return logprob
			
def sampleAlpha(params,stepVec,modelState,numIts):
	numParams = shape(params)[0]
	rawParams = log(params)
	rawParamSum = sum(rawParams)
	newParams = zeros(numParams)
	newParamSum = 0

	currentProb = (alphaLikelihood(exp(rawParams),modelState) + rawParamSum)

	for i in xrange(numIts):
		#print i
		uPrime = log(random.rand()) + currentProb

		#find the intervals
		r = random.rand(numParams)
		leftside = rawParams-r*stepVec
		rightside = leftside+stepVec
	
		under = False
		while not(under):
			newParams = random.uniform(leftside,rightside,numParams)
			newParamSum = sum(newParams)
			newProb = (alphaLikelihood(exp(newParams),modelState)
				+ newParamSum)
			if newProb > uPrime:
				under = True
			else:
				leftside = where(newParams<rawParams,newParams,leftside)
				rightside = where(newParams<rawParams,rightside,newParams)
	
		rawParams = newParams
		rawParamSum = newParamSum
		currentProb = newProb
		#print exp(rawParams)
	return exp(rawParams)


def sampleBeta(beta,step,modelState,numIts):
	rawParam = log(beta)
	newParam = 0

	currentProb = (betaLikelihood(exp(rawParam),modelState) + rawParam)

	for i in xrange(numIts):
		#print i
		uPrime = log(random.rand()) + currentProb

		#find the intervals
		r = random.rand()
		leftside = rawParam-r*step
		rightside = leftside+step
	
		under = False
		while not(under):
			newParam = random.uniform(leftside,rightside,1)
			newProb = (betaLikelihood(exp(newParam),modelState)
				+ newParam)
			if newProb > uPrime:
				under = True
			else:
				if newParam<rawParam:
					leftside = newParam
				else:
					rightside = newParam	
		rawParam = newParam
		currentProb = newProb
		#print exp(rawParam)
	return exp(rawParam)


def sampleZeta(zeta,step,modelState,numIts):
	rawParam = log(zeta)
	newParam = 0

	currentProb = (zetaLikelihood(exp(rawParam),modelState) + rawParam)

	for i in xrange(numIts):
		#print i
		uPrime = log(random.rand()) + currentProb

		#find the intervals
		r = random.rand()
		leftside = rawParam-r*step
		rightside = leftside+step
	
		under = False
		while not(under):
			newParam = random.uniform(leftside,rightside,1)
			newProb = (zetaLikelihood(exp(newParam),modelState)
				+ newParam)
			if newProb > uPrime:
				under = True
			else:
				if newParam<rawParam:
					leftside = newParam
				else:
					rightside = newParam	
		rawParam = newParam
		currentProb = newProb
		#print exp(rawParam)
	return exp(rawParam)



def optimize(alpha0,alpha1,alpha,beta,zeta,step,modelState,numIts):
	alphaArray = array([alpha0,alpha1,alpha])
	stepVec = array([step,step,step])
	alphas = sampleAlpha(alphaArray,stepVec,modelState,numIts)
	beta = sampleBeta(beta,step,modelState,numIts)
	zeta = sampleZeta(zeta,step,modelState,numIts)
	return [alphas[0],alphas[1],alphas[2],beta,zeta]
