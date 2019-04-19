import numpy as np 
import numpy.random as npr 


def J(z,x,c):
	"""
	z: labels vector of size N
	x: data set of size N by d
	c: centroids matrix k by d
	"""
	k = c.shape[0]
	N = z.size
	ret = 0
	for i in range(N):
		for j in range(k):
			ret += (z[i]==j)*np.linalg.norm(x[i]-c[j])**2
	return ret


def kmeans(_Y, k):
	"""
	input: _Y (feature matrix of size d by N)
		   k (number of groups)
	output: X (list of )
	"""
	Y = _Y.T
	N, d = Y.shape

	J_min = np.inf
	NUM_TRIALS = 100
	for trial_id in range(NUM_TRIALS):
		#initialize centroids
		MU = npr.uniform(low=np.amin(Y,axis=0), high=np.amax(Y,axis=0), size=(k,d))
		labels = np.zeros(N, dtype=int)
		ones = np.ones((N,d))
		while True:
			for i in range(N):
				labels[i] = np.argmin(np.linalg.norm(MU - Y[i],axis=1))

			old_MU = np.copy(MU)
			for j in range(k):
				MU[j] = np.sum(Y*(labels==j)[:,None], axis=0)/max(np.sum((labels==j)),1)
		
			if np.sum(np.linalg.norm(MU-old_MU,axis=1)) < 0.01:
				break
		J_new = J(labels,Y,MU)
		if J_new < J_min:
			J_min = J_new
			MU_best = np.copy(MU)

		output = [[] for _ in range(k)]
		labels[i] = np.argmin(np.linalg.norm(MU_best - Y[i],axis=1))
		for i in range(N):
			output[labels[i]].append(i)

	return MU_best
