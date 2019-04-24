import numpy as np 
from sklearn.decomposition import PCA,  DictionaryLearning
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.dataset_utils import samplePatches, generateVideoPatches
from utils.Kmeans import kmeans

def plotDictionary(features, title):
	num_features = features.shape[0]
	cols = 10
	rows = int(np.ceil(num_features/cols))
	s = 1
	fig = plt.figure(figsize=(cols*s, s*rows))
	plt.title(title)
	gs = gridspec.GridSpec(rows, cols)
	gs.update(wspace=0.0, hspace=0.02)
	for i in range(num_features):
	    ax = plt.subplot(gs[i//cols, i % cols])
	    plt.imshow(features[i], cmap="Greys_r")
	    plt.axis("off")
	plt.show()

def generatePSDDictionary(images, patch_size, num_samples, num_features):
	# https://cs.nyu.edu/~yann/research/sparse/index.html
	video_patches, _ = generateVideoPatches(patch_size, images)
	samples = samplePatches(num_samples, video_patches)
	samples = samples.reshape(samples.shape[0], samples.shape[1]**2)
	# m > n usually
	n = samples.shape[1]
	m = num_features

	Z = npr.random(size=(m, num_samples))
	B = npr.random(size=(n,m))
	B = (B.T/np.linalg.norm(B, axis=1)).T
	W = npr.random(size=(m,n))
	D = npr.random(size=m)
	G = np.diag(npr.random(size=m))

	Y = samples.T #n  by num_samples

	lmbda = 1.0
	alpha = 1.0
	lr = 1e-6


	for _ in range(200):
		# keep G,D,W & B constant, minimize wrt Z
		F = np.matmul(G, np.tanh(np.matmul(W,Y).T+D).T)

		for i in range(1000):
			dJ = 2*np.matmul(B.T, (np.matmul(B,Z)-Y) ) + lmbda * np.sum(np.sign(Z), axis=0) + 2*alpha*(Z-F)
			Z = Z - lr * dJ
	
		i = npr.randint(num_samples)
		z = Z[:,i]
		y = Y[:,i]
		f = np.matmul(G, np.tanh(np.matmul(W,y).T+D).T)
		# one step of stochastic gradient descent on G,D,W & B

		G -= -0.001*lr*2*alpha* np.matmul( G, z - f)
		D -= -lr*2*alpha*np.matmul( (np.matmul(G, (1- np.power(np.tanh(np.matmul(W,y).T+D).T, 2)) )).T , z-f)
		W -= -lr*2*alpha*y.T.dot(np.matmul(np.matmul(G, (1- np.power(np.tanh(np.matmul(W,y).T+D).T, 2)) ).T, z-f))
		B -= 0.001*lr*np.outer(np.matmul(B,z) - y, z)

		# print(np.linalg.norm(2*alpha* np.matmul( G.T, z - f)), \
		# 	np.linalg.norm(2*alpha*np.matmul( (np.matmul(G, (1- np.power(np.tanh(np.matmul(W,y).T+D).T, 2)) )).T , z-f)), \
		# 	np.linalg.norm(2*alpha*y.T.dot(np.matmul(np.matmul(G, (1- np.power(np.tanh(np.matmul(W,y).T+D).T, 2)) ).T, z-f))), \
		# 	np.linalg.norm(np.outer(np.matmul(B,z) - y, z)))

		B = (B.T/np.linalg.norm(B, axis=1)).T

	return B.T.reshape((num_features, patch_size, patch_size))
	
def generateKMeansDictionary(images, patch_size, num_samples, num_features):
	video_patches, _ = generateVideoPatches(patch_size, images)
	samples = samplePatches(num_samples, video_patches)
	samples = samples.reshape(samples.shape[0], samples.shape[1]**2)

	kmeans = KMeans(n_clusters=num_features).fit(samples)
	centers = kmeans.cluster_centers_

	centers = (centers.T/np.linalg.norm(centers,axis=1).T).T

	return centers.reshape((num_features, patch_size, patch_size))

def generateOptSparseDictionary(images, patch_size, num_samples, num_features):
	video_patches, _ = generateVideoPatches(patch_size, images)
	samples = samplePatches(num_samples, video_patches)
	alg = DictionaryLearning(n_components=num_features)


	# Squeeze sample patches to be array
	alg.fit(samples.reshape(np.shape(samples)[0], np.shape(samples)[1] ** 2))
	features = alg.components_

	filter_size = np.shape(samples)[1]

	features = (features.T/np.linalg.norm(features,axis=1).T).T
	# features = (features.T - np.mean(features,axis=1).T).T

	features = features.reshape(features.shape[0], filter_size, filter_size)

	return features
    

def computePCA(num_features, samples):

	pca = PCA(n_components=num_features)

	# Squeeze sample patches to be array
	# print("Num samples", len(samples))
	pca.fit(np.reshape(samples, (np.shape(samples)[0], np.shape(samples)[1] ** 2)))

	features = pca.components_

	filter_size = np.shape(samples)[1]

	features = (features.T/np.linalg.norm(features,axis=1).T).T
	# features = (features.T - np.mean(features,axis=1).T).T

	features = features.reshape(features.shape[0], filter_size, filter_size)

	return features


def generatePCADictionary(images, patch_size, num_samples, num_features):

    video_patches, _ = generateVideoPatches(patch_size, images)
    samples = samplePatches(num_samples, video_patches)

    return computePCA(num_features, samples)

def generateDictionaryPlots(images, patch_size, num_samples, num_features, s=1, cols=2):
	# s is the size of each square
	alg_names = ["PCA", "Kmeans", "OptSparse"]
	algs = [generatePCADictionary, generateKMeansDictionary, generateOptSparseDictionary]
	if not isinstance(num_features, list):
		num_features = [num_features]
	for i in range(len(algs)):
		alg = algs[i]
		name = alg_names[i]
		for f in num_features:
			features = alg(images, patch_size, num_samples, f)
			f = features.shape[0]
			rows = int(np.ceil(f/cols))
			fig = plt.figure(figsize=(cols*s, s*rows))
			gs = gridspec.GridSpec(rows, cols)
			gs.update(wspace=0.0, hspace=0.05)
			for j in range(f):
				ax = plt.subplot(gs[j//cols, j % cols])
				plt.imshow(features[j], cmap="Greys_r")
				plt.axis("off")
			plt.savefig("%s_k=%d_f=%d.png" % (name, patch_size, f)) 

