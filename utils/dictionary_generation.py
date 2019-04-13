import numpy as np 
from sklearn.decomposition import PCA
from utils.dataset_utils import samplePatches, generateVideoPatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plotDictionary(features, title=''):
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
	pass


def generateSIFTDictionary(images, patch_size, num_samples, num_features):
	pass


def computePCA(num_features, samples):

    pca = PCA(n_components=num_features)

    # Squeeze sample patches to be array
    print("Num samples", len(samples))
    pca.fit(np.reshape(samples, (np.shape(samples)[0], np.shape(samples)[1] ** 2)))

    features = pca.components_

    filter_size = np.shape(samples)[1]
    filter_features = np.zeros((len(features), filter_size, filter_size))

    for i, feature in enumerate(features):
        filter_features[i] = np.reshape(feature, (filter_size, filter_size))

    return filter_features


def generatePCADictionary(images, patch_size, num_samples, num_features):

    video_patches, _ = generateVideoPatches(patch_size, images)
    samples = samplePatches(num_samples, video_patches)

    return computePCA(num_features, samples)