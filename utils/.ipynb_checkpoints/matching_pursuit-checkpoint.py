import numpy as np
import scipy.signal
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.dataset_utils import generateImagePatches

def greedyMatchingPursuit(image, features, k_matches, verbose=False):
    patch_size = features[0].shape[0]

    # Loops through patches in an image
    patches, num_patches = generateImagePatches(patch_size, image)

    mat_patches = patches.reshape((patches.shape[0], np.prod(patches.shape[1:3]))) -0.5    
    mat_patches_true = np.copy(mat_patches)
    mat_features = features.reshape((features.shape[0], np.prod(features.shape[1:3])))

    # Normalize the patch matrix, clip features
#     row_sums = mat_patches.sum(axis=1)
#     mat_patches = mat_patches / row_sums[:, np.newaxis]
    # mat_features = np.clip(mat_features, 0, 1)

    # Sparse Code output
    S_code = list()
    
    # Progress bar
    pbar = range(k_matches)
    if verbose:
        pbar = tqdm(range(k_matches))
        
    # Loop over for K best matches
    for i in pbar:

        # Match list is number of patchs by number of features
        match_list = np.matmul(mat_patches, mat_features.T)

        # Get sorted indicies of match_list
        best_match_ind = np.argmax(np.abs(match_list.reshape(-1)))
        patch_id, feature_id = np.unravel_index(best_match_ind, (num_patches, features.shape[0]))

        # Remove contribution of feature from image
        mat_patches[patch_id] -= np.matmul(mat_patches_true, mat_features.T)[patch_id, feature_id] \
                                         * mat_features[feature_id]

        # Add this feature to our sparse code
        S_code.append((patch_id, feature_id, match_list[patch_id, feature_id]))

    return S_code

def convolutionalMatchingPursuit(image, features, k_matches, verbose=False):
	image_original = np.copy(image)
	
	# Sparse Code output
	S_code = list()

	# Progress bar
	pbar = range(k_matches)
	if verbose:
		pbar = tqdm(range(k_matches))

	# Loop over for K best matches
	for i in pbar:

		# Keep track of patch convolutions
		match_list = np.zeros(features.shape[0]) 
		convolved_features = np.zeros((features.shape[0], image.shape[0], image.shape[1]))

		# Convolve features with image
		for i, feature in enumerate(features): 
			convolved_features[i] = scipy.signal.convolve2d(image_original, feature, mode="same")
			match_list[i] = np.linalg.norm(convolved_features[i])

		# Get sorted indicies of match_list
		best_feature_ind = np.argmax(match_list)

		# Remove contribution of feature from image
		image -= convolved_features[best_feature_ind]

		# Add this feature to our sparse code
		S_code.append((best_feature_ind, match_list[best_feature_ind]))

	return S_code

def temporalMatchingPursuit(image, features, k_matches):
    pass

def videoMatchingPursuit(video, features, k_matches, algo):
    
    S_codes = list()
    # Loop over all images in video
    for image in tqdm(video):
        S_codes.append(algo(image, features, k_matches)) 
    
    return S_codes

def generateReconImage(S_code, original_image, features):
    patch_size = features[0].shape[0]
    stride = original_image.shape[1] // patch_size

    # Reconstructed image
    recon_image = np.zeros(original_image.shape)
    for patch_id, feature_id, weight in S_code:

        # Get location of image patch
        y = patch_id // stride
        x = patch_id % stride

        # Set equal to weighted sum of features
        recon_image[y * patch_size: (y + 1) * patch_size,
                    x * patch_size: (x + 1) * patch_size] += weight * features[feature_id]

    return recon_image + 0.5

def generateReconVideo(S_codes, original_video, features):
    
    # Reconstructed video
    recon_video = np.zeros((len(original_video), original_video[0].shape[0], original_video[0].shape[1]))
    
    # Loop over and apply recon image
    for i, (S_code, image) in enumerate(zip(S_codes, original_video)):
        recon_video[i] = generateReconImage(S_code, image, features)
    
    return recon_video

