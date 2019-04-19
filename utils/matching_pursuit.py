import numpy as np
import scipy.signal
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.dataset_utils import generateImagePatches

def greedyMatchingPursuit(image, features, k_matches, verbose=False):
    patch_size = features[0].shape[0]
    stride = image.shape[1] // patch_size

    # Loops through patches in an image
    patches, num_patches = generateImagePatches(patch_size, image)

    mat_patches = patches.reshape((patches.shape[0], np.prod(patches.shape[1:3]))) - 0.5    
    mat_features = features.reshape((features.shape[0], np.prod(features.shape[1:3])))
	
	# Return variables
    S_code = list()
    recon_image = np.zeros_like(image)
    residual = np.copy(mat_patches)
    
    # Progress bar
    pbar = range(k_matches)
    if verbose:
        pbar = tqdm(range(k_matches))
        
    # Loop over for K best matches
    for _ in pbar:

        # Match list is number of patches by number of features
        match_list = np.matmul(residual, mat_features.T)

        # Get sorted indicies of match_list
        best_match_ind = np.argmax(np.abs(match_list.reshape(-1)))
        patch_id, feature_id = np.unravel_index(best_match_ind, (num_patches, features.shape[0]))
		
		# Compute weight of this feature
        weight = np.matmul(mat_patches, mat_features.T)[patch_id, feature_id]

        # Add this feature to our sparse code
        S_code.append((patch_id, feature_id, match_list[patch_id, feature_id]))
		
        # Get location of image patch
        y = patch_id // stride
        x = patch_id % stride
        recon_image[y * patch_size: (y + 1) * patch_size,
                    x * patch_size: (x + 1) * patch_size] += weight * features[feature_id]
		
        # Compute Residual 
        residual[patch_id] -= weight * mat_features[feature_id]

    return S_code

def convolutionalMatchingPursuit(image, features, k_matches, verbose=False):
    recon_image = np.zeros_like(image)
    e = np.copy(image)-0.5
    h,w = image.shape
    f_size = features.shape[1]
    num_patches = (h-f_size)*(w-f_size)
    num_features = features.shape[0]
    patches = np.zeros((num_patches, f_size**2))
    kernels = features.reshape((num_features, f_size**2)).T

    for i in range(num_patches):
        x,y = i // (e.shape[0]-f_size), i % (e.shape[0]-f_size)
        patches[i] = e[x:x+f_size,y:y+f_size].reshape(-1)

    S_code = list()
    pbar = range(k_matches)
    if verbose:
        pbar = tqdm(range(k_matches))
    S_code = []
    for _ in pbar:
        activations = np.matmul(patches, kernels)
        tmp = np.copy(activations)
        for p,f in S_code:
            tmp[p,f] = 0
        best_match_ind = np.argmax(np.abs(activations.reshape(-1)))
        patch_id, feature_id = np.unravel_index(best_match_ind, activations.shape)
        x,y = patch_id // (image.shape[0]-f_size), patch_id % (image.shape[0]-f_size)

        recon_image[x:x+f_size,y:y+f_size] += activations[patch_id, feature_id]*features[feature_id]
        e[x:x+f_size,y:y+f_size] -= activations[patch_id, feature_id]*features[feature_id]
        #now we need to remove the contribution from all the patches that overlap
        
    return S_code, recon_image + 0.5

def orthogonalMatchingPursuit(image, features, k_matches, verbose=False):
    """
	Computes the sparse code, reconstructed image, and residual
	via orthogonal matching pursuit
	"""
    patch_size = features[0].shape[0]
    stride = image.shape[1] // patch_size

    # Loops through patches in an image
    patches, num_patches = generateImagePatches(patch_size, image)

    mat_patches = patches.reshape((patches.shape[0], np.prod(patches.shape[1:3]))) - 0.5    
    mat_features = features.reshape((features.shape[0], np.prod(features.shape[1:3])))
	
	# Return variables
    S_code = list()
    recon_image = np.zeros_like(image)
    residual = np.copy(mat_patches)
    
    # Progress bar
    pbar = range(k_matches)
    if verbose:
        pbar = tqdm(range(k_matches))
        
    # Loop over for K best matches
    feature_ids = set()
    for _ in pbar:

        # Match list is number of patches by number of features
        match_list = np.matmul(residual, mat_features.T)

        # Get sorted indicies of match_list
        best_match_ind = np.argmax(np.abs(match_list.reshape(-1)))
        patch_id, feature_id = np.unravel_index(best_match_ind, (num_patches, features.shape[0]))

        # Add this feature to our sparse code
        S_code.append((patch_id, feature_id, match_list[patch_id, feature_id]))
        
		# Compute weight of this feature
        weight = np.matmul(mat_patches, mat_features.T)[patch_id, feature_id]
        
        # Get location of image patch
        y = patch_id // stride
        x = patch_id % stride
        recon_image[y * patch_size: (y + 1) * patch_size,
                    x * patch_size: (x + 1) * patch_size] += weight * features[feature_id]
		
        # Compute Residual with all features used up til now
        feature_ids.add(feature_id)
        inds = np.array(list(feature_ids))
        projection = np.matmul(np.linalg.pinv(mat_features[inds]), mat_features[inds])
        residual[patch_id] = np.matmul((np.eye(projection.shape[0]) - projection) , mat_patches[patch_id])

    return S_code, recon_image + 0.5


def videoMatchingPursuit(video, features, k_matches, algo):
    
    S_codes = list()
    recon_video = np.zeros((len(video), video[0].shape[0], video[0].shape[1]))
    
    # Loop over all images in video
    for t in tqdm(range(len(video))):
        S_code, recon_image = algo(video[t], features, k_matches)
        
        S_codes.append(S_code) 
        recon_video[t] = recon_image
    
    return S_codes, recon_video

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

