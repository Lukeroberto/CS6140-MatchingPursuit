import numpy as np
import scipy.signal
from tqdm import tqdm_notebook as tqdm

from utils.dataset_utils import generateImagePatches

def orthogonalMatchingPursuit(image, features, k_matches):
    patch_size = features[0].shape[0]

    # Loops through patches in an image
    patches, num_patches = generateImagePatches(patch_size, image)

    mat_patches = patches.reshape((patches.shape[0], np.prod(patches.shape[1:3])))
    mat_features = features.reshape((features.shape[0], np.prod(features.shape[1:3])))

    # Normalize the patch matrix, clip features
#     row_sums = mat_patches.sum(axis=1)
#     mat_patches = mat_patches / row_sums[:, np.newaxis]
    mat_features = np.clip(mat_features, 0, 1)

    # Sparse Code output
    S_code = list()

    # Loop over for K best matches
    for i in tqdm(range(k_matches)):

        # Match list is number of patchs by number of features
        match_list = np.matmul(mat_patches, mat_features.T)

        # Get sorted indicies of match_list
        best_match_ind = np.argmax(match_list.reshape(-1))
        patch_id, feature_id = np.unravel_index(best_match_ind, (num_patches, features.shape[0]))

        # Remove contribution of feature from image
        mat_patches[patch_id] -= match_list[patch_id, feature_id] * mat_features[feature_id]

        # Add this feature to our sparse code
        S_code.append((patch_id, feature_id, match_list[patch_id, feature_id]))

    return S_code

def convolutionalMatchingPursuit(image, features, k_matches):
    image_float = image.astype(np.float32) / 256
    
    # Sparse Code output
    S_code = list()

    # Loop over for K best matches
    for i in tqdm(range(k_matches)):

        # Keep track of patch convolutions
        match_list = np.zeros(features.shape[0]) 
        convolved_features = np.zeros((features.shape[0], image_float.shape[0], image_float.shape[1]))
        
        # Convolve features with image
        for i, feature in enumerate(features): 
            convolved_features[i] = scipy.signal.convolve2d(image_float, feature, mode="same")
            match_list[i] = np.linalg.norm(convolved_features[i])
            
        # Get sorted indicies of match_list
        best_feature_ind = np.argmax(match_list)

        # Remove contribution of feature from image
        image_float -= convolved_features[best_feature_ind]

        # Add this feature to our sparse code
        S_code.append((best_feature_ind, match_list[best_feature_ind]))

    return S_code

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

    return recon_image

