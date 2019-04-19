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
    for _ in pbar:

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
        

        S_code.append((patch_id, feature_id))
    plt.figure()
    plt.imshow(image, cmap='Greys_r')
    plt.figure()
    plt.imshow(recon_image, cmap='Greys_r')
    # print(S_code)

    # e = np.copy(image)-0.5
    # recon_image = np.zeros_like(image)

    # # Sparse Code output
    # S_code = list()

    # # Progress bar
    # pbar = range(k_matches)
    # if verbose:
    # 	pbar = tqdm(range(k_matches))

    # num_features = features.shape[0]
    # patch_size = features.shape[1]
    # kernels = np.zeros((patch_size, patch_size, num_features))
    # for i in range(num_features):
    #     kernels[:,:,i] = features[i]

    # # Loop over for K best matches
    # for i in pbar:
    #     activations = scipy.signal.convolve(e[:,:,None], kernels, mode='same')
    #     best_match_ind = np.argmax(np.abs(activations.reshape(-1)))
    #     x,y,f = np.unravel_index(best_match_ind, activations.shape)
    #     image_patch = image[x:x+patch_size, y:y+patch_size]
    #     s1, s2 = image_patch.shape
    #     w = activations[x,y,f]

    #     recon_image[x:x+patch_size, y:y+patch_size] += features[f,:s1,:s2]
    #     e[x:x+patch_size, y:y+patch_size] = 0
        
    # plt.figure()
    # plt.imshow(image,cmap="Greys_r")
    # plt.figure()
    # plt.imshow(recon_image,cmap="Greys_r")
    # plt.show()
        

def temporalMatchingPursuit(image, features, k_matches, prior_code, verbose=False):
    pbar = range(k_matches)
    if verbose:
        pbar = tqdm(range(k_matches))
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

