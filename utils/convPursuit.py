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
        best_match_ind = np.argmax(np.abs(activations.reshape(-1)))
        patch_id, feature_id = np.unravel_index(best_match_ind, activations.shape)
        x,y = patch_id // (image.shape[0]-f_size), patch_id % (image.shape[0]-f_size)

        w = activations[patch_id, feature_id]
        recon_image[x:x+f_size,y:y+f_size] += w*features[feature_id]
        e[x:x+f_size,y:y+f_size] -= w*features[feature_id]
        #now we need to remove the contribution from all the patches that overlap
        for i in range(f_size**2):
            dx, dy = i // f_size, i % f_size
            a = np.copy(features[feature_id])
            a[:dx, :] = 0.0
            a[:, :dy] = 0.0
            p_id = patch_id + dx*(e.shape[0]-f_size)+dy
            if p_id >= num_patches: continue
            patches[p_id] -= w*a.reshape(-1)


        S_code.append((patch_id, feature_id))
    plt.figure()
    plt.imshow(image, cmap='Greys_r')
    plt.figure()
    plt.imshow(recon_image, cmap='Greys_r')