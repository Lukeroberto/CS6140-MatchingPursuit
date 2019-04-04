import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

import glob

def loadVideo(location, num_frames):

    rel_loc = "/image_00/data/"
    
    # Loop over all files in directory
    images = list()
    for filename in sorted(glob.glob(location + rel_loc + "*.png")):
        img = np.array(Image.open(filename))
        images.append(img)


    return images[:num_frames]

def generateImagePatches(patch_size, image):

    height, width = np.shape(image)
    assert height % patch_size == 0, height % patch_size
    assert width % patch_size == 0, width % patch_size

    x_stride = width // patch_size
    y_stride = height // patch_size
    num_patches = x_stride * y_stride

    patches = np.zeros((num_patches, patch_size, patch_size))

    # Loop over patches
    for patch_x in range(x_stride):
        for patch_y in range(y_stride):
            patches[patch_x + x_stride * patch_y] = image[patch_size * patch_y:patch_size * (patch_y + 1),
                                 patch_size * patch_x:patch_size * (patch_x + 1)]

    return patches, num_patches

def generateVideoPatches(patch_size, images):
    num_images = len(images)
    _, num_image_patches = generateImagePatches(patch_size, images[0])

    num_video_patches = num_image_patches * num_images
    video_patches = np.zeros((num_video_patches, patch_size, patch_size))

    for i, image in enumerate(images):
        video_patches[i * num_image_patches: (i + 1) * num_image_patches], _ = generateImagePatches(patch_size, image)

    return video_patches, num_video_patches

def samplePatches(num_samples, patches):
    random_patches = np.random.choice(patches.shape[0], num_samples, replace=False)
    return patches[random_patches, :, :]

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


def generateDictionary(images, patch_size, num_samples, num_features):

    video_patches, _ = generateVideoPatches(patch_size, images)
    samples = samplePatches(num_samples, video_patches)

    return computePCA(num_features, samples)
