import numpy as np
from PIL import Image

import glob

def loadVideo(location, num_frames):

    rel_loc = "/image_00/data/"
    
    # Loop over all files in directory
    images = list()
    for filename in sorted(glob.glob(location + rel_loc + "*.png")):
        img = np.array(Image.open(filename))
        images.append(img)


    return images[:num_frames]

def cropImages(images, width=50, height=50, offset_x=0.5, offset_y=0.5, downsampling=2):
    full_height, full_width = images[0].shape
    offset_x = int(full_width*offset_x)
    offset_y = int(full_height*offset_y)
    ret = []
    for image in images:
        i = image[offset_y:offset_y+downsampling*height, offset_x:offset_x+downsampling*width]
        ret.append(i[::downsampling,::downsampling])
    return ret

def generateImagePatches(patch_size, image):
    image = image.astype(np.float32) / 256
    height, width = np.shape(image)
    image = image[:(height//patch_size)*patch_size,:(width//patch_size)*patch_size]
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