import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import animation

import glob

def loadVideo(location, num_frames):

    rel_loc = "/image_00/data/"
    
    # Loop over all files in directory
    images = list()
    for filename in sorted(glob.glob(location + rel_loc + "*.png")):
        img = np.array(Image.open(filename))
        images.append(img.astype(np.float32) / 256)


    return images[:num_frames]

def cropImages(images, width=200, height=200, offset_x=0.5, offset_y=0.4, downsampling=1):
    full_height, full_width = images[0].shape
    offset_x = int(full_width*offset_x)
    offset_y = int(full_height*offset_y)
    ret = []
    for image in images:
        i = image[offset_y:offset_y+downsampling*height, offset_x:offset_x+downsampling*width]
        ret.append(i[::downsampling,::downsampling])
    return ret

def generateImagePatches(patch_size, image):
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

def compareImages(image1, image2):
    #Creates two subplots and unpacks the output array immediately
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
    ax1.imshow(image1, cmap="Greys_r")
    ax1.axis("off")

    ax2.imshow(image2, cmap="Greys_r")
    ax2.axis("off")

    plt.show()
    
def animateVideo(video):
    fig = plt.figure()
    ims = []
    for img in video:
        im = plt.imshow(img, cmap="Greys_r", animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    return ani

def compareVideos(videos):
    f, (ax1, ax2) = plt.subplots(1,2, sharey=True,figsize=(12,6))
    ims = []
    for i in range(len(video1)):
        im1 = ax1.imshow(video1[i], cmap="Greys_r", animated=True)
        im2 = ax2.imshow(video2[i], cmap="Greys_r", animated=True)
        
        ax1.axis("off")
        ax2.axis("off")
        ims.append([im1, im2])

    ani = animation.ArtistAnimation(f, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    return ani
    