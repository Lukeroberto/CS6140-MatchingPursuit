import numpy as np
from PIL import Image
import glob

def loadVideo(location, num_frames):

    rel_loc = "/image_00/data/"
    
    # Loop over all files in directory
    images = list()
    for filename in sorted(glob.glob(location + rel_loc + "*.png")):
        img = Image.open(filename)
        images.append(img)


    return images[:num_frames]
