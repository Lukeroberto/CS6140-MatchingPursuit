import numpy as np

def reconstructionLoss(image, recon_image):
    return np.linalg.norm(image - recon_image)

def energyLoss(image, recon_image, code_length, reg=0.1):
    return reconstructionLoss(image, recon_image) + reg * code_length


def videoReconstructionLoss(video, recon_video):
    loss = np.zeros(len(video))
    for i, _ in enumerate(video):
        loss[i] = reconstructionLoss(video[i], recon_video[i])
        
    return loss