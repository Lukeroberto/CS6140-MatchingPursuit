import numpy as np

def reconstructionLoss(image, recon_image):
	return np.linalg.norm(image - recon_image) ** 2

def energyLoss(image, recon_image, code_length, reg=0.1):
	return reconstructionLoss(image, recon_image) + reg * code_length

def reconstructionRatio(image, recon_image):
	original_norm = np.linalg.norm(image) ** 2
	return (original_norm - reconstructionLoss(image, recon_image)) / original_norm 

def videoLoss(video, recon_video, lossFunc):
	loss = np.zeros(len(video))
	for i, _ in enumerate(video):
		loss[i] = lossFunc(video[i], recon_video[i])

	return loss