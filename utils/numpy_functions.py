import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def uniform_vector(length):
	return np.ones((length))/length

def make_initial_prior(initial_image_prior,initial_rationality_prior,initial_speaker_prior):
	
	return np.log(np.multiply.outer(initial_image_prior,np.multiply.outer(initial_rationality_prior,initial_speaker_prior)))
