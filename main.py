import matplotlib
matplotlib.use('Agg')
import re
import requests
import time
import pickle
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from collections import defaultdict

from utils.config import *
from utils.numpy_functions import uniform_vector, make_initial_prior
from recursion_schemes.recursion_schemes import ana_greedy,ana_beam
from bayesian_agents.joint_rsa import RSA


urls = [
	"https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Arriva_T6_nearside.JPG/1200px-Arriva_T6_nearside.JPG",
	"https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/First_Student_IC_school_bus_202076.jpg/220px-First_Student_IC_school_bus_202076.jpg"
	]

# code is written to be able to jointly infer speaker's rationality and neural model
rat = [100.0]
model = ["vg"]
number_of_images = len(urls)
initial_image_prior=uniform_vector(number_of_images)
initial_rationality_prior=uniform_vector(1)
initial_speaker_prior=uniform_vector(1)
initial_world_prior = make_initial_prior(initial_image_prior,initial_rationality_prior,initial_speaker_prior)

# make a character level speaker, using torch model (instead of tensorflow model)
speaker_model = RSA(seg_type="char",tf=False)
speaker_model.initialize_speakers(model)
# set the possible images and rationalities
speaker_model.speaker_prior.set_features(images=urls,tf=False,rationalities=rat)
speaker_model.initial_speakers[0].set_features(images=urls,tf=False,rationalities=rat)
# generate a sentence by unfolding stepwise, from the speaker
literal_caption = ana_greedy(
	speaker_model,
	target=0,
	depth=0,
	speaker_rationality=0,
	speaker=0,
	start_from=list(""),
	initial_world_prior=initial_world_prior)

pragmatic_caption = ana_greedy(
	speaker_model,
	target=0,
	depth=1,
	speaker_rationality=0,
	speaker=0,
	start_from=list(""),
	initial_world_prior=initial_world_prior)

print("Literal caption:\n",literal_caption)
print("Pragmatic caption:\n",pragmatic_caption)
