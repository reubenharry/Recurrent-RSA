import time
import itertools
import scipy
import scipy.stats
import numpy as np
import math
from PIL import Image as PIL_Image
from keras.preprocessing import image
from keras.models import load_model
from utils.image_and_text_utils import index_to_char,char_to_index
from utils.config import *
from bayesian_agents.rsaWorld import RSA_World
from utils.numpy_functions import softmax
from train.Model import Model

class RSA:

	def __init__(
		self,
		seg_type,
		tf,
		):


			self.tf=tf
			self.seg_type=seg_type
			self.char=self.seg_type="char"

			#caches for memoization
			self._speaker_cache = {}
			self._listener_cache = {}
			self._speaker_prior_cache = {}

			if self.char:
				self.idx2seg=index_to_char
				self.seg2idx=char_to_index


	

	def initialize_speakers(self,paths):


		self.initial_speakers = [Model(path=path,
			dictionaries=(self.seg2idx,self.idx2seg)) for path in paths] 
		self.speaker_prior = Model(path="lang_mod",
			dictionaries=(self.seg2idx,self.idx2seg))
		# self.initial_speaker.set_features()

		# self.speaker_prior
		
		# self.images=images
		# print("NUMBER OF IMAGES:",self.number_of_images)





		

	def flush_cache(self):

		self._speaker_cache = {}
		self._listener_cache = {}
		self._speaker_prior_cache = {}	

	# memoization is crucial for speed of the RSA, which is recursive: memoization via decorators for speaker and listener
	# def memoize_speaker_prior(f):
	# 	def helper(self,state,world):
	
	# 		# world_prior_list = np.ndarray.tolist(np.ndarray.flatten(state.world_priors))
	# 		hashable_args = state,world

	# 		if hashable_args not in self._speaker_cache:
	# 			self._speaker_prior_cache[hashable_args] = f(self,state,world)
	# 		# else: print("cached")
	# 		return self._speaker_prior_cache[hashable_args]
	# 	return helper

	def memoize_speaker(f):
		def helper(self,state,world,depth):
	
			# world_prior_list = np.ndarray.tolist(np.ndarray.flatten(state.world_priors))
			hashable_args = state,world,depth

			if hashable_args not in self._speaker_cache:
				self._speaker_cache[hashable_args] = f(self,state,world,depth)
			# else: print("cached")
			return self._speaker_cache[hashable_args]
		return helper

	def memoize_listener(f):
		def helper(self,state,utterance,depth):
	
			# world_prior_list = np.ndarray.tolist(np.ndarray.flatten(state.world_priors))
			hashable_args = state,utterance,depth

			if hashable_args not in self._listener_cache:
				self._listener_cache[hashable_args] = f(self,state,utterance,depth)
			# else: print("cached")

			return self._listener_cache[hashable_args]
		return helper




	# @memoize_speaker_prior
	# def speaker_prior(self,state,world):
	# 	# print("SPEAKER PRIOR",(world.target,world.speaker,world.rationality))

	# 	pass


	@memoize_speaker
	def speaker(self,state,world,depth):
		# print("rationality",world.rationality)
		# print("world prior shape",state.world_priors[0].shape)
		# print("SPEAKER\n\n",depth)


		if depth==0:
			# print("S0")
			# print("TIMESTEP:",state.timestep,"INITIAL SPEAKER CALL")



			return self.initial_speakers[world.speaker].forward(state=state,world=world)
			
		else: 

			prior = self.speaker(state,world,depth=0)
			# self.initial_speakers[world.speaker].forward(state=state,world=world)
			# prior = self.speaker_prior.forward(state=state,world=world)

		# self.speaker(state=state,world=world,depth=0)
		if depth==1:

			scores = []
			for k in range(prior.shape[0]):
				# print(world.target,world.rationality,"FIRST")
				out = self.listener(state=state,utterance=k,depth=depth-1)

				
				scores.append(out[world.target,world.rationality,world.speaker])

			scores = np.asarray(scores)
			# print("SCORES",scores)
			# rationality in traditional RSA sense
			scores = scores*(self.initial_speakers[world.speaker].rationality_support[world.rationality])
			# update prior to posterior
			# print(scores.shape,prior.shape)
			posterior = (scores + prior) - scipy.misc.logsumexp(scores + prior)
			# print("POSTERIOR",posterior)

			return posterior

		elif depth==2:

			scores = []
			for k in range(prior.shape[0]):

				# print(world.rationality,"rat")
				out = self.listener(state=state,utterance=k,depth=depth-1)
				scores.append(out[world.target,world.rationality,world.speaker])

			scores = np.asarray(scores)
			# rationality not present at s2
			# update prior to posterior
			posterior = (scores + prior) - scipy.misc.logsumexp(scores + prior)

			return posterior

	@memoize_listener
	def listener(self,state,utterance,depth):

		# base case listener is either neurally trained, or inferred from neural s0, given the state's current prior on images

		# world = RSA_World(target=0,speaker=0,rationality=0)
		# image_prior = self.listener(state=state,utterance=utterance,depth=depth-1)
		# rationality_prior = np.asarray([0.3,0.7])

		world_prior = state.world_priors[state.timestep-1]
		# print("world prior",np.exp(world_prior))
		# if state.timestep < 4:
		# 	print("world priors",np.exp(state.world_priors[:4]))
		# 	print("timestep",state.timestep)

		# if depth==0:
		# else: world_prior = self.listener(state=state,utterance=utterance,depth=0)
		# print(world_prior.shape)

		# I could write: itertools product axes
		scores = np.zeros((world_prior.shape))
		for n_tuple in itertools.product(*[list(range(x)) for x in world_prior.shape]):
			# print(n_tuple)
			# print(world_prior.shape)
		# for j in range(self.number_of_images):
		# 	for i in range(len(rationality_prior)):
			# world.target=j
			world = RSA_World(target=n_tuple[state.dim["image"]],rationality=n_tuple[state.dim["rationality"]],speaker=n_tuple[state.dim["speaker"]])
			# world.set_values(n_tuple)

			# world.rationality=rationality_prior[i]
			# NOTE THAT NOT DEPTH-1 HERE
			out = self.speaker(state=state,world=world,depth=depth)
			# out = np.squeeze(out)

			# print(out,depth)
			scores[n_tuple]=out[utterance]

		scores = scores*state.listener_rationality
		world_posterior = (scores + world_prior) - scipy.misc.logsumexp(scores + world_prior)
		# print("world posterior listener complex shape",world_posterior.shape)
		return world_posterior

	def listener_simple(self,state,utterance,depth):


		# base case listener is either neurally trained, or inferred from neural s0, given the state's current prior on images

		# world = RSA_World(target=0,speaker=0,rationality=0)
		# image_prior = self.listener(state=state,utterance=utterance,depth=depth-1)
		# rationality_prior = np.asarray([0.3,0.7])

		world_prior = state.world_priors[state.timestep-1]
		assert world_prior.shape == (2,1,1)
		print("world prior",np.exp(world_prior))
		# world_prior = np.log(np.asarray([0.5,0.5]))
		# if depth==0:
		# else: world_prior = self.listener(state=state,utterance=utterance,depth=0)
		# print(world_prior.shape)

		# I could write: itertools product axes
		scores = np.zeros((2,1,1))
		for i in range(2):
			# print(n_tuple)
			# print(world_prior.shape)
		# for j in range(self.number_of_images):
		# 	for i in range(len(rationality_prior)):
			# world.target=j
			world = RSA_World(target=i,rationality=0,speaker=0)
			# world.set_values(n_tuple)

			# world.rationality=rationality_prior[i]
			# NOTE THAT NOT DEPTH-1 HERE
			out = self.speaker(state=state,world=world,depth=depth)
			# out = np.squeeze(out)

			# print(out,depth)
			scores[i]=out[utterance]

		scores = scores*state.listener_rationality
		world_posterior = (scores + world_prior) - scipy.misc.logsumexp(scores + world_prior)
		# print("world posterior listener simple shape",world_posterior.shape)

		return world_posterior



