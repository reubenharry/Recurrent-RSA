import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from utils.build_vocab import Vocabulary
from train.image_captioning.char_model import EncoderCNN, DecoderRNN
from PIL import Image
import torch
from utils.config import *
from utils.numpy_functions import softmax


class Model:

	def __init__(self,path,dictionaries):
		
		self.seg2idx,self.idx2seg=dictionaries
		self.path=path
		self.vocab_path='data/vocab.pkl'
		self.encoder_path=TRAINED_MODEL_PATH+path+"-encoder-5-3000.pkl"
		self.decoder_path=TRAINED_MODEL_PATH+path+"-decoder-5-3000.pkl"

		#todo: change
		embed_size=256
		hidden_size=512
		num_layers=1

		output_size=30

		transform = transforms.Compose([
			transforms.ToTensor(), 
			transforms.Normalize((0.485, 0.456, 0.406), 
								 (0.229, 0.224, 0.225))])
		
		self.transform = transform
		# Load vocabulary wrapper


		# Build Models
		self.encoder = EncoderCNN(embed_size)
		self.encoder.eval()  # evaluation mode (BN uses moving mean/variance)
		self.decoder = DecoderRNN(embed_size, hidden_size, 
							 output_size, num_layers)
		
		# Load the trained model parameters
		self.encoder.load_state_dict(torch.load(self.encoder_path,map_location={'cuda:0': 'cpu'}))
		self.decoder.load_state_dict(torch.load(self.decoder_path,map_location={'cuda:0': 'cpu'}))

		if torch.cuda.is_available():
			self.encoder.cuda()
			self.decoder.cuda()



	def forward(self,world,state):


		inputs = self.features[world.target].unsqueeze(1)


		states=None

		for seg in state.context_sentence:									  # maximum sampling length
			hiddens, states = self.decoder.lstm(inputs, states)		  # (batch_size, 1, hidden_size), 

			outputs = self.decoder.linear(hiddens.squeeze(1)) 

			predicted = outputs.max(1)[1]   

			predicted[0] = self.seg2idx[seg]
			inputs = self.decoder.embed(predicted)
			inputs = inputs.unsqueeze(1)		# (batch_size, vocab_size)

		hiddens, states = self.decoder.lstm(inputs, states)		  # (batch_size, 1, hidden_size), 
		outputs = self.decoder.linear(hiddens.squeeze(1)) 
		output_array = outputs.squeeze(0).data.cpu().numpy()

		log_softmax_array = np.log(softmax(output_array))



		return log_softmax_array

	def set_features(self,images,rationalities,tf):

		self.number_of_images = len(images)
		self.number_of_rationalities = len(rationalities)
		self.rationality_support=rationalities

		if tf:
			pass

		else:
			from utils.sample import to_var,load_image,load_image_from_path
			self.features = [self.encoder(to_var(load_image(url, self.transform), volatile=True)) for url in images]
			self.default_image = self.encoder(to_var(load_image_from_path("data/default.jpg", self.transform), volatile=True))


			# self.speakers = [Model(path) for path in paths]

			# imgs = [load_image(url) for url in urls]
			# self.images=[]

			# for img in imgs:
			# 	img_array = np.expand_dims(image.img_to_array(img),0)
			# 	img_rep = resnet(img_rep_layer).predict(img_array)
			# 	self.images.append(img_rep)

			# self.images = np.asarray(self.images)


			