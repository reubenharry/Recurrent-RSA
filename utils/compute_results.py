import pickle
import numpy as np
from charpragcap.rsa.cataRSA import CataRSA
from charpragcap.rsa.cataRSA_working_beam import CataRSA as CataRSA_working_beam
from charpragcap.utils.config import *
from charpragcap.utils.image_and_text_utils import char_to_index,vectorize_caption,get_rep_from_img_id,split_dataset
from charpragcap.utils.generate_clusters import generate_clusters
from charpragcap.utils.urls import reps


id_to_caption = pickle.load(open("charpragcap/resources/id_to_caption",'rb'))
train,val,test = split_dataset(id_to_caption)

full_test_ids = sorted(list(set([x.split('_')[0] for x in test])))

cataRSA = CataRSA(
	imgs=[reps[0]],
	img_paths=[],
	trained_s0_path=WEIGHTS_PATH+S0_WEIGHTS_PATH,
	trained_s0_prime_path=WEIGHTS_PATH+S0_PRIME_WEIGHTS_PATH,
	l0_type='from_s0',
	)

def compute_results(model,images):

	s0_results = {}
	for img_id in images:
		model.images=np.array([get_rep_from_img_id(img_id)])
		# out = model.ana_greedy(speaker_rationality=1.0, listener_rationality=1.0, depth=0,start_from="",img_prior=np.log(np.asarray([0.5])))
		out = model.ana_beam(
			speaker_rationality=1.0, 
			listener_rationality=1.0, 
			depth=0,start_from="",
			decay_rate=-1.0,
			img_prior=np.log(np.asarray([0.5]))
			)

		s0_results[img_id]=out
		print("RESULTS:",out)
		model._speaker_cache = {}
		model._listener_cache = {}


		pickle.dump(s0_results,open("charpragcap/resources/s0_results_beam",'wb'))

# compute_results(cataRSA,full_test_ids[:100])
generate_clusters()


def compute_results_pragmatic(model,depth,name):
	clusters = pickle.load(open("charpragcap/resources/clusters",'rb'))
	s0_results = {}
	out = model.ana_beam(
		speaker_rationality=1.0, 
		listener_rationality=1.0, 
		depth=depth,start_from="",
		decay_rate=-1.0,
		img_prior=np.log(np.asarray([1/2,1/2]))
		)
	for items in clusters[:20]:
		model.images=np.array([get_rep_from_img_id(item[1]) for item in items])
		# out = model.ana_greedy(speaker_rationality=1.0, listener_rationality=1.0, depth=0,start_from="",img_prior=np.log(np.asarray([0.5])))

		s0_results[tuple([(item[1],item[0]) for item in items])]=out
		print(out)
		print(items)
		model._speaker_cache = {}
		model._listener_cache = {}


		pickle.dump(s0_results,open("charpragcap/resources/s0_results_"+name,'wb'))

compute_results_pragmatic(cataRSA,1,"depth_1")
compute_results_pragmatic(cataRSA,2,"depth_2")
compute_results_pragmatic(cataRSA,3,"depth_3")