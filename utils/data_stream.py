import pickle
import numpy as np
import more_itertools
from PIL import Image as PIL_Image
from keras.preprocessing import image
from charpragcap.utils.image_and_text_utils import split_dataset,index_to_char,get_img_from_id
from charpragcap.utils.config import *
from charpragcap.resources.models.resnet import resnet


id_to_caption = pickle.load(open("charpragcap/resources/id_to_caption",'rb'))
reps = pickle.load(open(REP_DATA_PATH+'reps.pickle','rb'))
fc_resnet = resnet(img_rep_layer)


def single_stream(ids,X0_type='rep'):

	# TODO
	#find the other ids with same first part: hopefully can just look down the dict, if sorted right: check that
		#iterate through this list, taking each as a starting id, and returning the whole list of ids
		#use the lookup in a vectorized way (implemented in pandas) to get reps, if reps needed

	for full_id in ids:
		pairs=False
		if type(full_id)==tuple: pairs=True

	

		if pairs: 


			fst_id,snd_id = full_id
			if X0_type=='rep':
				try:
					fst_X0,snd_X0 = reps.ix[fst_id].values,reps.ix[snd_id].values
				except Exception:
					out = []
					for idx in [fst_id,snd_id]:
						img = get_img_from_id(idx,id_to_caption)
						img_vector = image.img_to_array(img)
						out.append(fc_resnet.predict([img_vector]))
					fst_X0,snd_X0 = tuple(out)
			elif X0_type=='id':
				fst_X0,snd_X0 = fst_id,snd_id
			fst_X1,fst_Y = id_to_caption[fst_id][caption]
			snd_X1,snd_Y = id_to_caption[snd_id][caption]
			yield (fst_X0,fst_X1,fst_Y,snd_X0,snd_X1,snd_Y)
			
		else:
			if X0_type=='rep': 
				try: 
					X0 = reps.ix[full_id].values
				except KeyError:
					img = get_img_from_id(full_id,id_to_caption)

					img_vector = np.expand_dims(image.img_to_array(img),0)
					print("\n\n\nshape",img_vector.shape)
					X0 = fc_resnet.predict(img_vector)

					print("\n\n\nGOT IMAGE\n\n\n")


			elif X0_type=='id': X0 = full_id
			X1,Y = id_to_caption[full_id][caption]
			yield (X0,X1,Y)

#divides stream into chunks, i.e. minibatches
def chunked_stream(ids):
	chunks = more_itertools.chunked(single_stream(ids),batch_size)

	for chunk in chunks:
		x1s,x2s,ys = list(zip(*chunk))
		yield ([np.asarray(x1s),np.expand_dims(np.asarray(x2s),-1)],np.asarray(ys))

#cycles chunked stream
def data(ids):
	# if X0_type=='rep':
	# 	reps = pickle.load(open(REP_DATA_PATH+'reps.pickle','rb'))
	while True:
		yield from chunked_stream(ids)

train,val,test = split_dataset()



#dimension checks

# a = (next(data(test)))
# assert a[0][0].shape[1] == rep_size
# assert a[0][1].shape[1] == a[1].shape[1] == (max_sentence_length+1)
# assert a[1].shape[2] == len(sym_set)
# print("PASSED DIMENSION CHECKS")