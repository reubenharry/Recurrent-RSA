import os.path
from PIL import Image as PIL_Image
import math
from subprocess import call
from charpragcap.utils.image_and_text_utils import vectorize_caption,valid_item,index_to_char,char_to_index,edit_region,get_img_from_id
from charpragcap.utils.config import *

print("rep_size",rep_size)

if __name__ == "__main__":
	
	if os.path.isfile('charpragcap/resources/resnet_reps/reps.pickle') :
		proceed = input('Are you sure you want to rebuild the data? (y/n) ')
	else:
		proceed = 'y'

	if proceed=='y':

		import json
		import numpy as np
		import pandas as pnd
		import pickle
		from keras.models import Model
		from keras.preprocessing import image
		from PIL import Image as PIL_Image
		from charpragcap.resources.models.resnet import resnet

		#define resnet from input to fully connected layer
		fc_resnet = resnet(img_rep_layer)

		def make_id_to_caption():
			valids = 0
			invalids = 0
			id_to_caption = {}
			json_data=json.loads(open('charpragcap/resources/visual_genome_JSON/region_descriptions.json','r').read())
			print("READ JSON, len:",len(json_data))

			

			for i,image in enumerate(json_data):

				for s in image['regions']:

					x_coordinate = s['x']
					y_coordinate = s['y']
					height = s['height']
					width = s['width']
					sentence = s['phrase'].lower()
					img_id = str(s['image_id'])
					region_id = str(s['region_id'])

					is_valid = valid_item(height,width,sentence,img_id)

					if is_valid:
						valids+=1
						box = edit_region(height,width,x_coordinate,y_coordinate)
						id_to_caption[img_id+'_'+region_id] = (vectorize_caption(sentence),box)
					else: invalids+=1

				if i%1000==0 and i>0:
					print("PROGRESS:",i)

				# if i >6000:
				# 	break
				# print(len(id_to_caption))
				# print(id_to_caption)
			print(len(id_to_caption))
			print("num valid/ num invalid",valids,invalids)
			pickle.dump(id_to_caption,open('charpragcap/resources/id_to_caption','wb'))


		# e.g.: id_to_caption = {'10_1382': ('the boy with ice cream',(139,82,421,87)),'11_1382': ('the man with ice cream',(139,82,421,87))}...
		print("MAKING id_to_caption")
		make_id_to_caption()
		print("COMPUTING AND STORING image reps")
		
		#feed each image corresponding to a region in image from id in id_to_caption into a [rep_size] dim vector (or consider fewer)
			#save as pandas dataframe, with labelled columns
		def store_image_reps():

			id_to_caption = pickle.load(open("charpragcap/resources/id_to_caption",'rb'))
			print("len id_to_caption",len(id_to_caption))

			size = 1000
			num_images = len(id_to_caption)
			full_output = np.random.randn(len(id_to_caption),rep_size)
			mod_num = num_images % size
			r = math.ceil(num_images/size)
			for j in range(math.ceil(len(sorted(list(id_to_caption)))/size)):
				print("RUNNING IMAGES THROUGH RESNET: step",j+1,"out of",
					len(range(math.ceil(len(list(id_to_caption))/size))))
				if j == r -1:
					num = mod_num
				else:
					num = size
				img_tensor = np.zeros((num, 224,224,3))
				
				for i,item in enumerate(sorted(list(id_to_caption))[j*size:((j*size)+num)]):

					img = get_img_from_id(item,id_to_caption)
					img_vector = image.img_to_array(img)
					img_tensor[i] = img_vector
				
				reps = fc_resnet.predict(img_tensor)
				# print("check",reps.shape[0],len(list(id_to_caption)[j*size:((j*size)+num)]))
				assert reps.shape[0]==len(list(id_to_caption)[j*size:((j*size)+num)])
				full_output[j*size:j*size+num] = reps[:num]
				


				df = pnd.DataFrame(full_output,index=sorted(list(id_to_caption)))

				assert df.shape == (len(id_to_caption),rep_size)

				df.to_pickle(REP_DATA_PATH+"reps.pickle")
				if j%10==0: df.to_pickle(REP_DATA_PATH+"reps.pickle_backup")

		# store_image_reps()
