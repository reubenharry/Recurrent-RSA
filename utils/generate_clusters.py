import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from charpragcap.utils.image_and_text_utils import devectorize_caption,split_dataset,get_rep_from_id
import copy

def cap_to_words(cap):
	return [word for word in re.sub('[^a-z ]',"",devectorize_caption(cap)).split() if word not in stopwords.words('english')]

if __name__ == '__main__':

	make = False
	name = "test_clusters"

	id_to_caption = pickle.load(open("charpragcap/resources/id_to_caption",'rb'))
	trains,vals,tests = split_dataset()

	# new_dic = {}
	# for x in val:
	# 	new_dic[x]=id_to_caption[x]
	# id_to_caption=new_dic

	# print(len(list(id_to_caption)))
	id_to_caption = pickle.load(open("charpragcap/resources/id_to_caption",'rb'))
	ids = [x for x in trains+tests+vals if int(x.split("_")[0])>414114][:10000]

	# vocab = {}
	# for x in ids:
	# 	words = [word for word in re.sub('[^a-z ]',"",cap).split() if word not in stopwords.words('english')] 
	# 	for word in words:
	# 		vocab.add(word)

	# vocab = sorted(list(vocab))

	if make:

		print(len(ids))

		id_to_words={}

		cluster_mat = np.zeros((len(ids),len(ids)))

		for i,idx in enumerate(ids):
			print(i)

			try: sent_1=id_to_words[idx]
			except:
				sent_1=set(cap_to_words(id_to_caption[idx][0][0]))
				id_to_words[idx]=sent_1

			for j,idx2 in enumerate(ids):

				try: sent_2 = id_to_words[idx2]
				except:
					sent_2=set(cap_to_words(id_to_caption[idx2][0][0]))			
					id_to_words[idx2]=sent_2

				overlap= len(sent_1.intersection(sent_2))
				cluster_mat[i,j]=overlap

		print("MADE MATRIX")
		pickle.dump(cluster_mat,open("charpragcap/resources/cluster_mat",'wb'))
		pickle.dump(id_to_words,open("charpragcap/resources/id_to_words",'wb'))

	cluster_mat=pickle.load(open("charpragcap/resources/cluster_mat",'rb'))
	id_to_words=pickle.load(open("charpragcap/resources/id_to_words",'rb'))

	excluded=[]
	clusters = []
	for i in range(len(ids)):
		if ids[i] not in excluded:

			cluster = [ids[x] for x in np.argsort(-cluster_mat[i])[:10]]
			print(cluster)
			print([id_to_words[x] for x in cluster])
			clusters.append(cluster)
			excluded+=cluster

	pickle.dump(clusters, open("charpragcap/resources/cluster_dicts/"+name,'wb'))


