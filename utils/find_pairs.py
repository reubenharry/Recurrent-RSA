import re
import pickle
from charpragcap.utils.image_and_text_utils import split_dataset,index_to_char,devectorize_caption
from nltk.corpus import stopwords


id_to_caption = pickle.load(open("charpragcap/resources/id_to_caption",'rb'))
train,val,test = split_dataset(id_to_caption)
print(len(test))

def cap_to_words(cap):
	return [word for word in re.sub('[^a-z ]',"",devectorize_caption(cap)).split() if word not in stopwords.words('english')],devectorize_caption(cap)





def find_pairs(item):

	cap = set(cap_to_words(id_to_caption[item][0][0])[0])
	fst_half_id,snd_half_id = item.split('_')
	l=[]
	
	for t in test:
		fst_half_new_id,snd_half_new_id = t.split('_')
		if fst_half_id!=fst_half_new_id:
			new_cap,full_cap = cap_to_words(id_to_caption[t][0][0])
			l.append((len(set(new_cap).intersection(cap)),full_cap,t))

	return (sorted(l,key=lambda x: x[0],reverse=True)[:30])

def all_pairs():
	out = []
	for i,t in enumerate(test):
		print(i)
		pairs = find_pairs(t)
		# if pairs[1][0]>1:
		print(pairs[:3])
		if pairs[2][0]>1:
			out.append(pairs[:3])
		if i>5000:
			break

	return sorted(out,key=lambda x:x[-1])
			# break

if __name__ == '__main__':
	out = list(all_pairs())
	print(out)
	pickle.dump(out,open('charpragcap/resources/distractor_pairs_short','wb'))
