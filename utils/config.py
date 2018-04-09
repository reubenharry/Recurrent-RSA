from collections import defaultdict
IMG_DATA_PATH="charpragcap/resources/visual_genome_data/"
REP_DATA_PATH="charpragcap/resources/resnet_reps/"
TRAINED_MODEL_PATH="data/models/"
WEIGHTS_PATH="charpragcap/resources/weights/"
S0_WEIGHTS_PATH="s0_weights"
S0_PRIME_WEIGHTS_PATH="s0_prime_weights"
caption,region = 0,1
start_token = {"word":"<start>","char":'^'}
stop_token = {"word":"<end>","char":'$'}
pad_token = '&'
sym_set = list('&^$ abcdefghijklmnopqrstuvwxyz')
stride_length = 10
start_index = 1
stop_index = 2
pad_index = 0
batch_size = 50
max_sentence_length = 60

train_size,val_size,test_size = 0.98,0.01,0.01
rep_size = 2048
img_rep_layer = 'hiddenrep'

char_to_index = defaultdict(int)
for i,x in enumerate(sym_set):
    char_to_index[x] = i
index_to_char = defaultdict(lambda:'')
for i,x in enumerate(sym_set):
    index_to_char[i] = x
