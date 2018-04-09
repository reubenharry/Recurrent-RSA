# show a first images and their ground truth captions
# check that stored rep = generated rep for a few random ones
# cehck that unmemoized is same as memoized

# tests that the saved reps are aved in the right order and so on

def check_reps():

	repsandcaps = single_stream(train,X0_type='rep')
	idsandcaps = single_stream(train,X0_type='id')
	for i in range(10):
		repandcaps = next(repsandcaps)
		idandcaps = next(idsandcaps)
		img_id = idandcaps[0]
		img_rep = repandcaps[0]
		print("img id:",img_id)
		real_rep = item_to_rep(img_id)
		stored_rep = np.expand_dims(img_rep,0)		

		print("real rep",real_rep)
		print("stored rep",stored_rep)
		print(real_rep==stored_rep)
		assert(np.array_equal(real_rep,stored_rep))

def view_data():
	for full_id,cap_in,cap_out in single_stream(test,X0_type='id'):
	    img = get_img_from_id(full_id,id_to_caption)
	    display(img)
	    print("".join([index_to_char[x] for x in cap_in]))
	    print(''.join([index_to_char[np.argmax(x)] for x in cap_out]))
	    break

