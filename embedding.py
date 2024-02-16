import json
import numpy as np

# load the result
with open('./data/word_to_index.json', 'r') as f:
    word_to_index = json.loads(f.read())

embeddings = np.zeros((len(word_to_index), 50))
filled = np.zeros(len(word_to_index))
with open('glove.6B/glove.6B.50d.txt','rt') as fi:
    full_content = fi.read().strip().split('\n')

cnt = 0
for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    if i_word in word_to_index:
        embeddings[int(word_to_index[i_word])] = np.array(i_embeddings)
        filled[int(word_to_index[i_word])] = 1
        cnt += 1
    if cnt == len(word_to_index):
        break

# randomize the rest of the embeddings
for i in range(len(filled)):
    if filled[i] == 0:
        embeddings[i] = np.random.rand(50)

np.save('./data/embeddings.npy', embeddings)