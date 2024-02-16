import re
import json

file_path = './data/TWITTER-Real-Graph-Partial.nel'

with open(file_path, 'r') as f:
    data = f.read()

# remove all lines not starting with 'n'
data = re.sub(r'^(?!n).*\n', '', data, flags=re.MULTILINE)

# remove 'n number' from each line
data = re.sub(r'^n \d+ ', '', data, flags=re.MULTILINE)

# traverse data, create index for each word
data = data.split()
word_to_index = {}
index = 0
for word in data:
    if word not in word_to_index:
        word_to_index[word] = index
        index += 1

# output the result
print(word_to_index)

# save the result
with open('./data/word_to_index.json', 'w') as f:
    f.write(json.dumps(word_to_index))

# get index to word
index_to_word = {v: k for k, v in word_to_index.items()}
print(index_to_word)

# save the result
with open('./data/index_to_word.json', 'w') as f:
    f.write(json.dumps(index_to_word))
    