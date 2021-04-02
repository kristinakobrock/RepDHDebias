import utils
import urllib.request
import numpy as np
import json


# URL to retrive pre-trained 300 dimensional gloVe embedding
embedding_300_url = "http://www.cs.virginia.edu/~tw8cb/word_embeddings/vectors.txt"
vocab_original, w2id_original, embedding_original = utils.read_embedding(embedding_300_url)
vocab, w2id, embedding = utils.restrict_vocab(vocab_original, w2id_original, embedding_original)
print("Original vocab size: ", len(vocab_original))
print("Restricted vocab size: ", len(vocab))

# URL to female specific words as listed by the authors
female_words_url = "https://raw.githubusercontent.com/uvavision/Double-Hard-Debias/master/data/female_word_file.txt"
female_words_data = urllib.request.urlopen(female_words_url)

# List of female words
female_words = []
for line in female_words_data:
    line = line.decode()
    line = line.split()
    female_words.append(line[0])
        
# URL to male specific words as listed by the authors
male_words_url = "https://raw.githubusercontent.com/uvavision/Double-Hard-Debias/master/data/male_word_file.txt"
male_words_data = urllib.request.urlopen(male_words_url)

# List of male words
male_words = []
for line in male_words_data:
    line = line.decode()
    line = line.split()
    male_words.append(line[0])
    
# Create List with female - male pairs from female-male specific words
female_male_pairs = []
for i, female in enumerate(female_words):
    female_male_pairs.append([female, male_words[i]])
    
# URLs to the files storing gender specific words as listed by the authors
gender_specific_url = "https://raw.githubusercontent.com/uvavision/Double-Hard-Debias/master/data/gender_specific_full.json"

# Empty list to accumulate gender specific words plus additional list after lowercasing
gender_specific_original = []
gender_specific = []


# Read out URL and add further gender specific words
with urllib.request.urlopen(gender_specific_url) as f:
    gender_specific_original.extend(json.load(f))

# Add lower case words to second list
for word in gender_specific_original:
    gender_specific.append(word.lower())
    
# URL to the file storing definitional pairs as listed by the authors
definitional_pairs_url = "https://raw.githubusercontent.com/uvavision/Double-Hard-Debias/master/data/definitional_pairs.json"

# Empty list to store definitional pairs plus additional list after lowercasing
definitional_pairs_original = []
definitional_pairs = []


# Read out url and add pairs in list
with urllib.request.urlopen(definitional_pairs_url) as f:
    definitional_pairs_original.extend(json.load(f))
    
# Add lower case pairs to second list
for [w1, w2] in definitional_pairs_original:
    definitional_pairs.append([w1.lower(), w2.lower()])


# Create list of single words instead of pairs  
definitional_words = []
for pair in definitional_pairs:
    for word in pair:
        definitional_words.append(word)
        
# URL to the file storing the equalize pairs as listed by the authors
equalize_pairs_url = "https://raw.githubusercontent.com/uvavision/Double-Hard-Debias/master/data/equalize_pairs.json"

# Empty list to store equalize pairs plus additional list after lowercasing
equalize_pairs_original = []
equalize_pairs = []

# Read out URL and add pairs to list
with urllib.request.urlopen(equalize_pairs_url) as f:
    equalize_pairs_original.extend(json.load(f))
    
# Add lower case pairs to second list
for [w1, w2] in equalize_pairs_original:
    equalize_pairs.append([w1.lower(), w2.lower()])
    
# Create list of single words instead of pairs
equalize_words = []
for pair in equalize_pairs:
    for word in pair:
        equalize_words.append(word)
        
# List of all gender specific words included in 
# female words, male words, gender specific words, equalize words and definitional words
exclude_words = list(set(female_words + male_words + gender_specific + definitional_words + equalize_words))


def embed(word, w2id=w2id, embedding=embedding):
    return embedding[w2id[word]]


# Remove gender specific words from the embedding to obtain vocabulary of neutral words
vocab_neutral = utils.exclude_vocab(vocab, exclude_words)
# save both neutral embeddings and the indices of the neutral words in original vocab
embedding_neutral = np.asarray([embed(word) for word in vocab_neutral])
id_neutral = [w2id[neutral] for neutral in vocab_neutral]
print("Neutral vocab size: ", len(vocab_neutral))


print("successfully loaded data_embeddings")