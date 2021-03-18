import urllib.request
import numpy as np


def read_embedding(url, skip_first = False):
    """
    Function to read out an embedding
    
    takes
    url: url to embedding

    returns
    vocab: list of words in the embedding
    w2id: dictionary mapping words to ids
    embedding: array storing the word vectors, row corresponds to word id
    """
    # Open url
    data = urllib.request.urlopen(url)
    vocab = []
    embedding = []

    # Each line contains one word and its embedding
    for i, line in enumerate(data):
        if skip_first:
            if i == 0:
                continue
        #if len(line) == 301:
        line = line.decode()
        # Split by spaces
        split = line.split()
        # First element(== the word) is added to vocabulary
        vocab.append(split[0])
        # All other elements(embedding vectors) are added to vectors
        embedding.append([float(elem) for elem in split[1:]])

    # Create a dictionary with word-id pairs based on the order
    w2id = {w: i for i, w in enumerate(vocab)}
    # Vectors are converted into an array
    embedding = np.array(embedding, dtype=np.float32) # due to problems with memory

    return vocab, w2id, embedding


def restrict_vocab(vocab, w2id, embedding):
    """
    Limits the vocab by removing words containing digits or special characters
        
    takes
    vocab: list of words in the embedding
    w2id: dictionary mapping words to ids
    embedding: array storing the word vectors

    returns
    limit_vocab: list of words in vocab that do not include digits or special characters
    limit_w2id: dictionary mapping words in limit_vocab to new ids
    limit_embedding: array storing the word vectors of the words in limit_vocab only
    """
        
    limit_vocab = []
    limit_embedding = []

    for i, word in enumerate(vocab[:50000]): # hoping that this gives us the most common words
        # If word includes either a digit or a special character move on to next word
        if (hasDigit(word) or hasSpecialChar(word)):
            continue
        # Else add word to limit_vocab and its embedding to limit_embedding    
        limit_vocab.append(word)
        limit_embedding.append(embedding[w2id[word]])

    # Convert embedding into an array    
    limit_embedding = np.array(limit_embedding, dtype=np.float32)#astype(float)
    # Create new dictionary containing only the words in limit_vocab and their new ids
    limit_w2id = {word: i for i, word in enumerate(limit_vocab)}

    return limit_vocab, limit_w2id, limit_embedding


def exclude_vocab(vocab, exclude):
    """
    Function to exclude specific words from vocabulary
        
    takes
    vocab: list of words in the embedding
    exclude: list of words to exclude from the vocabulary

    returns
    limited_vocab: vocab without the words in exclude
    """
        
    # Create copy of vocab
    limited_vocab = vocab.copy()
    # For all words that are in exclude and vocab
    for word in exclude:
        if word in limited_vocab:
            # Remove word from vocab
            limited_vocab.remove(word)

    return limited_vocab


def hasDigit(word):
    """Checks if a string contains any digits"""
    return any(char.isdigit() for char in word)


def hasSpecialChar(word):
    """Checks if a string contains special characters(except "_")"""
    special_characters = "!@#$%^&*()-+?=,<>/."
    return any(char in special_characters for char in word)


def unit_vec(vector):
    """calculates unit vector of passed vector"""
    
    unit = np.linalg.norm(vector)
    if unit != 0 and np.isnan(unit) == False :
        return vector/unit
    return vector   


def w_orth (word_emb, direction):
    """
    removes direction from word embedding by calculating the orthogonal word vector
    
    w_orth = w - (projection of w onto direction)
    w_orth = w - (direction * (w dot direction))
    
    takes 
    word_emb: word to remove the direction from
    direction: the direction to remove
    
    returns
    unit vector embedding orthogonal to direction   
    """
    
    # formula from Bolukbasi et al. (2016)
    new_word = word_emb - ((word_emb.dot(direction)) * direction)
    
    return unit_vec(new_word)


def debias_gn(wv):
    for v in wv:
        assert(len(v) == 300)

    wv = wv[:,:-1]

    for v in wv:
        assert(len(v) == 299)
    return wv


    
print("successfully loaded utils")