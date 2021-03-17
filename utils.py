import urllib.request
import numpy as np

class utils():
    """A collection of useful functions"""
    
    def read_embedding(url, skip_first = False):
        """Function to read out an embedding
        Input: url: url to embedding

        Returns: vocab: list of words in the embedding
                 w2id: dictionary mapping words to ids
                 embedding: array storing the word vectors,
                               row corresponds to word id"""
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
        embedding = np.array(embedding).astype(float)

        return vocab, w2id, embedding


    def restrict_vocab(vocab, w2id, embedding):
        """Limits the vocab by removing words containing digits or special characters
        Input: vocab: list of words in the embedding
               w2id: dictionary mapping words to ids
               embedding: array storing the word vectors

        Returns: limit_vocab: list of words in vocab that do not include digits or special characters
                 limit_w2id: dictionary mapping words in limit_vocab to new ids
                 limit_embedding: array storing the word vectors of the words in limit_vocab only"""
        limit_vocab = []
        limit_embedding = []

        for i, word in enumerate(vocab[:50000]): # hoping that this gives us the most common words
            # If word includes either a digit or a special character move on to next word
            if (utils.hasDigit(word) or utils.hasSpecialChar(word)):
                continue
            # Else add word to limit_vocab and its embedding to limit_embedding    
            limit_vocab.append(word)
            limit_embedding.append(embedding[w2id[word]])

        # Convert embedding into an array    
        limit_embedding = np.array(limit_embedding).astype(float)
        # Create new dictionary containing only the words in limit_vocab and their new ids
        limit_w2id = {word: i for i, word in enumerate(limit_vocab)}

        return limit_vocab, limit_w2id, limit_embedding
    
    
    def hasDigit(word):
        """Checks if a string contains any digits"""
        return any(char.isdigit() for char in word)

    def hasSpecialChar(word):
        """Checks if a string contains special characters(except "_")"""
        special_characters = "!@#$%^&*()-+?=,<>/."
        return any(char in special_characters for char in word)