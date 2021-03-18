import numpy as np
from sklearn.decomposition import PCA
import utils


class HardDebias():
        
    
    def idtfy_gender_subspace(embedding, word_sets, w2id, defining_sets, k=1):
        """
        identifies the bias (gender) subspace following Bolukbasi et al. 2016

        takes
        word_sets:      vocabulary
        w2id:           a dictionary to translate words contained in the vocabulary into their corresponding IDs
        defining_sets:  N defining sets (pairs if I=2) consisting of I words that differ mainly on the bias (gender) direction
        embedding:      the embedding of the vocabulary
        k:              an integer parameters that defines how many rows of SVD(C) constitute the bias (gender) subspace B, 
                        bias (gender) direction if k=1

        returns
        bias_subspace:  linear bias (gender) subspace (direction) that is assumed to capture most of the gender bias 
                        (denoted as B in Bolukbasi et al. 2016)
        """

        def embed(word, w2id=w2id, embedding=embedding):
            return embedding[w2id[word]]
        
        # following Bolukbasi et al. (2016)
        C = []
        for female_word, male_word in defining_sets:
            mean = (embed(female_word) + embed(male_word)) /2
            C.append(embed(female_word) - mean)
            C.append(embed(male_word) - mean)
        C = np.array(C)

        # applying PCA is the same as SVD when interpreting C as covariance matrix (Vargas & Cotterell 2020)
        pca = PCA(n_components = 10)
        pca.fit(C)

        # take the first k PCs (first for gender direction)
        B = []
        for i in range(k):
            B.append(pca.components_[i])
        B = np.array(B).flatten()

        return B
    
    
    def hard_debias (embedding, w2id, word_emb, equality_sets, B):
        """performs hard debias on a word embedding to neutralize it,

        takes 
        word_emb: word embedding of the word to be neutralized,
        equalize_pairs: equality pairs, each neutral word should be equidistant to all words in each equality set
        B: the bias subspace

        returns
        new_word_emb: the new embedding for word_emb
        """
        
        def embed(word, w2id=w2id, embedding=embedding):
            return embedding[w2id[word]]

        # if word_emb is a single embedding:
            # w_orth(word_emb)

        # if word_emb is the embeddings of all words to be neutralized:

        new_word_emb = np.zeros((word_emb.shape))
        for i, embedding in enumerate(word_emb):
            new_word_emb[i] = utils.w_orth(embedding, B)

        for equal_set in equality_sets:
            if equal_set[0] in w2id and equal_set[1] in w2id:
                mean = (embed(equal_set[0]) + embed(equal_set[1])) / 2
                mean_biased = mean - utils.w_orth(mean, B)
                v = mean - mean_biased 
                for word in equal_set:
                    word_biased = embed(word) - utils.w_orth(embed(word), B)
                    # new_embed = v + np.sqrt(1 - (np.linalg.norm(v)) ** 2) * ((word_biased - mean_biased) / unit_vec(word_biased - mean_biased))
                    new_embed = v * ((word_biased - mean_biased) / unit_vec(word_biased - mean_biased))

        return new_word_emb
    
    
    print("successfully loaded hard_debias")