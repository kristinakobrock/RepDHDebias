import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics as sk_m
from sklearn.cluster import KMeans
import utils
import hard_debias as hd


def most_biased(embedding, B, k=500):
    """function to obtain the most biased female and male words"""

    all_biased = np.ndarray((len(embedding),1))
    for i, word in enumerate(embedding):
        all_biased[i] = (sk_m.pairwise.cosine_similarity(word.reshape(1, 300), B.reshape(1, 300)))[0]

    most_biased_f = []
    most_biased_m = []

    for word in range(k):
        # female words
        fb_index = np.argmin(all_biased)
        most_biased_f.append(fb_index)
        all_biased[fb_index] = 0
        # male words
        mb_index = np.argmax(all_biased)
        most_biased_m.append(mb_index)
        all_biased[mb_index] = 0

    return most_biased_f, most_biased_m


#Gender alignment accuracy/ Neighborhood Metric:
def align_acc(males, females, k=2):
    """bias measurement using KMeans Clustering

    takes
    males: male words' embeddings
    females: female words'embeddings
    k: number of clusters to create
    ground truth labels:
    0 = male,
    1 = female

    returns
    alignment: alignment accuracy after clustering the embeddings according to gender
    """

    array_m_f = np.concatenate((males,females))

    #need: k (=1000) most biased female and male word's embedding (cosine similarity embedding & gender direction),
    # perform KMeans on embeddings with k=2
    # due to performance constraints only perform KMeans once
    kmeans = KMeans(n_clusters=k, random_state=0).fit(array_m_f)
    split = males.shape[0]

    correct = 0

    # compute alignment score: cluster assignment vs ground truth gender label
    for i in range(array_m_f.shape[0]):
        # correct clustering if word was assigned its ground truth label
        if i < split and kmeans.labels_[i] == 0:
            correct+= 1
        elif i >= split and kmeans.labels_[i] == 1:
            correct += 1

    # alignment score = max(a, 1-a)
    alignment = (1/(array_m_f.shape[0])) * correct
    alignment = np.maximum(alignment, 1-alignment)

    return alignment 


def double_hard_debias(embedding, w2id, embedding_neutral, id_neutral,  equality_sets, index_m, index_f, gender_subspace):
    """
    Double Hard Debias as proposed by Wang et al. (2020):

    takes
    embedding: all embeddings
    embedding_neutral: subset of embeddings that the bias should be removed from
    index_m: indices of most biased male words 
    index_f: indices of most biased female words 

    returns
    double_debias: full set of double-debiased embeddings
    """

    # create word lists of most biased male and female words
    males = np.asarray([embedding[i] for i in index_m])
    females = np.asarray([embedding[i] for i in index_f])    

    # decentralize all of the embeddings and store seperately
    words_decen = np.zeros((len(embedding),300), dtype='float32') # chose a smaller data type due to memory error
    words_neutral_decen = np.zeros((len(embedding_neutral),300), dtype='float32')
    # first calculate mean over full vocab
    mu = ((len(embedding)**(-1)) * np.sum(embedding, axis=0)).reshape(1,300)
    # then subtract mean from each word embedding
    for index, emb in enumerate(embedding):
        words_decen[index] = emb - mu

    for index, emb in enumerate(embedding_neutral):
        words_neutral_decen[index] = emb - mu


    # discover the frequency direction    
    # for all decentralized embeddings: compute PCA
    pca_freq = PCA().fit(words_decen)

    evaluations = []

    # in implementation of paper only consider 20 first PCs
    for i, pc in enumerate(pca_freq.components_):
        if i < 20:
            male_proj = np.zeros((len(males),300))
            male_debias = np.zeros((len(males),300))
            female_proj = np.zeros((len(females),300))
            female_debias = np.zeros((len(females),300))


            # remove PC direction and gender direction from all embeddings
            for index, male in enumerate(males):

                # remove direction of current PC
                male_proj[index] = utils.w_orth(male, pc)
                # remove gender direction: hard debias
                male_debias[index] = utils.w_orth(male_proj[index], gender_subspace)

            # repeat for female-biased words
            for index, female in enumerate(females):

                female_proj[index] = utils.w_orth(female, pc)
                female_debias[index] = utils.w_orth(female_proj[index], gender_subspace)

            # apply Neighbourhood Metric
            # compute gender alignment accuracy for each PC
            evaluations.append(align_acc(male_debias, female_debias))


    # evaluate which PC-rejection leads to most random cluster = evaluation smallest (closest to 0.5) 
    # in original paper corresponded to second PC    
    print("smallest PC: ", np.argmin(evaluations))
    best_pc = pca_freq.components_[np.argmin(evaluations)]

    first_debias = np.zeros((embedding.shape))
    first_neutral_debias = []
    # remove best PC-direction from all neutral, decentralized words
    for index, word in enumerate(words_decen):

        if index in id_neutral:
            first_debias[index] = utils.w_orth(word, best_pc)
            first_neutral_debias.append(utils.w_orth(word, best_pc))
        else:
            first_debias[index] = utils.unit_vec(word)

    first_neutral_debias = np.asarray(first_neutral_debias)

    # apply HardDebias to all neutral, once debiased, words
    double_neutral_debias = hd.hard_debias(embedding, w2id, first_neutral_debias, equality_sets, gender_subspace)

    double_debias = first_debias.copy()

    for index, word in enumerate(double_neutral_debias):
        double_debias[id_neutral[index]] = word

    return double_debias

    
print("successfully loaded double_hard_debias")