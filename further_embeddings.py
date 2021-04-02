import utils
import urllib.request
import numpy as np
import json

# Load further embeddings

embedding_url = "http://www.cs.virginia.edu/~tw8cb/word_embeddings/vectors.txt"
vocab_original, w2id_original, embedding_original = utils.read_embedding(embedding_url)
vocab, w2id, embedding = utils.restrict_vocab(vocab_original, w2id_original, embedding_original)

embedding_gn_url = "http://www.cs.virginia.edu/~tw8cb/word_embeddings/vectors300.txt"
vocab_gn_original, w2id_gn_original, embedding_gn_original = utils.read_embedding(embedding_gn_url)
vocab_gn, w2id_gn, embedding_gn = utils.restrict_vocab(vocab_gn_original, w2id_gn_original, embedding_gn_original)

vocab_gn_a = vocab_gn
w2id_gn_a = w2id_gn
embedding_gn_a = utils.debias_gn(embedding_gn)

embedding_hd_url = "http://www.cs.virginia.edu/~tw8cb/word_embeddings/vectors_hd.txt"
vocab_hd_original, w2id_hd_original, embedding_hd_original = utils.read_embedding(embedding_hd_url)
vocab_hd, w2id_hd, embedding_hd = utils.restrict_vocab(vocab_hd_original, w2id_hd_original, embedding_hd_original)

embedding_hd_a_url = "http://www.cs.virginia.edu/~tw8cb/word_embeddings/vectors_hd_a.txt"
vocab_hd_a_original, w2id_hd_a_original, embedding_hd_a_original = utils.read_embedding(embedding_hd_a_url)
vocab_hd_a, w2id_hd_a, embedding_hd_a = utils.restrict_vocab(vocab_hd_a_original, w2id_hd_a_original, embedding_hd_a_original)

embedding_gp_url = "http://www.cs.virginia.edu/~tw8cb/word_embeddings/gp_glove.txt"
vocab_gp_original, w2id_gp_original, embedding_gp_original = utils.read_embedding(embedding_gp_url, skip_first = True)
vocab_gp, w2id_gp, embedding_gp = utils.restrict_vocab(vocab_gp_original, w2id_gp_original, embedding_gp_original)

embedding_gp_gn_url = "http://www.cs.virginia.edu/~tw8cb/word_embeddings/gp_gn_glove.txt"
vocab_gp_gn_original, w2id_gp_gn_original, embedding_gp_gn_original = utils.read_embedding(embedding_gp_gn_url, skip_first = True)
vocab_gp_gn, w2id_gp_gn, embedding_gp_gn = utils.restrict_vocab(vocab_gp_gn_original, w2id_gp_gn_original, embedding_gp_gn_original)

# load definitional pairs in order to be able to identify gender subspace
definitional_pairs_url = "https://raw.githubusercontent.com/uvavision/Double-Hard-Debias/master/data/definitional_pairs.json"
definitional_pairs_original = []
definitional_pairs = []
with urllib.request.urlopen(definitional_pairs_url) as f:
    definitional_pairs_original.extend(json.load(f))
for [w1, w2] in definitional_pairs_original:
    definitional_pairs.append([w1.lower(), w2.lower()])

print("successfully loaded further_embeddings")