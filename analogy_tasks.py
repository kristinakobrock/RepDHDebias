import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from six import iteritems


def evaluate_analogy_msr(W, vocab, w2id):
    """Evaluate the trained word vectors on a variety of tasks"""

    prefix = 'ana_files'

    file_questions = 'word_relationship.questions'
    file_answers = 'word_relationship.answers'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0; # count correct semantic questions
    correct_syn = 0; # count correct syntactic questions
    correct_tot = 0 # count correct questions
    count_sem = 0; # count all semantic questions
    count_syn = 0; # count all syntactic questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words

    # this part needed to be adjusted because the files openly available on the Microsoft website
    # did not match the format of the files used in the paper
    with open('%s/%s' % (prefix, file_questions), 'r') as file_q, open('%s/%s' % (prefix, file_answers), 'r') as file_a:
        full_data = []
        for line_q, line_a in zip(file_q, file_a):
            tokens_q = line_q.rstrip().split(' ')
            tokens_a = line_a.rstrip().split(' ')
            full_data.append([tokens_q[0], tokens_q[1], tokens_q[2], tokens_a[1]])
        full_count += len(full_data)
        data = [x for x in full_data if all(word in vocab for word in x)]

    indices = np.array([[w2id[word] for word in row] for row in data])
    ind1, ind2, ind3, ind4 = indices.T

    predictions = np.zeros((len(indices),))
    num_iter = int(np.ceil(len(indices) / float(split_size)))
    for j in range(num_iter):
        subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

        pred_vec = (W[ind2[subset], :] - W[ind1[subset], :] +  W[ind3[subset], :])
            
        #cosine similarity if input W has been normalized
        dist = np.dot(W, pred_vec.T)

        for k in range(len(subset)):
            dist[ind1[subset[k]], k] = -np.Inf
            dist[ind2[subset[k]], k] = -np.Inf
            dist[ind3[subset[k]], k] = -np.Inf

        # predicted word index
        predictions[subset] = np.argmax(dist, 0).flatten()

    val = (ind4 == predictions) # correct predictions
    count_tot = count_tot + len(ind1)
    correct_tot = correct_tot + sum(val)

    print(len(val))
    print('ACCURACY TOP1-MSR: %.2f%% (%d/%d)' %
        (np.mean(val) * 100, np.sum(val), len(val)))
        
    
def evaluate_analogy_google(W, vocab, w2id):
    """Evaluate the trained w vectors on a variety of tasks"""

    # the files were adjusted as the only file openly available linked in the original paper 
    # was a single file instead of several
    filenames = [
        'capital-common-countries.txt', 'capital-world.txt', 'currency.txt',
        'city-in-state.txt', 'family.txt', 'gram1-adjective-to-adverb.txt',
        'gram2-opposite.txt', 'gram3-comparative.txt', 'gram4-superlative.txt',
        'gram5-present-participle.txt', 'gram6-nationality-adjective.txt',
        'gram7-past-tense.txt', 'gram8-plural.txt', 'gram9-plural-verbs.txt',
        ]
    prefix = 'ana_files'

    # to avoid memory overflow, could be increased/decreased
    # depending on system and vocab size
    split_size = 100

    correct_sem = 0; # count correct semantic questions
    correct_syn = 0; # count correct syntactic questions
    correct_tot = 0 # count correct questions
    count_sem = 0; # count all semantic questions
    count_syn = 0; # count all syntactic questions
    count_tot = 0 # count all questions
    full_count = 0 # count all questions, including those with unknown words

    for i in range(len(filenames)):
        with open('%s/%s' % (prefix, filenames[i]), 'r') as f:
            full_data = [line.rstrip().split(' ') for line in f]
            full_count += len(full_data)
            data = [x for x in full_data if all(word.lower() in vocab for word in x)]
            
        indices = np.array([[w2id[word.lower()] for word in row] for row in data])
        ind1, ind2, ind3, ind4 = indices.T

        predictions = np.zeros((len(indices),))
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        for j in range(num_iter):
            subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

            pred_vec = (W[ind2[subset], :] - W[ind1[subset], :] +  W[ind3[subset], :])
                
            #cosine similarity if input W has been normalized
            dist = np.dot(W, pred_vec.T)

            for k in range(len(subset)):
                dist[ind1[subset[k]], k] = -np.Inf
                dist[ind2[subset[k]], k] = -np.Inf
                dist[ind3[subset[k]], k] = -np.Inf

            # predicted word index
            predictions[subset] = np.argmax(dist, 0).flatten()

        val = (ind4 == predictions) # correct predictions
        count_tot = count_tot + len(ind1)
        correct_tot = correct_tot + sum(val)
        if i < 5:
            count_sem = count_sem + len(ind1)
            correct_sem = correct_sem + sum(val)
        else:
            count_syn = count_syn + len(ind1)
            correct_syn = correct_syn + sum(val)

        print("%s:" % filenames[i])
        print('ACCURACY TOP1: %.2f%% (%d/%d)' %
            (np.mean(val) * 100, np.sum(val), len(val)))

    print('Questions seen/total: %.2f%% (%d/%d)' %
        (100 * count_tot / float(full_count), count_tot, full_count))
    print('Semantic accuracy: %.2f%%  (%i/%i)' %
        (100 * correct_sem / float(count_sem), correct_sem, count_sem))
    print('Syntactic accuracy: %.2f%%  (%i/%i)' %
        (100 * correct_syn / float(count_syn), correct_syn, count_syn))
    print('Total accuracy: %.2f%%  (%i/%i)' % (100 * correct_tot / float(count_tot), correct_tot, count_tot))
        
        
print("successfully loaded analogy_tasks")