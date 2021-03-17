import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from six import iteritems

class analogy_tasks():

    def evaluate_analogy_msr(W, vocab, file_name='word_relationship.questions'):
        """Evaluate the trained word vectors on a variety of tasks"""

        #prefix = '/zf15/tw8cb/summer_2019/code/GloVe/eval/question-data/'

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

        with open('%s/' % (file_name), 'r') as f:
            full_data = []
            for line in f:
                tokens = line.rstrip().split(' ')
                full_data.append([tokens[0], tokens[1], tokens[2]]) #, tokens[4]])
            full_count += len(full_data)
            data = [x for x in full_data if all(word in vocab for word in x)]

        indices = np.array([[embed(word) for word in row] for row in data])
        ind1, ind2, ind3, ind4 = indices.T

        predictions = np.zeros((len(indices),))
        num_iter = int(np.ceil(len(indices) / float(split_size)))
        for j in range(num_iter):
            subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

            pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
                +  W[ind3[subset], :])
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

        #print("%s:" % filenames[i])
        print(len(val))
        print('ACCURACY TOP1-MSR: %.2f%% (%d/%d)' %
            (np.mean(val) * 100, np.sum(val), len(val)))
        
    
    def evaluate_analogy_google(W, vocab, w2id):
        """Evaluate the trained w vectors on a variety of tasks"""

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
                #print("full_data", full_data)
                #print(len(full_data))
                full_count += len(full_data)
                data = [x for x in full_data if all(word.lower() in vocab for word in x)]
            
            #print(len(data))
            #print("data", data)
            
            indices = np.array([[w2id[word.lower()] for word in row] for row in data])
            #print(indices.shape)
            ind1, ind2, ind3, ind4 = indices.T

            predictions = np.zeros((len(indices),))
            num_iter = int(np.ceil(len(indices) / float(split_size)))
            for j in range(num_iter):
                subset = np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))

                pred_vec = (W[ind2[subset], :] - W[ind1[subset], :]
                    +  W[ind3[subset], :])
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
        
        
    def evaluate_analogy_semeval2012(w_dict):
        score = evaluate_on_semeval_2012_2(w_dict)['all']
        print("Analogy prediction accuracy on {} {}".format("SemEval2012", score))

        
    def evaluate_ana(wv, w2i, vocab):
        W_norm = np.zeros(wv.shape)
        d = (np.sum(wv ** 2, 1) ** (0.5))
        W_norm = (wv.T / d).T

        evaluate_analogy_msr(W_norm, w2i)
        evaluate_analogy_google(W_norm, w2i)

        wv_dict = dict()
        for w in vocab:
            wv_dict[w] = W_norm[w2i[w], :]

        if isinstance(wv_dict, dict):
            w = Embedding.from_dict(wv_dict)
        evaluate_analogy_semeval2012(w)

    #     analogy_tasks = {
    #         "Google": fetch_google_analogy(),
    #         "MSR": fetch_msr_analogy()
    #     }

    #     analogy_results = {}

    #     for name, data in iteritems(analogy_tasks):
    #         analogy_results[name] = evaluate_analogy(w, data.X, data.y)
    #         print("Analogy prediction accuracy on {} {}".format(name, analogy_results[name]))