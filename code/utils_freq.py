import urllib.request
import wget
import zipfile
import os
import json

# load definitional pairs
definitional_pairs_url = "https://raw.githubusercontent.com/uvavision/Double-Hard-Debias/master/data/definitional_pairs.json"

definitional_pairs = []
with urllib.request.urlopen(definitional_pairs_url) as f:
    definitional_pairs.extend(json.load(f))

definitional_labels = []

for i, [w1, w2] in enumerate(definitional_pairs):
    definitional_pairs[i] = [w1.lower(), w2.lower()]
    definitional_labels.append(w1 + "&" + w2)

def download_txt(url_dataset):
    """
    downloads text from given dataset and returns list of consecutive lower case words

    takes
    url_dataset = the url of the text, here specific to download of 'text8.zip'

    returns
    word_list = text in form of lower case word list
    """

    wget.download(url_dataset)

    with zipfile.ZipFile('text8.zip', 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())

    with zipfile.ZipFile('text8.zip', 'r') as zip_ref:
            # only take limited number of characters, loose 'b prefix
            txt = (zip_ref.read('text8')[:1000000].decode('utf-8'))

    txt = txt.split()
    word_list = []
    for word in txt:
        word_list.append(word.lower())

    return word_list


def create_bag_of_words(data, number_words, stride):
    """
    splits given data into multiple bag of words

    takes
    data = original list of words
    number_words = number of words used in the sequence
    stride

    returns
    bags = list conftaining bag of words
    """

    # number of possible "convolutions" over data
    count = round((len(data)-number_words)/stride)

    bags = []
    j = 0
    for i in range(count):
        bags.append(list(data[j:(j+number_words)]))
        j = j+ stride

    return bags

def fake_frequency(data, chosen_word):
    """samples sentences in data containing the chosen word multiple times

    takes
    data = list of sentences
    chosen_word = word of which to increase the frequency

    returns
    data_freq = altered dataset
    """

    data_freq = data.copy()
    for sentence in data:
        if chosen_word in sentence:
            data_freq.append(sentence)

    return data_freq

print("successfully loaded utils for frequency investigation")
