import numpy as np
import string
import argparse

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath


'''Following are some helper functions from https://github.com/lixin4ever/E2E-TBSA/blob/master/utils.py to help parse the Targeted Sentiment Twitter dataset. You are free to delete this code and parse the data yourself, if you wish.

You may also use other parsing functions, but ONLY for parsing and ONLY from that file.
'''
def read_data(path):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
    """
    dataset = []
    with open(path, encoding='UTF-8') as fp:
        for line in fp:
            record = {}
            sent, tag_string = line.strip().split('####')
            record['sentence'] = sent
            word_tag_pairs = tag_string.split(' ')
            # tag sequence for targeted sentiment
            ts_tags = []
            # word sequence
            words = []
            for item in word_tag_pairs:
                # valid label is: O, T-POS, T-NEG, T-NEU
                eles = item.split('=')
                if len(eles) == 2:
                    word, tag = eles
                elif len(eles) > 2:
                    tag = eles[-1]
                    word = (len(eles) - 2) * "="
                if word not in string.punctuation:
                    # lowercase the words
                    words.append(word.lower())
                else:
                    # replace punctuations with a special token
                    words.append('PUNCT')
                if tag == 'O':
                    ts_tags.append('O')
                elif tag == 'T-POS':
                    ts_tags.append('T-POS')
                elif tag == 'T-NEG':
                    ts_tags.append('T-NEG')
                elif tag == 'T-NEU':
                    ts_tags.append('T-NEU')
                else:
                    raise Exception('Invalid tag %s!!!' % tag)
            record['words'] = words.copy()
            record['ts_raw_tags'] = ts_tags.copy()
            dataset.append(record)
    print("Obtain %s records from %s" % (len(dataset), path))
    return dataset

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/twitter1_train.txt', help='Train file')
    parser.add_argument('--test_file', type=str, default='data/twitter1_test.txt', help='Test file')
    parser.add_argument('--option', type=int, default=1, help='Option to run (1 = Randomly Initialized, 2 = Word2Vec, 3 = Bi-LSTM')
    args = parser.parse_args()

    # read the dataset
    train_set = read_data(path=args.train_file)
    test_set = read_data(path=args.test_file)
    # uncomment if you want to see the data format
    # print(train_set[0])
    
    # now, you must parse the dataset


    # example to load the Word2Vec model
    # note: this will only work on a cs machine (ex: data.cs.purdue.edu)
    wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/homes/cs577/hw2/w2v.bin"), binary=True)
    # you can get the vector for a word like so
    vector = wv_from_bin['man']
    # print(vector.shape)


    print("")
    print("Precision: ")
    print("Recall: ")
    print("F1: ")

if __name__ == '__main__':
    main()   




