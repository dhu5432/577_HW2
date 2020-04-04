import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as functional
from torch.autograd import Variable
import torch.optim as optim
import string
import argparse
import nltk
from timeit import default_timer as timer
from nltk import ngrams

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

'''Following are some helper functions from https://github.com/lixin4ever/E2E-TBSA/blob/master/utils.py to help parse the Targeted Sentiment Twitter dataset. You are free to delete this code and parse the data yourself, if you wish.

You may also use other parsing functions, but ONLY for parsing and ONLY from that file.
'''
embed_dimension = 300
epochs = 30
LSTM_flag = False

class NeuralNet(nn.Module):

    def __init__(self, LSTM):
        super(NeuralNet, self).__init__()
        LSTM_flag = LSTM
        if LSTM:
            self.lstm = nn.LSTM(embed_dimension+4, 2*embed_dimension+4, bidirectional=True)
            self.fc1 = nn.Linear(embed_dimension+4, 400)
        else:
            self.fc1 = nn.Linear(2*embed_dimension+4, 400)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(400, 200)
        self.act2 = nn.Sigmoid()
        self.fc3 = nn.Linear(200, 4)
        self.act3 = nn.LogSoftmax()

    def forward(self, input_, sentence=None):
        if LSTM_flag:
            input_, _ = self.lstm(sentence)
        a1 = self.fc1(input_)
        h1 = self.act1(a1)
        a2 = self.fc2(h1)
        h2 = self.act2(a2)
        a3 = self.fc3(h2)
        h3 = self.act3(a3)
        return h3


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

def argmax(vec):
    # return the argmax as a python int
    max = vec[0][0]
    max_index = 0
    for i in range(1,4):
        if vec[0][i] > max:
            max = vec[0][i]
            max_index = i
    return max_index
    #_, idx = torch.max(vec, 1)
    #return idx.item()

def main():
    torch.device('cpu')
    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/twitter1_train.txt', help='Train file')
    parser.add_argument('--test_file', type=str, default='data/twitter1_test.txt', help='Test file')
    parser.add_argument('--option', type=int, default=1, help='Option to run (1 = Randomly Initialized, 2 = Word2Vec, 3 = Bi-LSTM')
    args = parser.parse_args()
    print(vars(args)['option'])

    # Option 1
    if vars(args)['option'] == 1:
        train_set = read_data(path=args.train_file)
        test_set = read_data(path=args.test_file)

        #train_set1 = train_set[0:1692]
        #validation_set = train_set[1692:len(train_set)]
        #train_set = train_set1

        words = set()
        for sentence in train_set:
            split = sentence['sentence'].split()
            for word in split:
                words.add(word)
        tags = set()
        for sentence in train_set:
            for tag in sentence['ts_raw_tags']:
                tags.add(tag)

        # Create Word Embeddings
        dictionary_of_words = {}
        dictionary_of_words_reversed = {}
        dictionary_of_word_embeddings = {}
        counter = 0
        for word in words:
            dictionary_of_words[counter] = word
            dictionary_of_words_reversed[word] = counter
            counter += 1
        dictionary_of_words[counter] = ' '
        dictionary_of_words_reversed[' '] = counter
        word_embedding = nn.Embedding(len(dictionary_of_words), embed_dimension)
        for index in dictionary_of_words.keys():
            embedded_word = word_embedding(autograd.Variable(torch.LongTensor([index])))
            dictionary_of_word_embeddings[index] = embedded_word

        # Create Label Embeddings
        dictionary_of_labels = {}
        dictionary_of_labels_index = {}

        dictionary_of_labels_index[0] = 'T-NEU'
        dictionary_of_labels_index[1] = 'T-POS'
        dictionary_of_labels_index[2] = 'T-NEG'
        dictionary_of_labels_index[3] = 'O'

        dictionary_of_label_embeddings ={}
        dictionary_of_labels['T-NEU'] = [1, 0, 0, 0]  # T-NEU
        dictionary_of_labels['T-POS'] = [0, 1, 0, 0]  # T-POS
        dictionary_of_labels['T-NEG'] = [0, 0, 1, 0]  # T-NEG
        dictionary_of_labels['O'] = [0, 0, 0, 1]  # O
        dictionary_of_labels['START'] = [0, 0, 0, 0]  # START

        for key in dictionary_of_labels.keys():
            dictionary_of_label_embeddings[key] = torch.FloatTensor([dictionary_of_labels[key]])

        start = timer()
        # Training
        net = NeuralNet(False)
        opt = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0.1)
        criterion = nn.NLLLoss(torch.Tensor([1.2, 1.2, 1.2, 0.7]))

        for epoch in range(epochs):
            for sentence_num in range(len(train_set)):
                for word_num in range(len(train_set[sentence_num]['sentence'].split())):
                    word = train_set[sentence_num]['sentence'].split()[word_num]
                    word_embed = dictionary_of_word_embeddings[dictionary_of_words_reversed[word]]
                    gold_label_embed = dictionary_of_label_embeddings[train_set[sentence_num]['ts_raw_tags'][word_num]]
                    if word_num == 0:
                        arbitrary_embed = dictionary_of_word_embeddings[dictionary_of_words_reversed[' ']]
                        concat = torch.cat((arbitrary_embed, word_embed, dictionary_of_label_embeddings['START']), 1)

                    else:
                        prev_word = train_set[sentence_num]['sentence'].split()[word_num - 1]
                        prev_word_embed = dictionary_of_word_embeddings[dictionary_of_words_reversed[prev_word]]
                        prev_label_embed = dictionary_of_label_embeddings[train_set[sentence_num]['ts_raw_tags'][word_num-1]]
                        concat = torch.cat((prev_word_embed, word_embed, prev_label_embed), 1)
                    opt.zero_grad()
                    word_embed = autograd.Variable(concat)
                    pred = net(word_embed)
                    print(pred)
                    gold_label_class = torch.max(gold_label_embed, 1)[1]

                    loss = criterion(pred, gold_label_class)
                    loss.backward()
                    opt.step()

        end = timer()
        print(end - start)

        # Viterbi: Testing
        for sentence_num in range(len(test_set)):
            init_vvars = torch.full((1, 5), -10000)
            init_vvars[0][4] = 0
            backpointers = []
            forward_var = init_vvars
            for feat in range(len(test_set[sentence_num]['sentence'].split())):

                word = test_set[sentence_num]['sentence'].split()[feat]
                try:
                    word_embed = dictionary_of_word_embeddings[dictionary_of_words_reversed[word]]
                except KeyError:
                    word_embed = dictionary_of_word_embeddings[dictionary_of_words_reversed[' ']]

                bptrs_t = []
                viterbivars_t = []

                for tagg in range(4):
                    if feat == 0:
                        arbitrary_embed = dictionary_of_word_embeddings[dictionary_of_words_reversed[' ']]
                        concat = torch.cat((arbitrary_embed, word_embed,
                                            dictionary_of_label_embeddings[dictionary_of_labels_index[tagg]]), 1)
                        pred = net(autograd.Variable(concat))

                    else:
                        prev_word = test_set[sentence_num]['sentence'].split()[feat - 1]
                        try:
                            prev_word_embed = dictionary_of_word_embeddings[dictionary_of_words_reversed[prev_word]]
                        except KeyError:
                            prev_word_embed = dictionary_of_word_embeddings[dictionary_of_words_reversed[' ']]
                        prev_label_embed = dictionary_of_label_embeddings[dictionary_of_labels_index[
                            tagg]]  # dictionary_of_label_embeddings[validation_set[sentence_num]['ts_raw_tags'][word_num - 1]]

                        concat = torch.cat((prev_word_embed, word_embed, prev_label_embed), 1)
                        pred = net(autograd.Variable(concat))

                    next_tag_var = forward_var + torch.cat((pred, torch.zeros([1, 2])), 1)

                    best_tag_id = argmax(next_tag_var)

                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

                forward_var = torch.cat((viterbivars_t)).view(1, -1)
                forward_var = torch.cat((forward_var, torch.zeros([1, 2])), 1)
                backpointers.append(bptrs_t)
            terminal_var = forward_var
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            start = best_path.pop()
            best_path.reverse()
            #print(best_path)

            for word_num in range(len(test_set[sentence_num]['ts_raw_tags'])):
                predicted_tag = dictionary_of_labels_index[best_path[word_num]]
                true_tag = test_set[sentence_num]['ts_raw_tags'][word_num]
                if true_tag == 'T-NEU' and predicted_tag == 'T-NEU':
                    true_positive += 1
                elif true_tag == 'T-NEU' and predicted_tag == 'T-POS':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-NEU' and predicted_tag == 'T-NEG':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-NEU' and predicted_tag == 'O':
                    false_negative += 1

                elif true_tag == 'T-POS' and predicted_tag == 'T-NEU':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-POS' and predicted_tag == 'T-POS':
                    true_positive += 1
                elif true_tag == 'T-POS' and predicted_tag == 'T-NEG':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-POS' and predicted_tag == 'O':
                    false_negative += 1

                elif true_tag == 'T-NEG' and predicted_tag == 'T-NEU':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-NEG' and predicted_tag == 'T-POS':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-NEG' and predicted_tag == 'T-NEG':
                    true_positive += 1
                elif true_tag == 'T-NEG' and predicted_tag == 'O':
                    false_negative += 1

                elif true_tag == 'O' and predicted_tag == 'T-NEU':
                    false_positive += 1
                elif true_tag == 'O' and predicted_tag == 'T-POS':
                    false_positive += 1
                elif true_tag == 'O' and predicted_tag == 'T-NEG':
                    false_positive += 1
                elif true_tag == 'O' and predicted_tag == 'O':
                    true_negative += 1
                else:
                    print("SHOULDN'T BE HERE")
        print(true_positive)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = (2 * precision * recall) / (precision + recall)
        print(f1)

    # Option 2
    if vars(args)['option'] == 2:
        train_set = read_data(path=args.train_file)
        test_set = read_data(path=args.test_file)

        #train_set1 = train_set[0:1692]
        #validation_set = train_set[1692:len(train_set)]
        #train_set = train_set1

        words = set()
        for sentence in train_set:
            split = sentence['sentence'].split()
            for word in split:
                words.add(word)
        tags = set()
        for sentence in train_set:
            for tag in sentence['ts_raw_tags']:
                tags.add(tag)

        wv_from_bin = KeyedVectors.load_word2vec_format("/homes/cs577/hw2/w2v.bin", binary=True)

        # Create Label Embeddings
        dictionary_of_labels = {}
        dictionary_of_labels_index = {}

        dictionary_of_labels_index[0] = 'T-NEU'
        dictionary_of_labels_index[1] = 'T-POS'
        dictionary_of_labels_index[2] = 'T-NEG'
        dictionary_of_labels_index[3] = 'O'

        dictionary_of_label_embeddings ={}
        dictionary_of_labels['T-NEU'] = [1, 0, 0, 0]  # T-NEU
        dictionary_of_labels['T-POS'] = [0, 1, 0, 0]  # T-POS
        dictionary_of_labels['T-NEG'] = [0, 0, 1, 0]  # T-NEG
        dictionary_of_labels['O'] = [0, 0, 0, 1]  # O
        dictionary_of_labels['START'] = [0, 0, 0, 0]  # START

        for key in dictionary_of_labels.keys():
            dictionary_of_label_embeddings[key] = torch.FloatTensor([dictionary_of_labels[key]])

        start = timer()
        # Training
        net = NeuralNet(False)
        opt = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999))
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print("Epoch {}".format(epoch+1))
            for sentence_num in range(len(train_set)):
                for word_num in range(len(train_set[sentence_num]['sentence'].split())):
                    word = train_set[sentence_num]['sentence'].split()[word_num]
                    try:
                        word_embed = torch.from_numpy(wv_from_bin[word])
                    except KeyError:
                        word_embed = torch.zeros([1, 300])

                    gold_label_embed = dictionary_of_label_embeddings[train_set[sentence_num]['ts_raw_tags'][word_num]]
                    if word_num == 0:
                        arbitrary_embed = torch.zeros([1, 300])
                        concat = torch.cat((arbitrary_embed, word_embed.view(1, 300), dictionary_of_label_embeddings['START']), 1)

                    else:
                        prev_word = train_set[sentence_num]['sentence'].split()[word_num - 1]
                        try:
                            prev_word_embed = torch.from_numpy(wv_from_bin[prev_word])
                        except KeyError:
                            prev_word_embed = torch.zeros([1, 300])

                        prev_label_embed = dictionary_of_label_embeddings[train_set[sentence_num]['ts_raw_tags'][word_num-1]]
                        concat = torch.cat((prev_word_embed.view(1, 300), word_embed.view(1, 300), prev_label_embed), 1)
                    opt.zero_grad()
                    word_embed = autograd.Variable(concat)
                    pred = net(word_embed)
                    gold_label_class = torch.max(gold_label_embed, 1)[1]

                    loss = criterion(pred, gold_label_class)
                    loss.backward()
                    opt.step()

        end = timer()
        print(end - start)

        # Viterbi: Testing
        for sentence_num in range(len(test_set)):
            init_vvars = torch.full((1, 6), -10000)
            init_vvars[0][4] = 0
            backpointers = []
            forward_var = init_vvars
            for feat in range(len(test_set[sentence_num]['sentence'].split())):

                word = test_set[sentence_num]['sentence'].split()[feat]
                try:
                    word_embed = torch.from_numpy(wv_from_bin[word]).view(1, 300)
                except KeyError:
                    word_embed = torch.zeros([1, 300])

                bptrs_t = []
                viterbivars_t = []

                for tagg in range(4):
                    if feat == 0:
                        arbitrary_embed = torch.zeros([1,300])
                        concat = torch.cat((arbitrary_embed, word_embed,
                                            dictionary_of_label_embeddings[dictionary_of_labels_index[tagg]]), 1)
                        pred = net(autograd.Variable(concat))

                    else:
                        prev_word = test_set[sentence_num]['sentence'].split()[feat - 1]
                        try:
                            prev_word_embed = torch.from_numpy(wv_from_bin[prev_word]).view(1, 300)
                        except KeyError:
                            prev_word_embed = torch.zeros([1, 300])
                        prev_label_embed = dictionary_of_label_embeddings[dictionary_of_labels_index[
                            tagg]]  # dictionary_of_label_embeddings[validation_set[sentence_num]['ts_raw_tags'][word_num - 1]]

                        concat = torch.cat((prev_word_embed, word_embed, prev_label_embed), 1)
                        pred = net(autograd.Variable(concat))

                    next_tag_var = forward_var + torch.cat((pred, torch.zeros([1, 2])), 1)
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

                forward_var = torch.cat((viterbivars_t)).view(1, -1)
                forward_var = torch.cat((forward_var, torch.zeros([1, 2])), 1)
                backpointers.append(bptrs_t)
            terminal_var = forward_var
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            best_path = [best_tag_id]
            #print(best_tag_id)
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            start = best_path.pop()
            best_path.reverse()
            #print(best_path)

            for word_num in range(len(test_set[sentence_num]['ts_raw_tags'])):
                predicted_tag = dictionary_of_labels_index[best_path[word_num]]
                true_tag = test_set[sentence_num]['ts_raw_tags'][word_num]
                if true_tag == 'T-NEU' and predicted_tag == 'T-NEU':
                    true_positive += 1
                elif true_tag == 'T-NEU' and predicted_tag == 'T-POS':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-NEU' and predicted_tag == 'T-NEG':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-NEU' and predicted_tag == 'O':
                    false_negative += 1

                elif true_tag == 'T-POS' and predicted_tag == 'T-NEU':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-POS' and predicted_tag == 'T-POS':
                    true_positive += 1
                elif true_tag == 'T-POS' and predicted_tag == 'T-NEG':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-POS' and predicted_tag == 'O':
                    false_negative += 1

                elif true_tag == 'T-NEG' and predicted_tag == 'T-NEU':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-NEG' and predicted_tag == 'T-POS':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-NEG' and predicted_tag == 'T-NEG':
                    true_positive += 1
                elif true_tag == 'T-NEG' and predicted_tag == 'O':
                    false_negative += 1

                elif true_tag == 'O' and predicted_tag == 'T-NEU':
                    false_positive += 1
                elif true_tag == 'O' and predicted_tag == 'T-POS':
                    false_positive += 1
                elif true_tag == 'O' and predicted_tag == 'T-NEG':
                    false_positive += 1
                elif true_tag == 'O' and predicted_tag == 'O':
                    true_negative += 1
                else:
                    print(true_tag)
                    print(predicted_tag)
                    print("SHOULDN'T BE HERE")
        print(true_positive)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = (2 * precision * recall) / (precision + recall)
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))

    if vars(args)['option'] == 3:
        train_set = read_data(path=args.train_file)
        test_set = read_data(path=args.test_file)

        #train_set1 = train_set[0:1692]
        #validation_set = train_set[1692:len(train_set)]
        #train_set = train_set1

        words = set()
        for sentence in train_set:
            split = sentence['sentence'].split()
            for word in split:
                words.add(word)
        tags = set()
        for sentence in train_set:
            for tag in sentence['ts_raw_tags']:
                tags.add(tag)

        wv_from_bin = KeyedVectors.load_word2vec_format("/homes/cs577/hw2/w2v.bin", binary=True)

        # Create Label Embeddings
        dictionary_of_labels = {}
        dictionary_of_labels_index = {}

        dictionary_of_labels_index[0] = 'T-NEU'
        dictionary_of_labels_index[1] = 'T-POS'
        dictionary_of_labels_index[2] = 'T-NEG'
        dictionary_of_labels_index[3] = 'O'

        dictionary_of_label_embeddings ={}
        dictionary_of_labels['T-NEU'] = [1, 0, 0, 0]  # T-NEU
        dictionary_of_labels['T-POS'] = [0, 1, 0, 0]  # T-POS
        dictionary_of_labels['T-NEG'] = [0, 0, 1, 0]  # T-NEG
        dictionary_of_labels['O'] = [0, 0, 0, 1]  # O
        dictionary_of_labels['START'] = [0, 0, 0, 0]  # START

        for key in dictionary_of_labels.keys():
            dictionary_of_label_embeddings[key] = torch.FloatTensor([dictionary_of_labels[key]])

        start = timer()
        # Training
        net = NeuralNet(True)
        opt = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999))
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print("Epoch {}".format(epoch+1))
            for sentence_num in range(len(train_set)):
                for word_num in range(len(train_set[sentence_num]['sentence'].split())):
                    word = train_set[sentence_num]['sentence'].split()[word_num]
                    try:
                        word_embed = torch.from_numpy(wv_from_bin[word])
                    except KeyError:
                        word_embed = torch.zeros([1, 300])

                    gold_label_embed = dictionary_of_label_embeddings[train_set[sentence_num]['ts_raw_tags'][word_num]]
                    if word_num == 0:
                        arbitrary_embed = torch.zeros([1, 300])
                        concat = torch.cat((word_embed.view(1, 300), dictionary_of_label_embeddings['START']), 1)

                    else:
                        prev_label_embed = dictionary_of_label_embeddings[train_set[sentence_num]['ts_raw_tags'][word_num-1]]
                        concat = torch.cat((word_embed.view(1, 300), prev_label_embed), 1)
                    opt.zero_grad()
                    word_embed = autograd.Variable(concat)
                    pred = net(word_embed)
                    gold_label_class = torch.max(gold_label_embed, 1)[1]

                    loss = criterion(pred, gold_label_class)
                    loss.backward()
                    opt.step()

        end = timer()
        print(end - start)

        # Viterbi: Testing
        for sentence_num in range(len(test_set)):
            init_vvars = torch.full((1, 6), -10000)
            init_vvars[0][4] = 0
            backpointers = []
            forward_var = init_vvars
            for feat in range(len(test_set[sentence_num]['sentence'].split())):

                word = test_set[sentence_num]['sentence'].split()[feat]
                try:
                    word_embed = torch.from_numpy(wv_from_bin[word]).view(1, 300)
                except KeyError:
                    word_embed = torch.zeros([1, 300])

                bptrs_t = []
                viterbivars_t = []

                for tagg in range(4):
                    if feat == 0:
                        arbitrary_embed = torch.zeros([1,300])
                        concat = torch.cat((word_embed,
                                            dictionary_of_label_embeddings[dictionary_of_labels_index[tagg]]), 1)
                        pred = net(autograd.Variable(concat))

                    else:
                        prev_word = test_set[sentence_num]['sentence'].split()[feat - 1]
                        try:
                            prev_word_embed = torch.from_numpy(wv_from_bin[prev_word]).view(1, 300)
                        except KeyError:
                            prev_word_embed = torch.zeros([1, 300])
                        prev_label_embed = dictionary_of_label_embeddings[dictionary_of_labels_index[
                            tagg]]  # dictionary_of_label_embeddings[validation_set[sentence_num]['ts_raw_tags'][word_num - 1]]

                        concat = torch.cat((word_embed, prev_label_embed), 1)
                        pred = net(autograd.Variable(concat))

                    next_tag_var = forward_var + torch.cat((pred, torch.zeros([1, 2])), 1)
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

                forward_var = torch.cat((viterbivars_t)).view(1, -1)
                forward_var = torch.cat((forward_var, torch.zeros([1, 2])), 1)
                backpointers.append(bptrs_t)
            terminal_var = forward_var
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            best_path = [best_tag_id]
            #print(best_tag_id)
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            start = best_path.pop()
            best_path.reverse()
            #print(best_path)

            for word_num in range(len(test_set[sentence_num]['ts_raw_tags'])):
                predicted_tag = dictionary_of_labels_index[best_path[word_num]]
                true_tag = test_set[sentence_num]['ts_raw_tags'][word_num]
                if true_tag == 'T-NEU' and predicted_tag == 'T-NEU':
                    true_positive += 1
                elif true_tag == 'T-NEU' and predicted_tag == 'T-POS':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-NEU' and predicted_tag == 'T-NEG':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-NEU' and predicted_tag == 'O':
                    false_negative += 1

                elif true_tag == 'T-POS' and predicted_tag == 'T-NEU':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-POS' and predicted_tag == 'T-POS':
                    true_positive += 1
                elif true_tag == 'T-POS' and predicted_tag == 'T-NEG':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-POS' and predicted_tag == 'O':
                    false_negative += 1

                elif true_tag == 'T-NEG' and predicted_tag == 'T-NEU':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-NEG' and predicted_tag == 'T-POS':
                    false_negative += 1
                    false_positive += 1
                elif true_tag == 'T-NEG' and predicted_tag == 'T-NEG':
                    true_positive += 1
                elif true_tag == 'T-NEG' and predicted_tag == 'O':
                    false_negative += 1

                elif true_tag == 'O' and predicted_tag == 'T-NEU':
                    false_positive += 1
                elif true_tag == 'O' and predicted_tag == 'T-POS':
                    false_positive += 1
                elif true_tag == 'O' and predicted_tag == 'T-NEG':
                    false_positive += 1
                elif true_tag == 'O' and predicted_tag == 'O':
                    true_negative += 1
                else:
                    print(true_tag)
                    print(predicted_tag)
                    print("SHOULDN'T BE HERE")
        print(true_positive)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = (2 * precision * recall) / (precision + recall)
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
        print("F1: {}".format(f1))







    # uncomment if you want to see the data format
    #print(train_set[0]['sentence'])

    # now, you must parse the dataset




    # example to load the Word2Vec model
    # note: this will only work on a cs machine (ex: data.cs.purdue.edu)
    #wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/homes/cs577/hw2/w2v.bin"), binary=True)


    #for entry in train_set:
        #print(entry)



if __name__ == '__main__':
        main()




