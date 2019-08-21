from collections import defaultdict
import time
import random
import dynet as dy
import numpy as np
import os
import sys

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

def read_dataset(filename):
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield([w2i[x] for x in words.split(" ")], t2i[tag])

train = list(read_dataset(os.path.join(sys.path[0], "topicclass/topicclass_train.txt")))
w2i = defaultdict(lambda: UNK, w2i)

dev = list(read_dataset(os.path.join(sys.path[0], "topicclass/topicclass_valid.txt")))

test = list(read_dataset(os.path.join(sys.path[0], "topicclass/topicclass_test.txt")))

nwords = len(w2i)
ntags = len(t2i)

model = dy.ParameterCollection()
trainer = dy.AdamTrainer(model)

EMB_SIZE = 64
w_emb = model.add_lookup_parameters((nwords, 1, 1, EMB_SIZE))
WIN_SIZE = 3
FILTER_SIZE = 64

w_cnn = model.add_parameters((1, WIN_SIZE, EMB_SIZE, FILTER_SIZE))
b_cnn = model.add_parameters((FILTER_SIZE))

w_sm = model.add_parameters((ntags, FILTER_SIZE))
b_sm = model.add_parameters((ntags))

def calc_scores(words):
    dy.renew_cg()
    if len(words) < WIN_SIZE:
        words += [0] * (WIN_SIZE - len(words))

    cnn_in = dy.concatenate([dy.lookup(w_emb, x) for x in words], d=1)

    stride = (1,1)
    cnn_out = dy.conv2d_bias(cnn_in, w_cnn, b_cnn, stride=stride, is_valid=False)
    pool_out = dy.max_dim(cnn_out, d=1)
    pool_out = dy.reshape(pool_out, (FILTER_SIZE,))
    pool_out = dy.rectify(pool_out)
    return w_sm * pool_out + b_sm

print('Beginning training')
for epoch in range(100):
    random.shuffle(train)
    train_correct = 0.0
    train_loss = 0.0
    start = time.time()
    start_time = time.time()
    count = 0
    for words, tag in train:
        scores = calc_scores(words)
        predict = np.argmax(scores.npvalue())
        if predict == tag:
            train_correct +=1

        loss = dy.pickneglogsoftmax(scores, tag)
        train_loss += loss.value()
        count += 1
        loss.backward()
        trainer.update()
        if count % 10000 == 0:
            print("Training on " + repr(count) + " of " + repr(len(train)) + " sentences in epoch " + repr(
                epoch) + " took %.4f minutes. Loss so far is %.4f." % ((time.time() - start_time) / 60, train_loss/count))
            start_time = time.time()

    print("Epoch %r: train loss per sentence = %.4f; train accuracy = %.4f; time = %.2fm" % (epoch, train_loss/len(train), train_correct/len(train), (time.time()-start)/60))

    dev_correct = 0.0
    dev_start_time = time.time()
    predictions = []
    for words, tag in dev:
        scores = calc_scores(words).npvalue()
        predict = np.argmax(scores)
        predictions.append(predict)
        if predict == tag:
            dev_correct += 1
    print("Dev accuracy = %.4f, time = %.2fs" % (dev_correct/len(dev), time.time() - dev_start_time))
    out_file = open("dev_output_" + repr(epoch) + ".txt", "a+", encoding="utf8")
    out_file.write("Accuracy = " + repr(dev_correct/len(dev)))

    tags = list(t2i.keys())
    vocab = list(w2i.keys())

    ln = 0
    for words, tag in dev:
        sent = ""
        for idx in words:
            sent += vocab[idx] + " "
        out = tags[predictions[ln]] + " ||| " + sent + "\n"
        out_file.write(out)
        ln += 1
    out_file.close()

    predictions = []
    for words, tag in test:
        scores = calc_scores(words).npvalue()
        predict = np.argmax(scores)
        predictions.append(predict)

    out_file = open("output_" + repr(epoch) + ".txt", "a+", encoding="utf8")
    for line_num, line in enumerate(test):
        sent_idxs, tag = line
        sent = ""
        for idx in sent_idxs:
            sent += vocab[idx] + " "
        out = tags[predictions[line_num]] + " ||| " + sent + "\n"
        out_file.write(out)
    out_file.close()