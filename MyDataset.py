from collections import defaultdict
from torch.utils import data

import numpy as np

class MyDataset(data.Dataset):
    def __init__(self, train, batch_size):
        buckets = defaultdict(list)
        for sent, label in train:
            buckets[len(sent)].append((sent,label))

        self.batched_data = []

        lengths = list(buckets.keys())
        np.random.shuffle(lengths)
        for sent_len in lengths:
            sents = buckets[sent_len]
            np.random.shuffle(sents)
            batch_num = int(np.ceil(len(sents) / float(batch_size)))
            for i in range(batch_num):
                cur_batch_size = batch_size if i < batch_num -1 else len(sents) - batch_size*i
                batch_sents = [sents[i * batch_size + b][0] for b in range(cur_batch_size)]
                batch_labels = [sents[i * batch_size + b][1] for b in range(cur_batch_size)]

                self.batched_data.append((batch_sents, batch_labels))

        # del(self.batched_data[1302])
        for index, datum in enumerate(self.batched_data):
            if len(datum[1]) == 1:
                del(self.batched_data[index])
        print('We have ' + repr(len(self.batched_data)) + ' batches.')

    def __len__(self):
        return len(self.batched_data)

    def __getitem__(self, item):
        return self.batched_data[item]