from collections import Counter
import operator

with open('data/text8/train.txt.raw', 'rb') as f:
    text = f.readline()

vocab = dict(Counter([t for t in text]))

vocab_sorted = dict(sorted(vocab.items(), key=operator.itemgetter(1)))

with open('data/text8/vocab_byte', 'wt') as f:
    for k, v in vocab.items():
        f.write('{} {}\n'.format(k, v))
