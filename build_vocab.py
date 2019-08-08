import os
import re
import sentencepiece as spm
from collections import Counter

re_sc = re.compile('[@#$%\^&\*\(\)\-\=\[\]\{\}\.,/~\+\'"|_:><`┃]')

def train_spm(txt_path='train/train_data',
              spm_path='./spm',
              vocab_size=10000, input_sentence_size=11000000):
    spm_dir = os.path.dirname(spm_path)
    os.makedirs(spm_dir, exist_ok=True)
    spm.SentencePieceTrainer.Train(
        f' --input={txt_path} --model_type=bpe'
        f' --model_prefix={spm_path} --vocab_size={vocab_size}'
        f' --input_sentence_size={input_sentence_size}'
        )

stopwords = list(['!@#$%^&*()_+=,./'])
def text_cleaning(x, stopwords):
    for sw in stopwords:
        x = x.replace(sw,' ')
    x = x.lower()
    return x


def build_vocab(mode='train', dataset_path='.', vocab_size=10000):
    if mode != 'train':
        print('only [train] is supported')
        return

    data_review = os.path.join(dataset_path, 'train', 'train_data')

    print('writing preprocessed titles ...')
    reviews = []
    with open(data_review, 'rt') as f:
        for line in f:
            reviews.append(' '.join(re_sc.sub(' ', line).strip().split()))

    write_titles(reviews, './reviews.txt')

    print('training spm ...')
    train_spm('./reviews.txt', vocab_size=vocab_size)
    print('build wp vocab ...')

    wp_vocab = build_wp_vocab(reviews)

    return reviews, wp_vocab


def build_wp_vocab(reviews, spm_model_path='./spm.model'):
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model_path)

    wp_counter = Counter()

    max_wps_len = 0
    for _, review in enumerate(reviews):
        words = review.split()
        wps = []
        for w in words:
            wp = sp.EncodeAsPieces(w)
            max_wps_len = max(len(wp), max_wps_len)
            wps += wp

        for wp in wps:
            wp_counter[wp] += 1

    wp_vocab = [('PAD', max_wps_len)] + wp_counter.most_common()
    write_vocab(wp_vocab, './wp_vocab.txt')

    #print(wp_vocab[:100])

    return wp_vocab

def write_vocab(vocab, vocab_fn):
    with open(vocab_fn, 'w') as fp:
        for v, c in vocab:
            fp.write(f'{v}\t{c}\n')


def write_titles(titles, titles_path):
    f_titles = open(titles_path, 'w')

    for title in titles:
        f_titles.write(title + '\n')
