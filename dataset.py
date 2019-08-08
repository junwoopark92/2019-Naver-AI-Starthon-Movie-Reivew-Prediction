# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from collections import Counter
from kor_char_parser import decompose_str_as_one_hot

re_sc = re.compile('[@#$%\^&\*\(\)\-\=\[\]\{\}\.,/~\+\'"|_:><`┃]')


class MovieReviewDataset(Dataset):
    """
    영화리뷰 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """
    def __init__(self, dataset_path: str, max_length: int, max_word_len, max_wp_len, n_class):
        """
        initializer

        :param dataset_path: 데이터셋 root path
        :param max_length: 문자열의 최대 길이
        """
        self.max_word_len = max_word_len
        self.max_wp_len = max_wp_len

        self.spm_model_path = './spm.model'
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.spm_model_path)
        self.i2wp = [line.split('\t')[0] for line in open('./wp_vocab.txt')]
        self.wp2i = dict([(v, i) for i, v in enumerate(self.i2wp)])

        # 데이터, 레이블 각각의 경로
        data_review = os.path.join(dataset_path, 'train', 'train_data')
        data_label = os.path.join(dataset_path, 'train', 'train_label')

        print('movie review loading...')

        # 영화리뷰 데이터를 읽고 preprocess까지 진행합니다
        with open(data_review, 'rt', encoding='utf-8') as f:
            self.reviews = [line.strip() for line in f]
            self.char_reviews = preprocess(self.reviews, max_length)
        # 영화리뷰 레이블을 읽고 preprocess까지 진행합니다.
        with open(data_label) as f:
            self.labels = [np.float32(x) for x in f.readlines()]
            # self.labels = np.zeros(shape=(len(self.reviews), n_class-1), dtype=np.int32)
            # for i, x in enumerate(f.readlines()):
            #     self.labels[i, :int(x)] = 1

        print('reviews :', len(self.reviews))
        print('labels :', len(self.labels))
        #(self.labels)

    def get_x_text(self, review):
        words = re_sc.sub(' ', review).strip().split()
        words = words[:self.max_word_len]
        text_x_idx = torch.LongTensor(self.max_word_len *
                                      self.max_wp_len).zero_()
        text_x_len = torch.LongTensor(self.max_word_len).zero_()
        text_x_idx_split = torch.split(text_x_idx, self.max_wp_len)

        for i, word in enumerate(words):
            wps = self.sp.EncodeAsPieces(word)
            wps = wps[:self.max_wp_len]
            wp_indices = [self.wp2i[wp] for wp in wps if wp in self.wp2i]
            for j, wp_idx in enumerate(wp_indices):
                text_x_idx_split[i][j] = wp_idx
            text_x_len[i] = len(wp_indices)

        return review, text_x_idx, text_x_len

    def __len__(self):
        """

        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.reviews)

    def __getitem__(self, idx):
        """

        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.char_reviews[idx], self.get_x_text(self.reviews[idx]), self.labels[idx]

    # def __getitem__(self, idx):
    #     """
    #
    #     :param idx: 필요한 데이터의 인덱스
    #     :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
    #     """
    #     return self.reviews[idx], self.labels[idx]


def preprocess(data: list, max_length: int):
    """
     입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
     기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
     문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """
    vectorized_data = [decompose_str_as_one_hot(datum, warning=False) for datum in data]
    zero_padding = np.zeros((len(data), max_length), dtype=np.int32)
    for idx, seq in enumerate(vectorized_data):
        length = len(seq)
        if length >= max_length:
            length = max_length
            zero_padding[idx, :length] = np.array(seq)[:length]
        else:
            zero_padding[idx, :length] = np.array(seq)
    return zero_padding


def collate_fn(data: list):
    """
    PyTorch DataLoader에서 사용하는 collate_fn 입니다.
    기본 collate_fn가 리스트를 flatten하기 때문에 벡터 입력에 대해서 사용이 불가능해, 직접 작성합니다.

    :param data: 데이터 리스트
    :return:
    """
    review = []
    label = []
    for datum in data:
        review.append(datum[0])
        label.append(datum[1])
    # 각각 데이터, 레이블을 리턴
    return review, np.array(label)
