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

import argparse
import os
import re
import numpy as np
import torch
import pickle

from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import nsml
from dataset import MovieReviewDataset, preprocess, collate_fn
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML

from models import Regression, Classification, LSTMAttention, CNN_Text, ImgText2Vec
from build_vocab import build_vocab, re_sc, write_titles, train_spm
import sentencepiece as spm
import torch.nn.functional as F

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard

def bind_model(model, sp, wp_vocab, preprocess_infer, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dirname, *args):
        checkpoint = {
            'model': model.state_dict(),
        }
        torch.save(checkpoint, os.path.join(dirname, 'model.pt'))

        # save sp
        with open(os.path.join(dirname, 'sp_data.pkl'), 'wb') as f:
            pickle.dump(sp, f, pickle.HIGHEST_PROTOCOL)

        # save file
        with open(os.path.join(dirname, 'wp_vocab.pkl'), 'wb') as f:
            pickle.dump(wp_vocab, f, pickle.HIGHEST_PROTOCOL)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dirname, *args):
        checkpoint = torch.load(os.path.join(dirname, 'model.pt'))
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

        with open(os.path.join(dirname, 'sp_data.pkl'), 'rb') as f:
            sp.extend(pickle.load(f))

        with open(os.path.join(dirname, 'wp_vocab.pkl'), 'rb') as f:
            wp_vocab.extend(pickle.load(f))

        print('in_load', type(sp), type(wp_vocab))
        print('in_load', len(sp),len(wp_vocab))

        write_titles(sp, './reviews.txt')
        train_spm('./reviews.txt')
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load('./spm.model')

        i2wp = [line[0] for line in wp_vocab]
        wp2i = dict([(v, i) for i, v in enumerate(i2wp)])

        print('in_infer', type(sp_model), len(wp2i))

        preprocess_infer['wp2i'] = wp2i
        preprocess_infer['sp_model'] = sp_model

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        sp_model = preprocess_infer['sp_model']
        wp2i = preprocess_infer['wp2i']

        char_text = preprocess(raw_data, config.strmaxlen)
        char_text = torch.tensor(char_text)
        char_text = Variable(char_text.long()).cuda()

        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        inter_x_text = infer_preprocess(raw_data, sp_model, wp2i, config.max_words_len, config.max_wp_len)
        inter_x_text = (inter_x_text[0].cuda(), inter_x_text[1].cuda())

        #preprocessed_data = preprocess(raw_data, config.strmaxlen)
        model.eval()
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model(char_text, inter_x_text)
        point = output_prediction.data.squeeze(dim=1)#.tolist()
        print(config.model)
        if config.model in ['classification', 'cnntext', 'bilstmwithattn', 'ImgText2Vec']:
            #point = [np.argmax(p) for p in point]
            point = torch.sum(point > 0.5, dim=1).tolist()

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def infer_preprocess(raw_data, sp_model, wp2i, max_word_len, max_wp_len):
    text_x_idxs = []
    text_x_lens = []
    for d in raw_data:
        words = re_sc.sub(' ', d).strip().split()
        words = words[:max_word_len]
        text_x_idx = torch.LongTensor(max_word_len *
                                      max_wp_len).zero_()
        text_x_len = torch.LongTensor(max_word_len).zero_()
        text_x_idx_split = torch.split(text_x_idx, max_wp_len)

        for i, word in enumerate(words):
            wps = sp_model.EncodeAsPieces(word)
            wps = wps[:max_wp_len]
            wp_indices = [wp2i[wp] for wp in wps if wp in wp2i]
            for j, wp_idx in enumerate(wp_indices):
                text_x_idx_split[i][j] = wp_idx
            text_x_len[i] = len(wp_indices)

        text_x_idxs.append(text_x_idx.unsqueeze(0))
        text_x_lens.append(text_x_len.unsqueeze(0))
    return torch.cat(text_x_idxs, dim=0), torch.cat(text_x_lens, dim=0)


def ordinal_loss(logit, target):
    """Compute binary or softmax cross entropy loss."""
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--nsml_use', type=bool, default=True)
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--max_epoch', type=int, default=10)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=200)
    args.add_argument('--embedding', type=int, default=50)
    args.add_argument('--hidden_dim', type=int, default=100)
    args.add_argument('--keep_dropout', type=float, default=0.5)
    args.add_argument('--use_gpu', type=bool, default=True)
    args.add_argument('--vocab_size', type=int, default=10000)
    args.add_argument('--max_wp_len', type=int, default=10)
    args.add_argument('--max_words_len', type=int, default=20)

    args.add_argument('--learning_rate', type=float, default=2*1e-3)
    args.add_argument('--lr_decay', type=float, default=0.5)
    args.add_argument('--l2', type=float, default=1e-6)

    args.add_argument('--static', action='store_true', default=False, help='fix the embedding')

    # Select model
    args.add_argument('--model', type=str, default='ImgText2Vec', choices=['ImgText2Vec','regression', 'classification', 'bilstmwithattn', 'cnntext'])
    config = args.parse_args()

    print('HAS_DATASET :', HAS_DATASET)
    print('IS_ON_NSML :', IS_ON_NSML)
    print('DATASET_PATH :', DATASET_PATH)
    print(config)

    sp = []
    wp_vocab = []
    preprcess_infer = {}
    if config.mode == 'train':
        sp, wp_vocab = build_vocab(config.mode, DATASET_PATH, vocab_size=config.vocab_size)
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen,
                                     max_word_len=config.max_words_len, max_wp_len=config.max_wp_len, n_class=11)
        vocab_size = len(dataset.i2wp)
    else:
        vocab_size = 19488 #19475

    model_type = {
        'regression' : Regression(config.embedding, config.strmaxlen),
        'classification' : Classification(config.embedding, config.strmaxlen),
        'bilstmwithattn' : LSTMAttention(config),
        'cnntext': CNN_Text(config),
        'ImgText2Vec': ImgText2Vec(vocab_size, max_wp_len=config.max_wp_len, max_words_len=config.max_words_len)
    }

    model = model_type[config.model]
    print(model)
    print('GPU_NUM: ', GPU_NUM)
    if config.use_gpu:
        model = model.cuda()

    # DONOTCHANGE: Reserved for nsml use
    if config.nsml_use:
        bind_model(model, sp, wp_vocab, preprcess_infer, config)
    else:
        IS_ON_NSML = False
        DATASET_PATH = os.path.join('/tmp/pycharm_project562/16_tcls_movie')

    criterion_type = {
        'regression' : nn.MSELoss(),
        'classification' : nn.CrossEntropyLoss(),
        'bilstmwithattn' : nn.CrossEntropyLoss(),
        'cnntext': nn.CrossEntropyLoss(),
        'ImgText2Vec': nn.CrossEntropyLoss()
    }

    criterion = criterion_type[config.model]
    reg_criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), weight_decay=config.l2, lr=config.learning_rate)
    scheduler = StepLR(optimizer, step_size=3, gamma=config.lr_decay)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause and config.nsml_use:
        nsml.paused(scope=locals())

    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':

        print('train data loading...')
        # 데이터를 로드합니다(참고 : 데이터셋의 class 비율이 많이 다릅니다).

        train_loader = DataLoader(dataset=dataset,
                                  batch_size=config.batch,
                                  shuffle=True,
                                  #collate_fn=collate_fn,
                                  num_workers=4)
        total_batch = len(train_loader)
        print('training start!')
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.max_epoch):
            scheduler.step()
            avg_loss = []
            predictions = []
            label_vars = []
            for i, (x_char, x_text, labels) in enumerate(train_loader):
                review = x_text[0]
                x_text = (x_text[1].cuda(), x_text[2].cuda())
                x_char = Variable(x_char.long()).cuda()

                predictions = model(x_char, x_text)
                a = predictions.data.squeeze(dim=1)#.tolist()
                #print(torch.sum(a > 0.5, dim=1).tolist())

                # point = predictions.data.squeeze(dim=1).tolist()
                # print(config.model)
                # if config.model in ['classification', 'cnntext', 'bilstmwithattn', 'ImgText2Vec']:
                #     point = [np.argmax(p) for p in point]
                # print(point)

                label_vars = Variable(labels)
                #reg_vars = Variable((labels.clone() * 2 - 11) / 9)

                if config.use_gpu:
                    label_vars = label_vars.cuda()
                    #reg_vars = reg_vars.cuda()
                if config.model in ['classification','cnntext', 'bilstmwithattn', 'ImgText2Vec']:
                    # 0~10 범위의 label
                    #label_vars = label_vars.long()
                    label_vars = label_vars.float()
                #print(label_vars, predictions)
                #cls_loss = criterion(predictions, label_vars)
                #reg_loss = reg_criterion(reg, reg_vars)
                ord_loss = ordinal_loss(predictions, label_vars)

                loss = ord_loss #cls_loss# + reg_loss
                if config.use_gpu:
                    loss = loss.cuda()
                # print(review)
                # print('wps', x_text[0].size(), x_text[0].view(-1,10,5), x_text[1])
                # print('pred', predictions)
                # print('gt', label_vars)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i%100==0:
                    print('Batch : ', i + 1, '/', total_batch,
                        ', Loss in this minibatch: ', loss.item(), 'cls_loss:', ord_loss.item())#, 'reg_loss:', reg_loss.item())
                avg_loss.append(loss.item())
            print('epoch:', epoch, ' train_loss:', np.array(avg_loss).mean())

            # nsml ps, 혹은 웹 상의 텐서보드에 나타나는 값을 리포트하는 함수입니다.
            if config.nsml_use:
                nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.max_epoch,
                            train__loss=np.array(avg_loss).mean(), step=epoch)
                # DONOTCHANGE (You can decide how often you want to save the model)
                nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        datapath = os.path.join(DATASET_PATH, 'test/test_data')
        with open(os.path.join(DATASET_PATH, 'test/test_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()

        res = nsml.infer(reviews)
        print(res)
