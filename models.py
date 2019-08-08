# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad_pack

from nsml import GPU_NUM

class Regression(nn.Module):
    """
    영화리뷰 예측을 위한 Regression 모델입니다.
    """
    def __init__(self, embedding_dim: int, max_length: int):
        """
        initializer

        :param embedding_dim: 데이터 임베딩의 크기입니다
        :param max_length: 인풋 벡터의 최대 길이입니다 (첫 번째 레이어의 노드 수에 연관)
        """
        super(Regression, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_size = 251
        self.output_dim = 1  # Regression
        self.max_length = max_length

        # 임베딩
        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)

        # 레이어
        self.fc1 = nn.Linear(self.max_length * self.embedding_dim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, data: list):
        """

        :param data: 실제 입력값
        :return:
        """
        # 임베딩의 차원 변환을 위해 배치 사이즈를 구합니다.
        batch_size = len(data)
        # list로 받은 데이터를 torch Variable로 변환합니다.
        data_in_torch = Variable(torch.from_numpy(np.array(data)).long())
        # 만약 gpu를 사용중이라면, 데이터를 gpu 메모리로 보냅니다.
        if GPU_NUM:
            data_in_torch = data_in_torch.cuda()
        # 뉴럴네트워크를 지나 결과를 출력합니다.
        embeds = self.embeddings(data_in_torch)
        hidden = self.fc1(embeds.view(batch_size, -1))
        hidden = self.fc2(hidden)
        hidden = torch.relu(hidden)
        output = self.fc3(hidden)
        return output

class Classification(nn.Module):
    """
    영화리뷰 예측을 위한 Classification 모델입니다.
    """
    def __init__(self, embedding_dim: int, max_length: int):
        """
        initializer

        :param embedding_dim: 데이터 임베딩의 크기입니다
        :param max_length: 인풋 벡터의 최대 길이입니다 (첫 번째 레이어의 노드 수에 연관)
        """
        super(Classification, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_size = 251
        self.output_dim = 11  # Classification(0~10 범위의 label)
        self.max_length = max_length

        # 임베딩
        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)

        # 레이어
        self.fc1 = nn.Linear(self.max_length * self.embedding_dim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, self.output_dim)

    def forward(self, data: list):
        """

        :param data: 실제 입력값
        :return:
        """
        # 임베딩의 차원 변환을 위해 배치 사이즈를 구합니다.
        batch_size = len(data)
        # list로 받은 데이터를 torch Variable로 변환합니다.
        data_in_torch = Variable(torch.from_numpy(np.array(data)).long())
        # 만약 gpu를 사용중이라면, 데이터를 gpu 메모리로 보냅니다.
        if GPU_NUM:
            data_in_torch = data_in_torch.cuda()
        # 뉴럴네트워크를 지나 결과를 출력합니다.
        embeds = self.embeddings(data_in_torch)
        hidden = self.fc1(embeds.view(batch_size, -1))
        hidden = self.fc2(hidden)
        hidden = torch.relu(hidden)
        output = self.fc3(hidden)
        return output


class SentpieceModel(nn.Module):
    def __init__(self, x_vocab_size, char_emb_size=50,
                 word_emb_size=200, hidden_size=200, nlayers=2, dropout=0.2, max_wp_len=10, max_words_len=20, n_class=11):
        super(SentpieceModel, self).__init__()
        self.x_vocab_size = x_vocab_size
        self.char_emb_size = char_emb_size
        self.word_emb_size = word_emb_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.max_wp_len = max_wp_len
        self.max_words_len = max_words_len
        self.use_gpu = True

        V = 251
        D = self.char_emb_size
        C = 11
        Ci = 1
        Co = 100  # args.kernel_num
        Ks = [3, 4, 5, 6, 7, 8, 9]  # args.kernel_sizes

        self.char_embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        self.dropout = nn.Dropout(dropout)
        self.char_feature = nn.Linear(len(Ks) * Co, hidden_size)

        self.emb = nn.Embedding(x_vocab_size, self.word_emb_size, padding_idx=0)
        self.wp_lstm = nn.LSTM(self.word_emb_size, self.word_emb_size, nlayers, dropout=dropout)

        self.words_bilstm = nn.LSTM(self.word_emb_size, hidden_size // 2, num_layers=1,  batch_first=True, bidirectional=True)

        self.text_feature = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        self.regression = nn.Linear(11, 1)
        self.fc = nn.Linear(hidden_size, n_class)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.use_gpu:
            h0 = Variable(torch.zeros(2 * 1, batch_size, self.hidden_size // 2).cuda())
            c0 = Variable(torch.zeros(2 * 1, batch_size, self.hidden_size // 2).cuda())
        else:
            h0 = Variable(torch.zeros(2 * 1, batch_size, self.hidden_size // 2))
            c0 = Variable(torch.zeros(2 * 1, batch_size, self.hidden_size // 2))
        return (h0, c0)

    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(rnn_out, merged_state)
        weights = F.softmax(weights.squeeze(2)).unsqueeze(2)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)

    def forward(self, x_char, x_text):
        ### sentpiece net
        x_word_embs = self.sent2vec(x_text)
        hidden = self.init_hidden(x_word_embs.size(0))  #
        rnn_out, hidden = self.words_bilstm(x_word_embs, hidden)
        h_n, c_n = hidden
        attn_out = self.attention(rnn_out, h_n)

        ### char net
        x_char = self.char_embed(x_char)
        x_char = x_char.unsqueeze(1)  # (N, Ci, W, D)
        x_char = [F.leaky_relu(conv(x_char)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x_char = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_char]  # [(N, Co), ...]*len(Ks)
        x_char = torch.cat(x_char, 1)
        x_char = self.dropout(x_char)  # (N, len(Ks)*Co)
        x_char = self.char_feature(x_char)

        feat = torch.cat([x_char, attn_out], dim=1)
        feat = self.text_feature(feat)
        rate = self.fc(feat)

        reg = self.regression(rate)
        return rate, reg.clamp(-1, 1)

    def sent2vec(self, titles):
        sent, sent_lens = titles
        sent_split = sent.split(self.max_wp_len, 1)

        sent_lens_split = sent_lens.split(1, 1)

        batch_size = sent.size(0)
        vec = torch.zeros(batch_size, self.max_words_len, self.word_emb_size).cuda()
        sum_count = 0

        for i, (x, x_len) in enumerate(zip(sent_split, sent_lens_split)):
            if x_len.sum() == 0:
                continue
            sum_count += 1
            non_zero_idx = torch.nonzero(x_len)

            if len(non_zero_idx) != batch_size:
                x_len = x_len[non_zero_idx[:, 0]]
                x = x[non_zero_idx[:, 0]]

            x_len = x_len.contiguous().view(-1)
            x = x.contiguous()

            x_len, indices = x_len.sort(0, descending=True)
            x = x[indices]
            _, rev_indices = indices.sort()
            emb = self.emb(x)
            pack_emb = pack(emb, x_len.tolist(), batch_first=True)
            _, state = self.wp_lstm(pack_emb)
            output = state[0][:, rev_indices]
            output = output[-1]
            if len(non_zero_idx) != batch_size:
                vec[non_zero_idx[:, 0], i] = output
            else:
                vec[:, i] = output

        return vec


