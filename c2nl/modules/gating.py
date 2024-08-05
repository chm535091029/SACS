from __future__ import unicode_literals, print_function, division
import math
import torch
import numpy as np
import torch.nn as nn


class Gating(nn.Module):
    def __init__(self, in_feature,source_vocab_size=1000):
        super(Gating, self).__init__()
        # self.embedding = nn.Embedding(source_vocab_size, in_feature)
        self.lh = nn.Linear(in_feature, in_feature)
        # self.l1 = nn.Linear(init_summ_len, source_code_length)

    def forward(self, m_source_code, n_source_code):
        """
            m_source_code:[bs, src_len, d_model]
            e_method_name: [bs, init_summ_len,d_model]
            return->
            M: [batch_size, src_len, d_model]
        """
        q = self.lh(m_source_code)
        # print(q.shape)
        k1 = self.lh(m_source_code)
        # print(k1.shape)
        # k2 = self.embedding(summ_keywords)

        k2 = n_source_code
        # k2 = self.l1(k2)
        # k2 = k2.transpose(1, 2)
        # print(k2.shape)

        v1 = self.lh(m_source_code)
        # v2 = self.embedding(init_summ)
        v2 = n_source_code
        # v2 = self.l1(v2)
        v2 = v2

        kv1 = torch.sum(torch.mul(q, k1), dim=(1,), keepdim=True)
        kv2 = torch.sum(torch.mul(q, k2), dim=(1,), keepdim=True)
        kv1_1 = kv1 - torch.max(kv1, kv2)
        kv2_1 = kv2 - torch.max(kv1, kv2)
        kv1 = torch.exp(kv1_1)
        kv2 = torch.exp(kv2_1)
        kv1_S = kv1 / (kv1 + kv2)
        kv2_S = kv2 / (kv1 + kv2)
        kv1_S = torch.mul(kv1_S, v1)
        kv2_S = torch.mul(kv2_S, v2)

        M = kv1_S + kv2_S
        return M