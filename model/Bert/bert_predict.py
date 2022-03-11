#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch

from config import is_cuda, root_path
from model.Bert.bert_model import BertConfig
from model.Bert.bert_seq2seq import Seq2SeqModel
from model.Bert.tokenizer import load_chinese_base_vocab


class bertSeq2Seq(object):
    def __init__(self, model_path, is_cuda):
        self.word2idx = load_chinese_base_vocab()
        self.config = BertConfig(len(self.word2idx))
        self.bert_seq2seq = Seq2SeqModel(self.config)
        self.is_cuda = is_cuda
        if is_cuda:
            device = torch.device("cuda")
            self.bert_seq2seq.load_state_dict(torch.load(model_path))
            self.bert_seq2seq.to(device)
        else:
            checkpoint = torch.load(model_path,
                                    map_location=torch.device("cpu"))
            self.bert_seq2seq.load_state_dict(checkpoint)
        # 加载state dict参数
        self.bert_seq2seq.eval()

    def generate(self, text, k=5):
        result = self.bert_seq2seq.generate(text,
                                            beam_size=k,
                                            is_cuda=self.is_cuda)
        return result


if __name__ == "__main__":
    bs = bertSeq2Seq(os.path.join(root_path, 'saved_model/bert.model.epoch.1000'), is_cuda)
    text = '我想向你咨询一下医药相关问题'
    print(bs.generate(text, k=5))
