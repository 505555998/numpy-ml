#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Time    : 2019-12-19 16:30
# Author  : litf

import jieba

### 分词utils
class SmsSeg:
    def __init__(self, stopwords_path,userdict_path=None, ):
        if userdict_path is not None:
            self.userdict_path = userdict_path
            jieba.load_userdict(self.userdict_path)
        if stopwords_path is not None:
            self.stopwords_path = stopwords_path
            self.stopswords = self.get_stopswords()

    def lcut(self, sentence):
        sentence = sentence.replace(" ","").replace("\n","")
        # fix 20191121
        # 修改 1）停用词 只去符号，比如 “给” 也在停用词中
        #      2）保留数字
        l = list(filter(lambda x: x not in self.stopswords and len(x)>=2, jieba.lcut(sentence)))

        return l

    def get_stopswords(self):
        with open(self.stopwords_path) as f:
            return f.read().split("\n")


    # def test(self,text):
    #     seg = SmsSeg(stopwords_path=self.stopwords_path,
    #                  userdict_path=self.userdict_path)
    #
    #     print("seg:",seg.lcut(text))
    #
    #     import jieba
    #     print("jieba",jieba.lcut(text))
