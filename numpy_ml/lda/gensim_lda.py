#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Time    : 2019-12-19 12:01
# Author  : litf

##################################################
#### 一些辅助函数，doc 转换成：词袋模型

# gensim.corpora.dictionary.Dictionary
# doc2bow

from gensim.corpora import Dictionary
dct = Dictionary(["máma mele maso".split(), "ema má máma".split()])
# Convert document into the bag-of-words (BoW) format = list of (token_id, token_count).
dct.doc2bow(["this","is","máma"])
#  [(2, 1)]
dct.doc2bow(["this","is","máma"], return_missing=True)
# ([(2, 1)], {'is': 1, 'this': 1}) # this is 不再dict 中

dct.add_documents([["cat", "say", "meow"], ["dog"]])


#### gensim corpus
from gensim.test.utils import common_corpus
common_corpus
# [[(0, 1), (1, 1), (2, 1)],
#  [(0, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)],
#  [(2, 1), (5, 1), (7, 1), (8, 1)],]



##################################################

# 1. 读取语料

with open("numpy_ml/lda/shediaoyingxiongchuan_jinyong.txt",mode="r",encoding="utf-8",) as f:
    f_str = f.read()

len(f_str)

# 查看章节
import re
re.findall("第.{1,3}回.{1,10}",f_str)
# 匹配任意除了 \n


# 1.1 分章节
docs_thems = re.findall("第.{1,3}回",f_str)
docs_thems.append("全书完")

docs_thems_idx = [f_str.find(docs_them) for docs_them in docs_thems]
len(docs_thems_idx)

# Out[158]:
# [2707,
#  29834,
#  55979,
#  78522,
#  102364,
#  124074,
f_str[2707:2720]
# '第一回 风雪惊变\n钱塘江浩'

docs = [f_str[docs_thems_idx[idx]:docs_thems_idx[idx+1]] for idx in range(len(docs_thems_idx)-1)]
len(docs)  # 40个章节的故事



# 2. 分词
from importlib import reload
from numpy_ml.lda import seg_utils
reload(seg_utils)
from numpy_ml.lda.seg_utils import *
seg = SmsSeg(stopwords_path="numpy_ml/lda/stopwords.txt",)
seg.get_stopswords()[:3] # ['!', 'n', 'r'] \n 没去掉

docs_seg = list(map(lambda x: seg.lcut(str(x).strip()),docs))

docs_seg[0]



# 2.1 查看高频词汇

from collections import Counter
c = Counter(sum(docs_seg,[]))
c.most_common(10)
# [('郭靖', 2612),
#  ('黄蓉', 1752),
#  ('洪七公', 1057),
#  ('欧阳锋', 1046),
#  ('说道', 1043),
#  ('师父', 876),
#  ('黄药师', 870),
#  ('心中', 776),
#  ('武功', 770),
#  ('两人', 715)]



# 3. 建立字典
from gensim.corpora import Dictionary
dictionary = Dictionary(docs_seg)

len(dictionary) # 41501
#存储字典文件
# dictionary.save('')
#加载字典文件
# Dictionary.load('')


# 4.bag of words represention
# 4.1 词袋向量表示doc
# Term Document Frequency，也就是词频
corpus = [dictionary.doc2bow(doc_seg) for doc_seg in docs_seg]

# 4.2 tfidf 表示doc
from gensim.models import TfidfModel
tfidf = TfidfModel(corpus)
corpus[0][0] # (0, 1)

corpus_tfidf = tfidf[corpus]


#  5. lda
from gensim.models import LdaModel
# 用词袋模型表示doc,会把郭靖 黄蓉这样高频都显示出来

lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)

# 用tfidf 表示doc
lda = LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=40)


# 1）Get a single topic as a formatted string
lda.print_topics()
lda.print_topic(topicno=0)


# 2）top topics
# `(topic_repr, coherence_score)`
# Calculate the coherence for each topic; default is Umass coherence.

# 计算主题一致性
from gensim.models.coherencemodel import CoherenceModel

top_topics = lda.top_topics(corpus=corpus,topn=10)
len(top_topics) # 10
top_topics[9]


# 3)  get_document_topics 某doc 的主题
lda.get_document_topics(corpus[6])
# [(3, 0.22955635), (6, 0.7703246)]


# 4）print_topics
# Get the most significant topics (alias for `show_topics()` method).

lda.print_topics(num_topics = 10,num_words=10)
#  (9,
#   '0.008*"郭靖" + 0.006*"黄蓉" + 0.004*"欧阳锋" + 0.003*"说道" + 0.003*"黄药师" + 0.003*"梅超风" + 0.003*"两人" + 0.003*"师父" + 0.003*"只见" + 0.003*"心中"')]


# 5) get_topics
# which represents the term topic matrix learned during inference.
lda.get_topics().shape
# (10, 41501)



# 6. 评估
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

# Model perplexity and topic coherence
# provide a convenient measure to judge how good a given topic model is


# 困惑度，越低越好
# Compute Perplexity

print('\nPerplexity: ', lda.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
# Perplexity:  -33.670002762618736


# 计算主题一致性，越高越好
# Compute Coherence Score
from gensim.models import CoherenceModel
coherence_model_lda = CoherenceModel(model=lda,
                                     corpus=corpus,
                                     dictionary=dictionary,
                                     coherence='u_mass') # c_v
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
# Coherence Score:  -3.0498673090606365






# 7. 可视化
# 在 notebook 上
# Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# vis


# 8. Building LDA Mallet Model
import gensim
# Mallet’s version, however, often gives a better quality of topics.
# # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
mallet_path = 'numpy_ml/lda/mallet-2.0.8/bin/mallet' # update this path
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path,
                                             corpus=corpus,
                                             num_topics=40,
                                             id2word=dictionary)

ldamallet.print_topics()


# 9. 选取最优参数k
import gensim

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)



# 10.在每个句子中找到主要话题
# 11.找到每个主题最具代表性的doc
# 12.主题在doc上的分布比例





