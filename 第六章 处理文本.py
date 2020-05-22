# 6.1清洗文本
text_data = ["Interrobang. By Aishwarya Henriette   ",
             "Parking And Going. By Karl Gautier",
             "  Today Is The night. By Jarek Prakash    "]  # 创建文本

strip_whitespace = [string.strip() for string in text_data]  # 去掉文本两端的空格
strip_whitespace
remove_periods = [string.replace(".", "") for string in strip_whitespace]    # 删除句点
remove_periods


def capitalizer(string:str) -> str:     # 创建自定义函数
    return string.upper()


[capitalizer(string) for string in remove_periods]  # 应用自定义函数

import re


def replace_letters_with_X(string: str) -> str:       # 创建函数 <使用正则表达式做复杂的字符串操作>
    return re.sub(r"[a-zA-Z]", "X", string)


[replace_letters_with_X(string) for string in remove_periods]   # 应用自定义函数


# 6.2 解析并清洗HTML
from bs4 import BeautifulSoup
html = """
    <div class='full_name'><span style='font-weight:bold'>
    Masego Azra"
    """         # 创建一些HTML 代码

soup = BeautifulSoup(html, "lxml")      # 解析 HTML
soup.find("div", {"class": "full_name"}).text   # 查找class 是 "full_name" 的div标签,并查看文本

# 6.3 移除标点
import unicodedata
import sys
text_data = ['Hi!!!!! I.Love. This. Song....',
             '10000% Agree!!!! #LoveIT',
             'Right?!?!']

punctuation = dict.fromkeys(i for i in range(sys.maxunicode)    # 创建一个标点字典
                            if unicodedata.category(chr(i)).startswith('P'))

[string.translate(punctuation) for string in text_data]     # 移除每个字符串里的标点
# 将Unicode 中的标点符号作为 key,None 作为其 value，然后将字符串中所有在punctuation字典中出现过的字符转换成None，高效地移除它们。

# 6.4 文本分词
from nltk.tokenize import word_tokenize
string = "The science of today is the technology of tomorrow."
word_tokenize(string)   # 分词
# ['The',
#  'science',
#  'of',
#  'today',
#  'is',
#  'the',
#  'technology',
#  'of',
#  'tomorrow',
#  '.']

from nltk.tokenize import sent_tokenize
string = "The science of today is the techology of tomorrow. Tomorrow is today."
sent_tokenize(string)   # 切分成句子
# ['The science of today is the techology of tomorrow.', 'Tomorrow is today.']

# 6.5 删除停止词 （stop word)
from nltk.corpus import stopwords
tokenized_words = ['i','am','going','to','go','to','the','store','and','park']
stop_words = stopwords.words('english') # 加载停止词
[word for word in tokenized_words if word not in stop_words]    # ['going', 'go', 'store', 'park']
stop_words  # 查看停止词 注意: stopwords 假设所有单词为小写形式

# 6.6 提取词干
from nltk.stem.porter import PorterStemmer

tokenized_words = ['i','am','humbled','by','this','traditional','meeting']
porter = PorterStemmer()    # 创建词干转换器
[porter.stem(word) for word in tokenized_words] # ['i', 'am', 'humbl', 'by', 'thi', 'tradit', 'meet']

# 6.7 标注词性

from nltk import pos_tag
from nltk import word_tokenize

text_data = "Chris loved outdoor running"

text_tagged = pos_tag(word_tokenize(text_data))
text_tagged     # [('Chris', 'NNP'), ('loved', 'VBD'), ('outdoor', 'RP'), ('running', 'VBG')]
# 一些常用标签:
# NPP 单数专有名词
# NN  单数或复数的名词
# RB  副词
# VBD 过去式的动词
# VBG 名动词或动词的现在分词形式
# JJ  形容词
# PRP 人称代词

[word for word, tag in text_tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS']]  # 过滤单词
# ['Chris'] 找到所有名词

# 现实中更可能遇到的情况是有一份数据，每个观察值包含一条推文，我们想把这些句子转换成用词性表示的特征(例如，如果有专有名词,特征值为1,否则为0
from sklearn.preprocessing import MultiLabelBinarizer
tweets = ["I am eating a burrito for breakfast",
          "Political science is an amazing field",
          "San Francisco is an awesome city"]

tagged_tweets = []
for tweet in tweets:
    tweet_tag = pos_tag(word_tokenize(tweet))
    tagged_tweets.append([tag for word, tag in tweet_tag])  # 为每个推文的每个单词加标签

one_hot_multi = MultiLabelBinarizer()       # 使用 one-hot 编码 将标签转换成特征
one_hot_multi.fit_transform(tagged_tweets)

one_hot_multi.classes_  # 查看特征名

# P106页 布朗语料库 可自行做了解

# 6.8 将文本编码成词袋 （Bag of Words)
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# 创建一组特征来表示观察值文本中包含的特定单词的数量
text_data = np.array(['I love Brazil.Brazil!',
                      'Sweden is best',
                      'Germany beats both'])
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

bag_of_words    # 稀疏矩阵
bag_of_words.toarray()  # 可以使用 toarray 查看每个观察值的词频统计矩阵
count.get_feature_names()   # 查看特征名

# 创建一个只包含国家名字的词袋特征矩阵
count_2gram = CountVectorizer(ngram_range=(1,2), stop_words="english",vocabulary=['brazil'])
# ngram_range 可以设置n元模型最大元和最小元 ，vocabulary 可以将观察值限定在仅在特定单词表中出现过的单词或短语
bag = count_2gram.fit_transform(text_data)
bag.toarray()
count_2gram.vocabulary_

# 6.9 按单词的重要性加权
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])
# 创建TF-IDF 特征矩阵
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)

feature_matrix.toarray()

tfidf.get_feature_names()   # 知道排序,与vocabulary方法效果类似
tfidf.vocabulary_
