# -*- coding: utf-8 -*-

import jieba
import numpy as np
import time

"""
Naive Bayes句子分类模型
请在pass处按注释要求插入代码
"""

train_path = "./train.txt"
test_path = "./test.txt"

sum_words_neg = 0   # 训练集负向语料的总词数（用于计算词频）
sum_words_pos = 0   # 训练集正向语料的总词数

neg_sents_train = []  # 训练集中负向句子
pos_sents_train = []  # 训练集中正向句子
neg_sents_test = []  # 测试集中负向句子
pos_sents_test = []  # 测试集中正向句子
stopwords = []  # 停用词

def mystrip(ls):
    """
    消除句尾换行
    """
    for i in range(len(ls)):
        ls[i] = ls[i].strip("\n")
    return ls

def remove_stopwords(_words):
    """
    去掉停用词
    :param _words: 分词后的单词list
    :return: 去除停用词（无意义词）后的list
    """
    _i = 0

    for _ in range(len(_words)):
        if _words[_i] in stopwords:
            _words.pop(_i)
        else:
            _i += 1

    return _words

def my_init():
    """
    函数功能：对训练集做统计，记录训练集中正向和负向的单词数，并记录正向或负向条件下，每个词的出现次数，并收集测试句子
    return: 负向词频表，正向词频表（记录每个词及其出现次数）
    """
    neg_words = []  # 负向词列表
    _neg_dict = {}  # 负向词频表
    pos_words = []  # 正向词列表
    _pos_dict = {}  # 正向词频表

    global sum_words_neg, sum_words_pos, neg_sents_train, pos_sents_train, stopwords

    # 读入stopwords
    with open("./stopwords.txt", encoding="utf-8") as f:
        stopwords = f.readlines()
        stopwords = mystrip(stopwords)

    # 收集训练集正、负向的句子
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip('\n')
            if line[0] == "0":
                neg_sents_train.append(line[1:])
            else:
                pos_sents_train.append(line[1:])

    # 收集测试集正、负向的句子
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip('\n')
            if line[0] == "0":  #
                neg_sents_test.append(line[1:])
            else:
                pos_sents_test.append(line[1:])
    # 获得负向训练语料的词列表neg_words
    for i in range(len(neg_sents_train)):
        words = jieba.lcut(neg_sents_train[i])
        words = remove_stopwords(words)
        neg_words.extend(words)


    # 获得负向训练语料的词频表_neg_dict
    for word in neg_words:
        if word in _neg_dict:
            _neg_dict[word] += 1
        else:
            _neg_dict[word] = 1

    # 获得正向训练语料的词列表pos_words
    for i in range(len(pos_sents_train)):
        words = jieba.lcut(pos_sents_train[i])
        words = remove_stopwords(words)  # 去掉停用词
        pos_words.extend(words)

    # 获得正向训练语料的词频表_pos_dict
    for word in pos_words:
        if word in _pos_dict:
            _pos_dict[word] += 1
        else:
            _pos_dict[word] = 1

    return _neg_dict, _pos_dict


if __name__ == "__main__":
    # 统计训练集：
    neg_dict, pos_dict = my_init()

    rights = 0  # 记录模型正确分类的数目
    neg_dict_keys = neg_dict.keys()
    pos_dict_keys = pos_dict.keys()

    # 测试：
    for i in range(len(neg_sents_test)):  # 用negative的句子做测试
        st = jieba.lcut(neg_sents_test[i])  # 分词，返回词列表
        st = remove_stopwords(st)  # 去掉停用词

        p_neg = 0  # Ci=neg的时候，目标函数的值
        p_pos = 0  # Ci=pos的时候，目标函数的值

        # 拉普拉斯平滑参数
        alpha = 1.

        # 计算情感类别的先验概率
        p_neg_class = (len(neg_sents_train) + alpha) / (len(neg_sents_train) + len(pos_sents_train) + 2 * alpha)
        p_pos_class = (len(pos_sents_train) + alpha) / (len(neg_sents_train) + len(pos_sents_train) + 2 * alpha)

        # 遍历测试句子中的每个词
        for word in st:
            # p(word|Ci=neg)
            p_word_given_neg = (neg_dict.get(word, 0) + alpha) / (sum(neg_dict.values()) + len(neg_dict) * alpha)
            # p(word|Ci=pos)
            p_word_given_pos = (pos_dict.get(word, 0) + alpha) / (sum(pos_dict.values()) + len(pos_dict) * alpha)

            p_neg += np.log(p_word_given_neg)
            p_pos += np.log(p_word_given_pos)

        p_neg += np.log(p_neg_class)
        p_pos += np.log(p_pos_class)
        #     p_neg *= p_word_given_neg
        #     p_pos *= p_word_given_pos

        # # 乘以先验概率
        # p_neg *= p_neg_class
        # p_pos *= p_pos_class

        # 如果 p_pos < p_neg，则表示当前句子更可能属于负向情感类别
        if p_pos < p_neg:
            #print(p_neg)
            rights += 1

    for i in range(len(pos_sents_test)):  # 用positive的数据做测试
        st = jieba.lcut(pos_sents_test[i])
        st = remove_stopwords(st)

        p_neg = 0  # Ci=neg的时候，目标函数的值
        p_pos = 0  # Ci=pos的时候，目标函数的值

        # 拉普拉斯平滑参数
        alpha = 1.

        # 计算情感类别的先验概率
        p_neg_class = (len(neg_sents_train) + alpha) / (len(neg_sents_train) + len(pos_sents_train) + 2 * alpha)
        p_pos_class = (len(pos_sents_train) + alpha) / (len(neg_sents_train) + len(pos_sents_train) + 2 * alpha)

        # 遍历测试句子中的每个词
        for word in st:
            # p(word|Ci=neg)
            p_word_given_neg = (neg_dict.get(word, 0) + alpha) / (sum(neg_dict.values()) + len(neg_dict) * alpha)
            # p(word|Ci=pos)
            p_word_given_pos = (pos_dict.get(word, 0) + alpha) / (sum(pos_dict.values()) + len(pos_dict) * alpha)

            p_neg += np.log(p_word_given_neg)
            p_pos += np.log(p_word_given_pos)

        p_neg += np.log(p_neg_class)
        p_pos += np.log(p_pos_class)
        #     p_neg *= p_word_given_neg
        #     p_pos *= p_word_given_pos

        # # 乘以先验概率
        # p_neg *= p_neg_class
        # p_pos *= p_pos_class
        
        if p_pos >= p_neg:
            #print(p_pos)
            rights += 1

    print("准确率:{:.1f}%".format(rights / (len(pos_sents_test) + len(neg_sents_test)) * 100))
