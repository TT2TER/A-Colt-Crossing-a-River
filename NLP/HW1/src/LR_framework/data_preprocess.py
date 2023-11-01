# -*- coding='utf-8' -*-
import numpy as np
import random
import jieba
import os
from torch.utils.data import DataLoader, Dataset
from gensim.models.keyedvectors import KeyedVectors
import pickle
"""
程序功能：读取原始语料数据，制作Dataset并保存，以用于在训练时生成mini-batch
注意：本实现中，模型的输入是拼接句子中的所有词向量后得到的一个向量，设置句子的长度为固定值80（超出的部分截断，不足的部分padding），
可以根据任务要求决定模型输入为词向量求和还是拼接（请修改string2vec和get_data中的实现）
"""
TRAIN_SPLIT = 0.8  # 训练集所占比例
WORD_DIM = 300  # 词向量维度
MAX_SENT_LEN = 80
stopwords = []

def mystrip(ls):
    """
    函数功能：消除句尾换行
    """
    for i in range(len(ls)):
        ls[i] = ls[i].strip("\n")
    return ls

def remove_stopwords(_words):
    """
    函数功能：去掉停用词
    """
    _i = 0
    for _ in range(len(_words)):
        if _words[_i] in stopwords or _words[_i].strip() == "":
            # print(_words[_i])
            _words.pop(_i)
        else:
            _i += 1
    return _words

def load_data():
    """
    函数功能：读取原始语料。
    :return: 积极句子，积极label，消极句子，消极label
    """
    jieba.setLogLevel(jieba.logging.INFO)

    pos_sentence, pos_label, neg_sentence, neg_label = [], [], [], []

    pos_fname = './标注语料/positive'  # 请改为积极语料库实际路径
    neg_fname = './标注语料/negative'  # 请改为消极语料库实际路径

    for f_name in os.listdir(pos_fname):
        with open(pos_fname+'/'+f_name, encoding='utf-8') as f_i:
            sent = ""
            for line in f_i:
                line = line.strip()
                if line:
                    sent += line
            words = jieba.lcut(sent, cut_all=True)
            pos_sentence.append(remove_stopwords(words))

            pos_label.append(1)  # label为1表示积极，label为0表示消极
            
    for f_name in os.listdir(neg_fname):
        with open(neg_fname+'/'+f_name, encoding='utf-8') as f_i:
            sent = ""
            for line in f_i:
                line = line.strip()
                if line:
                    sent += line
            words = jieba.lcut(sent, cut_all=True)
            neg_sentence.append(remove_stopwords(words))
            neg_label.append(0)

    return pos_sentence, pos_label, neg_sentence, neg_label

def string2vec(word_vectors, sentence):
    """
    函数功能：将sentence中的string词语转换为词向量
    :param word_vectors: 词向量表
    :param sentence: 原始句子（单词为string格式）
    :return: 将string改为词向量
    注意：如果采用词向量求和的方式，请取消本函数中截断和padding的操作
    """
    for i in range(len(sentence)):  # 遍历所有句子
        sentence[i] = sentence[i][:MAX_SENT_LEN]  # 截断句子
        line = sentence[i]
        for j in range(len(line)):
            if line[j] in word_vectors:  # 如果是登录词 得到其词向量表示
                line[j] = word_vectors.get_vector(line[j])
            else:  # 如果不是登录词 设置为随机词向量
                line[j] = np.random.uniform(-0.01, 0.01, WORD_DIM).astype("float32")
        if len(line) < MAX_SENT_LEN:  # padding词设置为随机词向量
            for k in range(MAX_SENT_LEN-len(line)):
                sentence[i].append(np.random.uniform(-0.01, 0.01, WORD_DIM).astype("float32"))
    return sentence


class Corpus(Dataset):
    """
    定义数据集对象，用于构建DataLoader迭代器
    """
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

def get_data():
    global stopwords
    # 读取停用词列表：
    with open("./stopwords.txt", encoding="utf-8") as f:
        stopwords = f.readlines()
        stopwords = mystrip(stopwords)
    # 读取原始数据：
    pos_sentence, pos_label, neg_sentence, neg_label = load_data()

    sentence = pos_sentence + neg_sentence
    label = pos_label + neg_label

    sentence = sentence[:]
    label = label[:]

    shuffle = list(zip(sentence, label))
    random.shuffle(shuffle)  # 打乱数据集
    sentence[:], label[:] = zip(*shuffle)

    # 划分训练集、测试集
    assert len(sentence) == len(label)
    length = int(TRAIN_SPLIT*len(sentence))
    train_sentence = sentence[:length]
    train_label = label[:length]
    test_sentence = sentence[length:]
    test_label = label[length:]

    # 加载词向量
    print("loading word2vec...")
    word_vectors = KeyedVectors.load_word2vec_format("sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5")
    print("loading end")

    # 将string单词转为词向量
    train_sentence = string2vec(word_vectors, train_sentence)
    test_sentence = string2vec(word_vectors, test_sentence)

    # 拼接一句话中的所有词向量（可根据要求改为对所有词向量求和）
    train_sentence = [np.concatenate(wordvecs) for wordvecs in train_sentence]
    test_sentence = [np.concatenate(wordvecs) for wordvecs in test_sentence]

    # 生成数据集
    train_set = Corpus(train_sentence, train_label)
    test_set = Corpus(test_sentence, test_label)

    return train_set, test_set

if __name__ == "__main__":
    # 生成并保存数据集，注意根据实际情况设置输出路径
    train_set, test_set = get_data()
    outpath = './train_set.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(train_set, f)
    outpath = './test_set.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(test_set, f)
