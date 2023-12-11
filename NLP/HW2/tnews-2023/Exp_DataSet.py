import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset
from gensim.models.keyedvectors import KeyedVectors
import pickle
import jieba
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModel
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


class Dictionary(object):
    def __init__(self, path):
        self.word2tkn = {"[PAD]": 0}
        self.tkn2word = ["[PAD]"]

        self.label2idx = {}
        self.idx2label = []

        # 获取 label 的 映射
        with open(os.path.join(path, "labels.json"), "r", encoding="utf-8") as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data["label"], one_data["label_desc"]
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) - 1

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]


class Corpus(object):
    """
    完成对数据集的读取和预处理，处理后得到所有文本数据的对应的 token 表示及相应的标签。

    该类适用于任务一、任务二，若要完成任务三，需对整个类进行调整，例如，可直接调用预训练模型提供的 tokenizer 将文本转为对应的 token 序列。
    """

    def __init__(self, path, max_token_per_sent):
        self.dictionary = Dictionary(path)

        self.max_token_per_sent = max_token_per_sent

        self.train = self.tokenize(os.path.join(path, "train.json"))
        self.valid = self.tokenize(os.path.join(path, "dev.json"))
        self.test = self.tokenize(os.path.join(path, "test.json"), True)

        # -----------------------------------------------------begin-----------------------------------------------------#
        # 若要采用预训练的 embedding, 需处理得到 token->embedding 的映射矩阵 embedding_weight。矩阵的格式参考 nn.Embedding() 中的参数 _weight
        # 注意，需考虑 [PAD] 和 [UNK] 两个特殊词向量的设置
        # 先检查本地有没有存好的 embedding_weight，若有则直接读取，若没有则生成并存储
        if os.path.exists("embedding_weight.pkl"):
            with open("embedding_weight.pkl", "rb") as f:
                self.embedding_weight = pickle.load(f)
                print("loaded embedding_weight...")
        else:
            # 生成 embedding_weight
            print("loading word2vec...")
            word_vectors = KeyedVectors.load_word2vec_format(
                "sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5"
            )
            print("loading end")
            self.embedding_weight = np.zeros((len(self.dictionary.tkn2word), 300))
            # 先处理 [PAD] 和 [UNK]
            self.embedding_weight[0] = np.zeros(300).astype("float32")
            # [UNK]会在后面处理
            # 再处理其他词
            for i in range(1, len(self.dictionary.tkn2word)):
                if self.dictionary.tkn2word[i] in word_vectors:
                    self.embedding_weight[i] = word_vectors.get_vector(
                        self.dictionary.tkn2word[i]
                    )
                else:
                    self.embedding_weight[i] = np.random.uniform(
                        -0.01, 0.01, 300
                    ).astype(
                        "float32"
                    )  # 这个是UNK的词向量？
            # 存储 embedding_weight
            self.embedding_weight = torch.from_numpy(self.embedding_weight).to(
                torch.float32
            )
            with open("embedding_weight.pkl", "wb") as f:
                pickle.dump(self.embedding_weight, f)

        # ------------------------------------------------------end------------------------------------------------------#

    def pad(self, origin_token_seq):
        """
        padding: 将原始的 token 序列补 0 至预设的最大长度 self.max_token_per_sent
        """
        if len(origin_token_seq) > self.max_token_per_sent:
            return origin_token_seq[: self.max_token_per_sent]
        else:
            return origin_token_seq + [
                0 for _ in range(self.max_token_per_sent - len(origin_token_seq))
            ]

    def tokenize(self, path, test_mode=False):
        """
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        """
        idss = []
        labels = []
        with open(path, "r", encoding="utf8") as f:
            total_lines = sum(1 for line in f)
        with open(path, "r", encoding="utf8") as f:
            for line in tqdm(f, total=total_lines, dynamic_ncols=True):
                one_data = json.loads(line)  # 读取一条数据
                sent = one_data["sentence"]
                # -----------------------------------------------------begin-----------------------------------------------------#
                # 若要采用预训练的 embedding, 需在此处对 sent 进行分词
                # 使用jieba分词
                sent = jieba.lcut(sent)
                global stopwords
                # 读取停用词列表：
                with open("./stopwords.txt", encoding="utf-8") as f:
                    stopwords = f.readlines()
                    stopwords = mystrip(stopwords)
                # 去除停用词
                sent = remove_stopwords(sent)

                # ------------------------------------------------------end------------------------------------------------------#
                # 向词典中添加词
                for word in sent:
                    self.dictionary.add_word(word)

                ids = []
                for word in sent:
                    ids.append(self.dictionary.word2tkn[word])
                idss.append(self.pad(ids))

                # 测试集无标签，在 label 中存测试数据的 id，便于最终预测文件的打印
                if test_mode:
                    label = json.loads(line)["id"]
                    labels.append(label)
                else:
                    label = json.loads(line)["label"]
                    labels.append(self.dictionary.label2idx[label])

            idss = torch.tensor(np.array(idss))
            labels = torch.tensor(np.array(labels)).long()

        return TensorDataset(idss, labels)


class Corpus_bert(object):
    # 针对bert模型的预处理
    def __init__(self, path, max_token_per_sent,method=-1):
        self.dictionary = Dictionary(path)
        self.method=method
        self.max_token_per_sent = max_token_per_sent

        self.train = self.tokenize(os.path.join(path, "train.json"))
        self.valid = self.tokenize(os.path.join(path, "dev.json"))
        self.test = self.tokenize(os.path.join(path, "test.json"), True)

    def tokenize(self, path, test_mode=False):
        """
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        """
        idss = []
        labels = []
        attention_masks = []
        token_type_ids = []
        method=self.method
        if method==-1:
            print("error:method=-1")
        if method ==0 or method ==1:
            print("base")
            tokenizer = BertTokenizer.from_pretrained(
                pretrained_model_name_or_path="bert-base-chinese",  # 可选，huggingface 中的预训练模型名称或路径，默认为 bert-base-chinese
                cache_dir="./",  # 将数据保存到的本地位置，使用cache_dir 可以指定文件下载位置
                force_download=False,
            )
        if method ==2 or method ==3:
            print("wwm")
            tokenizer = BertTokenizer.from_pretrained(
                pretrained_model_name_or_path="hfl/chinese-bert-wwm",  # 可选，huggingface 中的预训练模型名称或路径，默认为 bert-base-chinese
                cache_dir="./",  # 将数据保存到的本地位置，使用cache_dir 可以指定文件下载位置
                force_download=False,
            )
        if method ==4 or method ==5:
            print("ernie")
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path="nghuyong/ernie-3.0-base-zh",  # 可选，huggingface 中的预训练模型名称或路径，默认为 bert-base-chinese
                cache_dir="./",  # 将数据保存到的本地位置，使用cache_dir 可以指定文件下载位置
                force_download=False,
            )
        first=True #第一个句子
        with open(path, "r", encoding="utf8") as f:
            total_lines = sum(1 for line in f)
        with open(path, "r", encoding="utf8") as f:
            for line in tqdm(f, total=total_lines, dynamic_ncols=True):
                one_data = json.loads(line)  # 读取一条数据
                sent = one_data["sentence"]
                # -----------------------------------------------------begin-----------------------------------------------------#
                tokenizer_output = tokenizer.encode_plus(
                    sent,
                    add_special_tokens=True,
                    padding="max_length",#补齐
                    truncation=True,# 截断
                    max_length=self.max_token_per_sent,
                    return_token_type_ids=True,
                    # 返回attention_mask
                    return_attention_mask=True,
                    # 返回special_tokens_mask 特殊符号标识
                    return_special_tokens_mask=True,
                    return_tensors="pt",# 返回的类型为pytorch tensor
                )
                # print(tokenizer_output)
                # print(tokenizer.decode(tokenizer_output.input_ids))
                # ------------------------------------------------------end------------------------------------------------------#
                if first:
                    idss = tokenizer_output["input_ids"]
                    attention_masks = tokenizer_output["attention_mask"]
                    token_type_ids = tokenizer_output["token_type_ids"]
                    first=False
                else:
                    # print(idss.shape)
                    # print(tokenizer_output["input_ids"].shape)
                    # print(tokenizer_output["input_ids"])
                    idss = torch.cat((idss, tokenizer_output["input_ids"]), 0)
                    attention_masks = torch.cat(
                        (attention_masks, tokenizer_output["attention_mask"]), 0
                    )
                    token_type_ids = torch.cat(
                        (token_type_ids, tokenizer_output["token_type_ids"]), 0
                    )

                # 测试集无标签，在 label 中存测试数据的 id，便于最终预测文件的打印
                if test_mode:
                    label = json.loads(line)["id"]
                    labels.append(label)
                else:
                    label = json.loads(line)["label"]
                    labels.append(self.dictionary.label2idx[label])

            # idss = torch.tensor(np.array(idss))
            labels = torch.tensor(np.array(labels)).long()
            # attention_masks = torch.tensor(np.array(attention_masks))
            # token_type_ids = torch.tensor(np.array(token_type_ids))
            print(idss.shape)
            print(labels.shape)
            print(attention_masks.shape)
            print(token_type_ids.shape)

        return TensorDataset(idss, attention_masks,token_type_ids, labels)
