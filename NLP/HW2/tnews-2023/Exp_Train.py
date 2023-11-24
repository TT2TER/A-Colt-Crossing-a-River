import torch
import torch.nn as nn
import time
import json
import os

from tqdm import tqdm
from torch.utils.data import  DataLoader
from Exp_DataSet import Corpus
from Exp_Model import BiLSTM_model, Transformer_model
import pickle
import matplotlib.pyplot as plt
all_accs = {}
all_epoches = []
def draw():
    #折线图
    x = all_epoches#横坐标
    k1 = all_accs[0]#线1的纵坐标
    k2 = all_accs[1]#线2的纵坐标
    k3 = all_accs[2]#线3的纵坐标
    k4 = all_accs[3]#线4的纵坐标
    k5 = all_accs[4]#线5的纵坐标
    k6 = all_accs[5]#线6的纵坐标
    k7 = all_accs[6]#线7的纵坐标
    plt.plot(x,k1,'s-',color = 'r',label="BLSTM-H_t(Avg)-L+L")
    plt.plot(x,k2,'o-',color = 'g',label="BLSTM-H_t")
    plt.plot(x,k3,'*-',color = 'b',label="BLSTM-H_{0-t}")
    plt.plot(x,k4,'+-',color = 'y',label="BLSTM")
    plt.plot(x,k5,'x-',color = 'c',label="BLSTM-H_{0-t}(Max)")
    plt.plot(x,k6,'d-',color = 'm',label="BLSTM-H_{0-t}(Avg)")
    plt.plot(x,k7,'v-',color = 'k',label="BLSTM-ATTN")#Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification (Zhou et al., ACL 2016)
    plt.xlabel("region length")#横坐标名字
    plt.ylabel("accuracy")#纵坐标名字
    plt.legend(loc = "best")#图例
    plt.savefig("acc.png")
    plt.show()

def train(method):
    '''
    进行训练
    '''
    max_valid_acc = 0
    epoches = []
    valid_accs = []

    for epoch in range(num_epochs):
        model.train()

        total_loss = []
        total_true = []

        tqdm_iterator = tqdm(data_loader_train, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for data in tqdm_iterator:
            # 选取对应批次数据的输入和标签
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            # 模型预测
            y_hat = model(batch_x,method)

            loss = loss_function(y_hat, batch_y)

            optimizer.zero_grad()   # 梯度清零
            loss.backward()         # 计算梯度
            optimizer.step()        # 更新参数

            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)
            
            total_true.append(torch.sum(y_hat == batch_y).item())
            total_loss.append(loss.item())

            tqdm_iterator.set_postfix(loss=sum(total_loss) / len(total_loss),
                                      acc=sum(total_true) / (batch_size * len(total_true)))
        
        tqdm_iterator.close()

        train_loss = sum(total_loss) / len(total_loss)
        train_acc = sum(total_true) / (batch_size * len(total_true))

        valid_acc = valid(method)

        if valid_acc > max_valid_acc:
            torch.save(model, os.path.join(output_folder, "model.ckpt"))

        print(f"epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc*100:.2f}%, valid accuracy: {valid_acc*100:.2f}%")
        epoches.append(epoch)
        valid_accs.append(valid_acc*100)
    all_epoches[:]=epoches
    all_accs[method]=valid_accs


def valid(method):
    '''
    进行验证，返回模型在验证集上的 accuracy
    '''
    total_true = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_valid, dynamic_ncols=True):
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            y_hat = model(batch_x,method)
            # 取分类概率最大的类别作为预测的类别
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat]).to(device)

            total_true.append(torch.sum(y_hat == batch_y).item())

        return sum(total_true) / (batch_size * len(total_true))


def predict(method):
    '''
    读取训练好的模型对测试集进行预测，并生成结果文件
    '''
    test_ids = [] 
    test_pred = []

    model = torch.load(os.path.join(output_folder, "model.ckpt")).to(device)
    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_test, dynamic_ncols=True): 
            batch_x, batch_y = data[0].to(device), data[1]

            y_hat = model(batch_x,method)
            y_hat = torch.tensor([torch.argmax(_) for _ in y_hat])

            test_ids += batch_y.tolist()
            test_pred += y_hat.tolist()

    # 写入文件
    with open(os.path.join(output_folder, "predict.json"), "w") as f:
        for idx, label_idx in enumerate(test_pred):
            one_data = {}
            one_data["id"] = test_ids[idx]
            one_data["pred_label_desc"] = dataset.dictionary.idx2label[label_idx][1]
            json_data = json.dumps(one_data)    # 将字典转为json格式的字符串
            f.write(json_data + "\n")
            

if __name__ == '__main__':
    dataset_folder = './data/tnews_public'
    output_folder = './output'

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('running on device ' + str(device))

    #-----------------------------------------------------begin-----------------------------------------------------#
    # 以下为超参数，可根据需要修改
    embedding_dim = 300     # 每个词向量的维度
    max_token_per_sent = 50 # 每个句子预设的最大 token 数
    batch_size = 64
    num_epochs = 5
    lr = 1e-3
    #------------------------------------------------------end------------------------------------------------------#
    #检查是否有dataset.pkl文件，有则直接读取，没有则生成
    if os.path.exists("./dataset.pkl"):
        with open("./dataset.pkl", "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = Corpus(dataset_folder, max_token_per_sent)
    #将dataset打包为pickle文件，方便下次直接读取
    with open("./dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)


    vocab_size = len(dataset.dictionary.tkn2word)

    data_loader_train = DataLoader(dataset=dataset.train, batch_size=batch_size, shuffle=True)
    data_loader_valid = DataLoader(dataset=dataset.valid, batch_size=batch_size, shuffle=False)
    data_loader_test = DataLoader(dataset=dataset.test, batch_size=batch_size, shuffle=False)
    for i in range(0,7):
        print("method:",i)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 可修改选择的模型以及传入的参数
        model = BiLSTM_model(vocab_size=vocab_size, ntoken=max_token_per_sent, d_emb=embedding_dim,embedding_weight=dataset.embedding_weight).to(device)                            
        #------------------------------------------------------end------------------------------------------------------#
        
        # 设置损失函数
        loss_function = nn.CrossEntropyLoss()
        # 设置优化器                                       
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  
    
        # 进行训练
        train(i)

        # 对测试集进行预测
        predict(i)
    draw()
