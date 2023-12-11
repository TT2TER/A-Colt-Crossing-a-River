import torch.nn as nn
import torch as torch
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=512, d_hid=80, nhead=15, nlayers=2, dropout=0.2, embedding_weight=None):
        super(Transformer_model, self).__init__()
        self.vocab_size = vocab_size
        self.ntoken = ntoken
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.pos_encoder = PositionalEncoding(d_model=d_emb, max_len=ntoken,dropout=dropout)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=d_emb, nhead=nhead, dim_feedforward=d_hid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=nlayers)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 transformer 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选

        # 请自行设计分类器
        # 使用线性分类器
        self.classify__1 = nn.Sequential(
            # nn.Linear(d_emb, 64),
            # nn.ReLU(),
            # nn.Linear(64, 15),
            nn.Linear(d_emb, 15)
        )
        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x,method):
        x = self.embed(x)     
        x = x.permute(1, 0, 2)          
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 transformer_encoder 的隐藏层输出进行处理和选择，并完成分类
        x = self.dropout(x) # 可选
        if method == -1:
            print("please choose a method")
            return
        if method == 0:
            # print(x.shape)
            x=torch.sum(x,dim=1)
        if method == 1:
            x=torch.sum(x,dim=1)/x.size(1)
        if method == 2:
            x= x[:,-1,:]
        if method == 4:
            #不太对
            m =nn.MaxPool1d(self.ntoken)
            print(x.shape)
            x= m(x.permute(0,2,1)).squeeze(2)
            print(x.shape)
        x = self.classify__1(x)
        # print(x.shape)
        #------------------------------------------------------end------------------------------------------------------#
        return x
    
    
class BiLSTM_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=100, d_hid=80, nlayers=4, dropout=0.4, embedding_weight=None):
        super(BiLSTM_model, self).__init__()
        self.ntoken = ntoken
        self.d_emb = d_emb
        self.d_hid = d_hid
        self.nlayers = nlayers
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.lstm = nn.LSTM(input_size=d_emb, hidden_size=d_hid, num_layers=nlayers, bidirectional=True, batch_first=True,dropout=dropout)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 bilstm 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选

        # 请自行设计分类器
        # 使用线性分类器
        self.classify__1 = nn.Sequential(
            nn.Linear(self.nlayers, 256),
            nn.ReLU(),
            nn.Linear(256, 15),
            nn.LogSoftmax(dim=1)
            # nn.Linear(self.nlayers, 15)
        )
        self.classify_0 = nn.Sequential(
            # nn.Linear(self.nlayers*2, 256),
            # nn.ReLU(),
            # nn.Linear(256, 15),
            # nn.LogSoftmax(dim=1)
            nn.Linear(self.nlayers*2, 15),
        )
        self.classify_1 = nn.Sequential(
            # nn.Linear(self.nlayers*self.d_hid*2, 256),
            # nn.ReLU(),
            # # nn.Dropout(dropout),
            # nn.Linear(256, 15),
            # nn.LogSoftmax(dim=1)
            nn.Linear(self.nlayers*self.d_hid*2, 15),
        )
        self.classify_2 = nn.Sequential(
            # nn.Linear(self.ntoken*self.d_hid*2, 256),
            # nn.ReLU(),
            # nn.Linear(256, 15),
            # nn.LogSoftmax(dim=0)
            nn.Linear(self.ntoken*self.d_hid*2, 15),
        )
        self.classify_3 = nn.Sequential(
            # nn.Linear(54, 256),
            # nn.ReLU(),
            # nn.Linear(256, 15),
            # nn.LogSoftmax(dim=0)
            nn.Linear(54, 15)
        )
        self.classify_4 = nn.Sequential(
            nn.Linear(self.d_hid*2, 15)
        )


    def attention_net(self, x, query, mask=None): 
        
        d_k = query.size(-1)     # d_k为query的维度
       
        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
#         print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        # 打分机制 scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  
#         print("score: ", scores.shape)  # torch.Size([128, 38, 38])
        
        # 对最后一个维度 归一化得分
        alpha_n = F.softmax(scores, dim=-1) 
#         print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
        # 对权重化的x求和
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        context = torch.matmul(alpha_n, x).sum(1)
        
        return context, alpha_n
        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, input, method=-1):
        self.input_num=input.shape[0]
        x = self.embed(input)
        output,(h_n,c_n) = self.lstm(x)
        # print(output.shape)
        if method == -1:
            print("please choose a method")
            return
        
        #-----------------------------------------------------begin-----------------------------------------------------#
        #method-1:同层隐藏输出avg，层间拼接
        if method == 0:
            forward_hidden = h_n[0,:,-1].unsqueeze(1)
            backward_hidden = h_n[1,:,0].unsqueeze(1)
            layer_hidden = torch.cat((forward_hidden,backward_hidden),dim=1)
            m=nn.AvgPool1d(2)
            combine_hidden = m(layer_hidden).squeeze(0)
            for i in range(self.nlayers-1):
                forward_hidden = h_n[2*i,:,-1].unsqueeze(1)
                backward_hidden = h_n[2*i+1,:,0].unsqueeze(1)
                layer_hidden = torch.cat((forward_hidden,backward_hidden),dim=1)
                m=nn.AvgPool1d(2)
                layer_hidden = m(layer_hidden).squeeze(0)
                combine_hidden = torch.cat((combine_hidden,layer_hidden),dim=1)
            ans = self.classify__1(combine_hidden)
        #method0:直接将每一层最后的一个隐藏结点进行拼接，最终拼接为[(batch_size,nlayers*2)]
        if method == 1:
            forward_hidden = h_n[0,:,-1].unsqueeze(1)
            # print(forward_hidden)
            backward_hidden = h_n[1,:,0].unsqueeze(1)
            for i in range(self.nlayers-1):
                forward_hidden = torch.cat((forward_hidden,h_n[2*i+2,:,-1].unsqueeze(1)),dim=1)
                backward_hidden = torch.cat((backward_hidden,h_n[2*i+3,:,0].unsqueeze(1)),dim=1)
            combine_hidden = torch.cat((forward_hidden,backward_hidden),dim=1)
            ans = self.classify_0(combine_hidden)
        #method1:
        #将每一层的正向和反向的隐藏层进行拼接，最终拼接为[(batch_size,nlayers*2*hidden_size)]
        if method == 2:
            forward_hidden = h_n[0]
            backward_hidden = h_n[1]
            for i in range(self.nlayers-1):
                forward_hidden = torch.cat((forward_hidden,h_n[2*i+2]),dim=1)
                backward_hidden = torch.cat((backward_hidden,h_n[2*i+3]),dim=1)
            combine_hidden = torch.cat((forward_hidden,backward_hidden),dim=1)
            ans = self.classify_1(combine_hidden)
        #method2:直接将输出线性相加
        # 对 bilstm 的隐藏层输出进行处理和选择，并完成分类
        if method == 3:
            output = self.dropout(output).reshape(-1, self.ntoken*self.d_hid*2)   # ntoken*nhid*2 (2 means bidirectional)，取得所有的输出线性相加
            ans = self.classify_2(output)
        
        #method3:将每一层的正向和反向的隐藏层进行最大池化pooling
        if method == 4:
            m1 = nn.MaxPool1d(2,3)
            forward_hidden = m1(h_n)
            backward_hidden = m1(h_n)
            m2 = nn.MaxPool1d(8)
            forward_hidden = m2(forward_hidden.permute(1,2,0)).permute(2,0,1).squeeze(0)
            backward_hidden = m2(backward_hidden.permute(1,2,0)).permute(2,0,1).squeeze(0)
            combine_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
            
            ans = self.classify_3(combine_hidden)
        
        #method4:将每一层的正向和反向的隐藏层进行平均pooling
        if method == 5:
            m1 = nn.AvgPool1d(2,3)
            forward_hidden = m1(h_n)
            backward_hidden = m1(h_n)
            m2 = nn.AvgPool1d(8)
            forward_hidden = m2(forward_hidden.permute(1,2,0)).permute(2,0,1).squeeze(0)
            backward_hidden = m2(backward_hidden.permute(1,2,0)).permute(2,0,1).squeeze(0)
            combine_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
            
            ans = self.classify_3(combine_hidden)

        #method5:Bi-LSTM+Attention
        if method == 6:
            # print(output.shape)
            # output=output.permute(1,0,2)
            query = self.dropout(output)
            attn_output,_ = self.attention_net(output,query)
            ans=self.classify_4(attn_output)
        #------------------------------------------------------end------------------------------------------------------#
        return ans

class BERT_model(nn.Module):
    def __init__(self, pretrained):
        super(BERT_model, self).__init__()
        self.bert = pretrained
        self.classify = nn.Sequential(
            nn.Linear(768, 15),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, method=-1):
        if method == -1:
            print("please choose a method")
            return
        with torch.no_grad():
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if method == 0 or method ==2 or method ==4:
            # print(output)
            output = output.last_hidden_state[:, 0]
            # print(output.shape)
        if method ==1 or method ==3 or method ==5:
            output = output.last_hidden_state
            # 将output的第二个维度进行平均
            output = torch.mean(output, dim=1)
            # print(output.shape)
        # if method ==4:
        #     # print(output.pooler_output)
        #     # print(output.last_hidden_state.shape)
        #     output = output.pooler_output
        #     # print(output)
        #     # output = torch.mean(output, dim=1)
        output = self.classify(output)
        return output