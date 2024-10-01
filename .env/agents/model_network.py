#from Env import reward_compute as rc, config
from agents import test_method as tm
import tensorflow as tf
import keras
from spektral.layers import GCNConv# global_sum_pool
from absl import flags
import os
import numpy as np
import time
import datetime

tf.compat.v1.enable_eager_execution()

# by dlhu, 05/2024

class GCN(tf.keras.Model): # nn.Module
    def __init__(self, feature_size, embedding_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feature_size, embedding_size, cached=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = keras.layers.ReLU(self.conv1(x, edge_index, edge_weight))
        x = keras.layers.dropout(x)
        # (num_nodes, embedding_size)
        return x

class TrajEmbedding(tf.keras.Model): # nn.Module
    def __init__(self, feature_size, embedding_size, device):
        super(TrajEmbedding, self).__init__()
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.device = device
        self.gcn = GCN(feature_size, embedding_size).to(self.device)

    def forward(self, network, traj_seqs):
        """
        padding and spatial embedding trajectory with network topology
        :param network: the Pytorch geometric data object 
        :param traj_seqs: list [batch,node_seq] 
        :return: packed_input
        """
        batch_size = len(traj_seqs)
        seq_lengths = list(map(len, traj_seqs))

        for traj_one in traj_seqs:
            traj_one += [0]*(max(seq_lengths)-len(traj_one))

        # prepare sequence tensor
        embedded_seq_tensor = tf.zeros((batch_size, max(seq_lengths), self.embedding_size), dtype=tf.float32) # torch.zeros torch.float32

        seq_lengths = tf.concat(seq_lengths, dtype=tf.int64).to(self.device) # torch.LongTensor
        traj_seqs = tf.Variable(traj_seqs).to(self.device) # torch.tensor

        # get node embeddings from gcn
        # (num_nodes, embedding_size)
        node_embeddings = self.gcn(network)

        # get embedding for trajectory embeddings
        for idx, (seq, seqlen) in enumerate(zip(traj_seqs, seq_lengths)):
            embedded_seq_tensor[idx, :seqlen] = node_embeddings.index_select(0, seq[:seqlen])

        # move to cuda device
        seq_lengths = seq_lengths.cpu()
        embedded_seq_tensor = embedded_seq_tensor.to(self.device)

        # packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths, batch_first=True, enforce_sorted=False)

        return embedded_seq_tensor, seq_lengths

'''
    input: single point
    output: the embedding of single point
'''

class NodeEmbedding(tf.keras.Model): # nn.Module
    def __init__(self, feature_size, embedding_size, device):
        super(NodeEmbedding, self).__init__()
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.device = device
        self.gcn = GCN(feature_size, embedding_size).to(self.device)

    def forward(self, network, traj_seqs):
        batch_size = len(traj_seqs)
        seq_lengths = list(map(len, traj_seqs))

        for traj_one in traj_seqs:
            traj_one += [0] * (max(seq_lengths) - len(traj_one))

        embedded_seq_tensor = tf.zeros((batch_size, max(seq_lengths), self.embedding_size), dtype=tf.float32)

        seq_lengths = tf.concat(seq_lengths, dtype = tf.int64).to(self.device)
        traj_seqs = tf.Variable(traj_seqs).to(self.device)

        node_embeddings = self.gcn(network)

        node_to_embedding_dict = {}
        for i in range(node_embeddings.size(0)):
            node_to_embedding_dict[i] = node_embeddings[i].tolist()

        # 获取结点嵌入
        for idx, traj in enumerate(traj_seqs):
            for j, point in enumerate(traj[:seq_lengths[idx]]):
                # embedded_seq_tensor[idx, j] = node_embeddings.index_select(0,torch.tensor([j]).to(self.device))
                embedded_seq_tensor[idx,j] = tf.Variable(node_to_embedding_dict[int(point)]).to(self.device)

        seq_lengths = seq_lengths.cpu()
        embedded_seq_tensor = embedded_seq_tensor.to(self.device)

        return embedded_seq_tensor, seq_lengths

class TimeEmbedding(tf.keras.Model): # nn.Module
    def __init__(self, date2vec_size, device):
        super(TimeEmbedding, self).__init__()
        self.device = device
        self.date2vec_size = date2vec_size

    def forward(self, time_seqs):
        """
        padding and timestamp series embedding
        :param time_seqs: list [batch,timestamp_seq]
        :return: packed_input
        """
        batch_size = len(time_seqs)
        seq_lengths = list(map(len, time_seqs))

        for time_one in time_seqs:
            time_one += [[0 for i in range(self.date2vec_size)]]*(max(seq_lengths)-len(time_one))

        # vec_time_seqs = self.d2vec(time_seqs).to(self.device)

        # prepare sequence tensor
        embedded_seq_tensor = tf.zeros((batch_size, max(seq_lengths), self.date2vec_size), dtype=tf.float32)

        seq_lengths = tf.concat(seq_lengths, dtype = tf.int64).to(self.device)
        # time_seqs = torch.tensor(time_seqs).to(self.device)
        vec_time_seqs = tf.Variable(time_seqs).to(self.device)

        # get embedding for trajectory embeddings
        for idx, (seq, seqlen) in enumerate(zip(vec_time_seqs, seq_lengths)):
            embedded_seq_tensor[idx, :seqlen] = seq[:seqlen]

        # move to cuda device
        seq_lengths = seq_lengths.cpu()
        embedded_seq_tensor = embedded_seq_tensor.to(self.device)

        # packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths, batch_first=True,enforce_sorted=False)
        return embedded_seq_tensor


class ST_LSTM(tf.keras.Model): # nn.Module
    def __init__(self, embedding_size, hidden_size, num_layers, dropout_rate, device):
        super(ST_LSTM, self).__init__()
        self.device = device
        self.bi_lstm = tf.keras.layers.LSTM(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout_rate,
                               bidirectional=True) #nn.LSTM
        # self-attention weights
        self.w_omega = keras.layers.Parameter(tf.Variable(hidden_size * 2, hidden_size * 2)) # nn.Parameter
        self.u_omega = keras.layers.Parameter(tf.Variable(hidden_size * 2, 1)) # nn.Parameter

        tf.random.uniform(self.w_omega, -0.1, 0.1) # nn.init.uniform_
        tf.random.uniform(self.u_omega, -0.1, 0.1) # nn.init.uniform_
        self.embedding_size = embedding_size

    def getMask(self, seq_lengths):
        """
        create mask based on the sentence lengths
        :param seq_lengths: sequence length after `pad_packed_sequence`
        :return: mask (batch_size, max_seq_len)
        """
        max_len = int(seq_lengths.max())

        # (batch_size, max_seq_len)
        mask = tf.ones((seq_lengths.size()[0], max_len)).to(self.device) # torch.ones

        for i, l in enumerate(seq_lengths):
            if l < max_len:
                mask[i, l:] = 0

        return mask

    def forward(self, packed_input):
        # output features (h_t) from the last layer of the LSTM, for each t
        # (batch_size, seq_len, 2 * num_hiddens)
        packed_output, _ = self.bi_lstm(packed_input)  # output, (h, c)
        outputs, seq_lengths = tf.pad(packed_output, [self.embedding_size,1])  #pad_packed_sequence(packed_output, batch_first=True) # \dlhu需要确认修改后功能不变

        # get sequence mask
        mask = self.getMask(seq_lengths)

        # Attention...
        # (batch_size, seq_len, 2 * num_hiddens)
        u = tf.tanh(tf.matmul(outputs, self.w_omega)) # tf.tanh torch.matul
        # (batch_size, seq_len)
        att =tf.matmul(u, self.u_omega).squeeze() # tf.matul

        # add mask
        att = att.masked_fill(mask == 0, -1e10)

        # (batch_size, seq_len,1)
        att_score = tf.nn.softmax(att, dim=1).unsqueeze(2) # F.softmax
        # normalization attention weight
        # (batch_size, seq_len, 2 * num_hiddens)
        scored_outputs = outputs * att_score

        # weighted sum as output
        # (batch_size, 2 * num_hiddens)
        out = tf.sum(scored_outputs, dim=1) # tf.sum
        return out
       

class ST_Encoder(tf.keras.Model): # nn.Module
    def __init__(self, feature_size, date2vec_size, embedding_size, hidden_size,
                                    num_layers, dropout_rate, device):
        super(ST_Encoder, self).__init__()
        self.embedding_S = TrajEmbedding(feature_size, embedding_size, device)
        self.embedding_T = TimeEmbedding(date2vec_size, device)
        self.encoder_ST = ST_LSTM(embedding_size+date2vec_size, hidden_size, num_layers, dropout_rate, device)

    def forward(self, network, traj_seqs, time_seqs):
        s_input, seq_lengths = self.embedding_S(network, traj_seqs)
        t_input = self.embedding_T(time_seqs)

        st_input = tf.concat((s_input, t_input), dim=2) # torch.cat

        #packed_input = pack_padded_sequence(st_input, seq_lengths, batch_first=True, enforce_sorted=False)   #\dlhu tensorflow中的LSTM层可以自行处理变长输入序列，所以我把这里删了，下同
        #att_output = self.encoder_ST(packed_input)
        
        att_output = self.encoder_ST(st_input)

        return att_output

class ST_Encoder2(tf.keras.Model): # nn.Module
    def __init__(self, feature_size, date2vec_size, embedding_size, hidden_size,
                                    num_layers, dropout_rate, device):
        super(ST_Encoder2, self).__init__()
        self.embedding_S = NodeEmbedding(feature_size, embedding_size, device)
        self.embedding_T = TimeEmbedding(date2vec_size, device)
        self.encoder_ST = ST_LSTM(embedding_size+date2vec_size, hidden_size, num_layers, dropout_rate, device)

    def forward(self, network, traj_seqs, time_seqs):
        s_input, seq_lengths = self.embedding_S(network, traj_seqs)
        t_input = self.embedding_T(time_seqs)

        st_input = tf.concat((s_input, t_input), dim=2) # torch.cat

        #packed_input = pack_padded_sequence(st_input, seq_lengths, batch_first=True, enforce_sorted=False)
        #att_output = self.encoder_ST(packed_input)
        att_output = self.encoder_ST(st_input)

        return att_output

class STTrajSimEncoder(tf.keras.Model): # nn.Module
    def __init__(self, feature_size, embedding_size, date2vec_size, hidden_size, num_layers, dropout_rate, concat, device):
        super(STTrajSimEncoder, self).__init__()
        self.stEncoder = ST_Encoder(feature_size, date2vec_size, embedding_size, hidden_size,
                                    num_layers, dropout_rate, device)
        self.concat = concat

    def forward(self, network, traj_seqs, time_seqs):
        """
        :param network: the Pytorch geometric data object
        :param traj_seqs: list [batch,node_seq]
        :param time_seqs: list [batch,timestamp_seq]
        :return: the Spatio-Temporal embedding of  trajectory
        """

        st_emb = self.stEncoder(network, traj_seqs, time_seqs)
        return st_emb