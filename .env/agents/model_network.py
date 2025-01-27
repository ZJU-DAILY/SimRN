import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import GCNConv, MessagePassing
import time
import datetime
import numpy as np
import torch.distributed as dis


class GCN(nn.Module):
    def __init__(self, feature_size, embedding_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feature_size, embedding_size, cached=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        # (num_nodes, embedding_size)
        return x

class TrajEmbedding(nn.Module):
    def __init__(self, feature_size, embedding_size, device):
        super(TrajEmbedding, self).__init__()
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.device = device
        self.gcn = GCN(feature_size, embedding_size).to(self.device)

    def forward(self, network, traj_seqs):
        """
        padding and spatial embedding trajectory with network topology
        :param network: the Pytorch geometric data object 网络图结构
        :param traj_seqs: list [batch,node_seq] 轨迹序列
        :return: packed_input
        """
        batch_size = len(traj_seqs)
        seq_lengths = list(map(len, traj_seqs))

        for traj_one in traj_seqs:
            traj_one += [0] * (max(seq_lengths) - len(traj_one))

        # prepare sequence tensor
        embedded_seq_tensor = torch.zeros((batch_size, max(seq_lengths), self.embedding_size), dtype=torch.float32)

        seq_lengths = torch.LongTensor(seq_lengths).to(self.device)
        traj_seqs = torch.tensor(traj_seqs).to(self.device)

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


class TimeEmbedding(nn.Module):
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
            time_one += [[0 for i in range(self.date2vec_size)]] * (max(seq_lengths) - len(time_one))

        # vec_time_seqs = self.d2vec(time_seqs).to(self.device)

        # prepare sequence tensor
        embedded_seq_tensor = torch.zeros((batch_size, max(seq_lengths), self.date2vec_size), dtype=torch.float32)

        seq_lengths = torch.LongTensor(seq_lengths).to(self.device)
        # time_seqs = torch.tensor(time_seqs).to(self.device)
        vec_time_seqs = torch.tensor(time_seqs).to(self.device)

        # get embedding for trajectory embeddings
        for idx, (seq, seqlen) in enumerate(zip(vec_time_seqs, seq_lengths)):
            embedded_seq_tensor[idx, :seqlen] = seq[:seqlen]

        # move to cuda device
        seq_lengths = seq_lengths.cpu()
        embedded_seq_tensor = embedded_seq_tensor.to(self.device)

        # packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths, batch_first=True,enforce_sorted=False)
        return embedded_seq_tensor


class ST_LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, dropout_rate, device):
        super(ST_LSTM, self).__init__()
        self.device = device
        self.bi_lstm = nn.LSTM(input_size=embedding_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout_rate,
                              bidirectional=True)
        # self-attention weights
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def getMask(self, seq_lengths):
        """
        create mask based on the sentence lengths
        :param seq_lengths: sequence length after `pad_packed_sequence`
        :return: mask (batch_size, max_seq_len)
        """
        max_len = int(seq_lengths.max())
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)
        for i, l in enumerate(seq_lengths):
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def to(self, device):
        self.device = device
        self.bi_lstm = self.bi_lstm.to(device)
        self.w_omega = nn.Parameter(self.w_omega.to(device))
        self.u_omega = nn.Parameter(self.u_omega.to(device))
        return super().to(device)

    def forward(self, packed_input):
        # if str(packed_input.data.device) != str(self.device):
        #     packed_input = packed_input.to(self.device)
        
        # for name, param in self.bi_lstm.named_parameters():
        #     print(f"Parameter {name} is on device: {param.device}")
            
        # output features (h_t) from the last layer of the LSTM, for each t
        packed_output, _ = self.bi_lstm(packed_input)  # output, (h, c)
        outputs, seq_lengths = pad_packed_sequence(packed_output, batch_first=True)

        # get sequence mask
        mask = self.getMask(seq_lengths)

        # Attention...
        u = torch.tanh(torch.matmul(outputs, self.w_omega))
        att = torch.matmul(u, self.u_omega).squeeze()

        # add mask
        att = att.masked_fill(mask == 0, -1e10)

        # normalization attention weight
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        scored_outputs = outputs * att_score

        # weighted sum as output
        out = torch.sum(scored_outputs, dim=1)
        return out


class ST_Encoder(nn.Module):
    def __init__(self, feature_size, date2vec_size, embedding_size, hidden_size,
                 num_layers, dropout_rate, device):
        super(ST_Encoder, self).__init__()
        self.embedding_S = TrajEmbedding(feature_size, embedding_size, device)
        self.embedding_T = TimeEmbedding(date2vec_size, device)
        self.encoder_ST = ST_LSTM(embedding_size + date2vec_size, hidden_size, num_layers, dropout_rate, device)

    def forward(self, network, traj_seqs, time_seqs):
        s_input, seq_lengths = self.embedding_S(network, traj_seqs)
        t_input = self.embedding_T(time_seqs)

        st_input = torch.cat((s_input, t_input), dim=2)

        packed_input = pack_padded_sequence(st_input, seq_lengths, batch_first=True, enforce_sorted=False)

        att_output = self.encoder_ST(packed_input)

        return att_output


class ModelPartitioner:
    def __init__(self):
        self.computation_weights = {
            'traj_embedding': 1.2,
            'time_embedding': 0.8,
            'st_lstm': 1.5 
        }
    
    def estimate_layer_cost(self, layer_name: str, params_count: int) -> float:
        """Computation Cost"""
        return params_count * self.computation_weights.get(layer_name, 1.0)
    
    def estimate_communication_cost(self, output_size: int) -> float:
        """Communication Cost"""
        return output_size * 0.1  # Communication cost parameter
    
    def partition_network(self, layers_info: list) -> dict:
        """Distributed Partition"""
        n = len(layers_info)
        
        # dp[i][j]: 
        dp = [[float('inf')] * 2 for _ in range(n + 1)]
        split_point = [0] * (n + 1)
        
        dp[0][0] = 0
        
        compute_costs = []
        for layer in layers_info:
            cost = self.estimate_layer_cost(
                layer['name'], 
                layer['params_count']
            )
            compute_costs.append(cost)
        
        for i in range(1, n + 1):
            gpu0_cost = sum(compute_costs[:i])
            dp[i][0] = gpu0_cost
            
            for k in range(1, i):
                comm_cost = self.estimate_communication_cost(
                    layers_info[k-1]['output_size']
                )
                
                total_cost = dp[k][0] + sum(compute_costs[k:i]) + comm_cost
                
                if total_cost < dp[i][1]:
                    dp[i][1] = total_cost
                    split_point[i] = k
        
        k = split_point
        return {
            'cuda:0': [layer['name'] for layer in layers_info[:k[0]]],
            'cuda:1': [layer['name'] for layer in layers_info[k[0]:]]，
	  'cuda:2': [layer['name'] for layer in layers_info[:k[1]]],
            'cuda:3': [layer['name'] for layer in layers_info[k[1]:]]
        }

class STTrajSimEncoder(nn.Module):
    def __init__(self, feature_size, embedding_size, date2vec_size, hidden_size, 
                 num_layers, dropout_rate, concat, device):
        super(STTrajSimEncoder, self).__init__()
        self.comm_monitor = CommunicationMonitor()
        
        self.layers = {
            'traj_embedding': TrajEmbedding(feature_size, embedding_size, device),
            'time_embedding': TimeEmbedding(date2vec_size, device),
            'st_lstm': ST_LSTM(embedding_size + date2vec_size, hidden_size, 
                              num_layers, dropout_rate, device)
        }
        
        layers_info = [
            {
                'name': 'traj_embedding',
                'params_count': sum(p.numel() for p in self.layers['traj_embedding'].parameters()),
                'output_size': embedding_size
            },
            {
                'name': 'time_embedding',
                'params_count': sum(p.numel() for p in self.layers['time_embedding'].parameters()),
                'output_size': date2vec_size
            },
            {
                'name': 'st_lstm',
                'params_count': sum(p.numel() for p in self.layers['st_lstm'].parameters()),
                'output_size': hidden_size * 2
            }
        ]
        
        partitioner = ModelPartitioner()
        self.partition = partitioner.partition_network(layers_info)
        
        for name in self.partition['cuda:0']:
            self.layers[name].to('cuda:0')
            if hasattr(self.layers[name], 'device'):
                self.layers[name].device = 'cuda:0'

        for name in self.partition['cuda:1']:
            self.layers[name].to('cuda:1')
            if hasattr(self.layers[name], 'device'):
                self.layers[name].device = 'cuda:1'
	
        for name in self.partition['cuda:2']:
            self.layers[name].to('cuda:2')
            if hasattr(self.layers[name], 'device'):
                self.layers[name].device = 'cuda:2'

       for name in self.partition['cuda:3']:
            self.layers[name].to('cuda:3')
            if hasattr(self.layers[name], 'device'):
                self.layers[name].device = 'cuda:3'
        
        self.gpu0_layers = nn.ModuleDict({
            name: self.layers[name] for name in self.partition['cuda:0']
        })
        
        self.gpu1_layers = nn.ModuleDict({
            name: self.layers[name] for name in self.partition['cuda:1']
        })

        self.gpu2_layers = nn.ModuleDict({
            name: self.layers[name] for name in self.partition['cuda:2']
        })
        
        self.gpu3_layers = nn.ModuleDict({
            name: self.layers[name] for name in self.partition['cuda:3']
        })
            
        self.concat = concat
        print("Network partition:", self.partition)
        
    def forward(self, network, traj_seqs, time_seqs):
        s_device = 'cuda:0'
        t_device = 'cuda:0'
        
        s_input = None
        t_input = None
        seq_lengths = None
        
        if 'traj_embedding' in self.gpu0_layers:
            network = network.to('cuda:0')
            # s_input, seq_lengths = self.gpu0_layers['traj_embedding'](network, traj_seqs)
            s_input, seq_lengths = self.layers['traj_embedding'](network, traj_seqs)
        else if 'traj_embedding' in self.gpu1_layers:  
            network = network.to('cuda:1')
            s_input, seq_lengths = self.layers['traj_embedding'].to('cuda:1')(network, traj_seqs)
            s_device = 'cuda:1'
        else if 'traj_embedding' in self.gpu2_layers:  
            network = network.to('cuda:2')
            s_input, seq_lengths = self.layers['traj_embedding'].to('cuda:2')(network, traj_seqs)
            s_device = 'cuda:2'
        else if 'traj_embedding' in self.gpu3_layers:  
            network = network.to('cuda:3')
            s_input, seq_lengths = self.layers['traj_embedding'].to('cuda:3')(network, traj_seqs)
            s_device = 'cuda:3'

        if 'time_embedding' in self.gpu0_layers:
            t_input = self.layers['time_embedding'](time_seqs)
        else if 'time_embedding' in self.gpu1_layers:  
            t_input = self.layers['time_embedding'].to('cuda:1')(time_seqs)
            t_device = 'cuda:1'
        else if 'time_embedding' in self.gpu2_layers:  
            t_input = self.layers['time_embedding'].to('cuda:2')(time_seqs)
            t_device = 'cuda:2'
        else if 'time_embedding' in self.gpu3_layers:  
            t_input = self.layers['time_embedding'].to('cuda:3')(time_seqs)
            t_device = 'cuda:3'
        
        # assert s_input is not None and t_input is not None, "Both spatial and temporal inputs must be computed"
        
        if 'st_lstm' in self.gpu0_layers: st_device = 'cuda:0'
        else if 'st_lstm' in self.gpu1_layers: st_device = 'cuda:1'
        else if 'st_lstm' in self.gpu2_layers: st_device = 'cuda:2'
        else if 'st_lstm' in self.gpu3_layers: st_device = 'cuda:3'
        if st_device != s_device:
            # print('s need to move to', st_device)
            self.comm_monitor.start_transfer()
            s_input = s_input.to(st_device)
	 dis.all_reduce(tensor, op=dis.ReduceOp.SUM)
            self.comm_monitor.record("all_reduce", s_input.element_size() * s_input.nelement())
            self.comm_monitor.end_transfer(s_input.numel())

        if st_device != t_device:
            # print('t need to move to', st_device)
            self.comm_monitor.start_transfer()
	  t_input = t_input.to(st_device)
	  dis.all_reduce(tensor, op=dis.ReduceOp.SUM)
            self.comm_monitor.record("all_reduce", t_input.element_size() * t_input.nelement())
            self.comm_monitor.end_transfer(t_input.numel())

        st_input = torch.cat((s_input, t_input), dim=2)
        packed_input = pack_padded_sequence(st_input, seq_lengths, 
                                          batch_first=True, enforce_sorted=False)
        

        output = self.layers['st_lstm'].to(st_device)(packed_input)
        
        return output
        
    def get_communication_stats(self):
        """Communication Cost Computing"""
        total_transfer = sum(t['size'] for t in self.comm_monitor.transfer_sizes)
        avg_speed = sum(t['speed'] for t in self.comm_monitor.transfer_sizes) / len(self.comm_monitor.transfer_sizes)
        return {
            'total_transfer_mb': total_transfer,
            'avg_speed_mbs': avg_speed,
            'transfer_details': self.comm_monitor.transfer_sizes
        }
    

class CommunicationMonitor:
    def __init__(self):
        self.start_time = None
        self.transfer_sizes = []
    
   def record(self, op_name, bytes_sent):
        if self.start_time is None:
            raise RuntimeError("CommunicationMonitor is not started. Call start() first.")

        end_time = time.time()
        duration = end_time - self.start_time

        self.communication_stats[op_name]["count"] += 1
        self.communication_stats[op_name]["total_time"] += duration
        self.communication_stats[op_name]["total_bytes"] += bytes_sent
        self.start_time = time.time()

    def start_transfer(self):
        torch.cuda.synchronize()
        self.start_time = time.time()
        
    def end_transfer(self, tensor_size):
        torch.cuda.synchronize()
        transfer_time = time.time() - self.start_time
        transfer_speed = tensor_size * 4 / (transfer_time * 1024 * 1024)  # MB/s
        self.transfer_sizes.append({
            'size': tensor_size * 4 / (1024 * 1024),  # MB
            'time': transfer_time,
            'speed': transfer_speed
        })