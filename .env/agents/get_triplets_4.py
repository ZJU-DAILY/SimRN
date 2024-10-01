# by dlhu, 05/2024
# Generate the multiple p/n samples and get the triplets

import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
from torch_geometric.data import Data
from agents.traj_preprocess_3 import Date2vec
import agents.spatial_similarity as spatial_com
import agents.temporal_similarity as temporal_com
import random
import pickle
import yaml

random.seed(1933)
config = yaml.load(open('config.yaml'))
period = config["period"]

# load the road network
def load_netowrk(dataset):
    """
    load road network
    :param dataset: the city name of road network
    :return: road network graph
    """
    # change the path for new road network
    edge_path = "./data/" + dataset + "/road/edge_weight_p.csv" # edge_weight_direction_interval1.csv
    node_embedding_path = "./data/" + dataset + "/node_features.npy"

    node_embeddings = np.load(node_embedding_path)
    df_edge = pd.read_csv(edge_path, sep=',')

    edge_index = df_edge[["s_node", "e_node"]].to_numpy()

    edge_attr = df_edge["length"].to_numpy()

    edge_index = tf.constant(edge_index).t().contiguous() # torch.LongTensor
    node_embeddings = tf.Variable(node_embeddings, dtype=tf.float32) # torch.tensor torch.float
    edge_attr = tf.Variable(edge_attr, dtype=tf.float32) # torch.tensor torch.float

    print("node embeddings shape: ", node_embeddings.shape)
    print("edge_index shap: ", edge_index.shape)
    print("edge_attr shape: ", edge_attr.shape)

    road_network = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)

    return road_network

# def load_network():
#     dataset = str(config["dataset"])
#     rdnetwork = pd.read_csv('./data/{}/road/edge_weight.csv'.format(dataset),
#                             usecols=['section_id', 's_node', 'e_node', 'length'])

#     roadnetwork = nx.DiGraph()
#     for row in rdnetwork.values:
#         roadnetwork.add_edge(int(row[1]), int(row[2]), distance=row[-1])

#     return roadnetwork
# graph = load_network()

# load the data

class DataLoader():
    def __init__(self):
        self.kseg = config["kseg"]
        self.train_set = 2000
        self.vali_set = 2800
        self.test_set = 6000
        self.d2vec = Date2vec()

    def load(self, load_part):
        # split train, vali, test set
        node_list_int = np.load(str(config["shuffle_node_file"]), allow_pickle=True)
        time_list_int = np.load(str(config["shuffle_time_file"]), allow_pickle=True)
        d2vec_list_int = np.load(str(config["shuffle_d2vec_file"]), allow_pickle=True)

        train_set = self.train_set
        vali_set = self.vali_set
        test_set = self.test_set

        if load_part=='train':
            return node_list_int[:train_set], time_list_int[:train_set], d2vec_list_int[:train_set]
        if load_part=='vali':
            return node_list_int[train_set:vali_set], time_list_int[train_set:vali_set], d2vec_list_int[train_set:vali_set]
        if load_part=='test':
            return node_list_int[vali_set:test_set], time_list_int[vali_set:test_set], d2vec_list_int[vali_set:test_set]

    def ksegment_ST(self):
        # Simplify the trajectory
        kseg_coor_trajs = np.load(str(config["shuffle_kseg_file"]), allow_pickle=True)[:self.train_set]
        time_trajs = np.load(str(config["shuffle_time_file"]), allow_pickle=True)[:self.train_set]

        kseg_time_trajs = []
        for t in time_trajs:
            kseg_time = []
            seg = len(t) // self.kseg
            t = np.array(t)
            for i in range(self.kseg):
                if i == self.kseg - 1:
                    kseg_time.append(np.mean(t[i * seg:]))
                else:
                    kseg_time.append(np.mean(t[i * seg:i * seg + seg]))
            kseg_time_trajs.append(kseg_time)
        kseg_time_trajs = np.array(kseg_time_trajs)
        print(kseg_time_trajs.shape)
        print(kseg_time_trajs[0])

        max_lat = 0
        max_lon = 0
        for traj in kseg_coor_trajs:
            for t in traj:
                if max_lat<t[0]:
                    max_lat = t[0]
                if max_lon<t[1]:
                    max_lon = t[1]
        kseg_coor_trajs = kseg_coor_trajs/[max_lat,max_lon]
        kseg_coor_trajs = kseg_coor_trajs.reshape(-1,self.kseg*2)
        kseg_time_trajs = kseg_time_trajs/np.max(kseg_time_trajs)
        
        kseg_ST = np.concatenate((kseg_coor_trajs, kseg_time_trajs), axis=1)
        print("kseg_ST len: ", len(kseg_ST))
        print("kseg_ST shape: ", kseg_ST.shape)

        return kseg_ST
    
    def generate_partial(self, traj, length):
        if len(traj) < length:
            return -1,-1
        start_index = random.randint(0, len(traj) - length)
        return start_index, traj[start_index:start_index+length]

    # generate positive/negative samples
    def get_triplets(self):
        train_node_list, train_time_list, train_d2vec_list = self.load(load_part='train')

        anchor_index = list(range(len(train_node_list)))
        random.shuffle(anchor_index)

        apn_node_triplets = []
        apn_time_triplets = []
        apn_d2vec_triplets = []

        # for j in range(1):
        for j in range(10):
            for i in anchor_index:

                a_sample = train_node_list[i]  # anchor sample
                sub_len = int(len(a_sample) * 2 / 3)
                p_start_index_list = []
                p_sample_list = []
                n_index_list = []
                n_start_index_list = []
                n_sample_list = []
                # the number of pairs
                for k in range(3):
                    # positive
                    p_start_index, p_sample = self.generate_partial(a_sample, sub_len)
                    p_start_index_list.append(p_start_index)
                    p_sample_list.append(p_sample)
                    # negative
                    n_index = random.randint(0, len(train_node_list) - 1)
                    n_start_index, n_sample = self.generate_partial(train_node_list[n_index], sub_len)  # negative sample
                    while n_start_index == -1:
                        n_index = random.randint(0, len(train_node_list) - 1)
                        n_start_index, n_sample = self.generate_partial(train_node_list[n_index],sub_len)  # negative sample
                    n_index_list.append(n_index)
                    n_start_index_list.append(n_start_index)
                    n_sample_list.append(n_sample)

                ok = True
                if str(config["distance_type"]) == "TP":
                    if spatial_com.TP_dis(a_sample, p_sample) == -1 or spatial_com.TP_dis(a_sample, n_sample) == -1:
                        ok = False
                elif str(config["distance_type"]) == "DITA":
                    if spatial_com.DITA_dis(a_sample, p_sample) == -1 or spatial_com.DITA_dis(a_sample, n_sample) == -1:
                        ok = False
                elif str(config["distance_type"]) == "discret_frechet":
                    if spatial_com.frechet_dis(a_sample, p_sample) == -1 or spatial_com.frechet_dis(a_sample,
                                                                                                    n_sample) == -1:
                        ok = False
                elif str(config["distance_type"]) == "LCRS":
                    if spatial_com.LCRS_dis(a_sample,
                                            p_sample) == spatial_com.longest_traj_len * 2 or temporal_com.LCRS_dis(
                            a_sample, p_sample) == temporal_com.longest_trajtime_len * 2:
                        ok = False
                elif str(config["distance_type"]) == "NetERP":
                    if spatial_com.NetERP_dis(a_sample, p_sample) == -1 or spatial_com.NetERP_dis(a_sample,
                                                                                                  n_sample) == -1:
                        ok = False

                if ok:
                    apn_node_triplets.append([a_sample, p_sample_list, n_sample_list])  # nodelist

                    a_sample = train_time_list[i]
                    # p_sample1 = train_node_list[p_index]
                    # p_sample = p_sample1[:len(p_sample)]
                    p_sample = [a_sample[p_start_index:p_start_index + sub_len] for p_start_index in p_start_index_list]
                    n_sample = [train_time_list[n_index][n_start_index:n_start_index+sub_len] for n_index, n_start_index in zip(n_index_list,n_start_index_list)]
                    apn_time_triplets.append([a_sample, p_sample, n_sample])  # timelist

                    a_sample = train_d2vec_list[i]
                    d2v = self.d2vec(p_sample)
                    p_sample = [d2v[x].tolist() for x in range(len(p_sample))]
                    d2v = self.d2vec(n_sample)
                    n_sample = [d2v[x].tolist() for x in range(len(n_sample))]

                    apn_d2vec_triplets.append([a_sample, p_sample, n_sample])  # d2veclist
                if len(apn_node_triplets) == len(train_node_list) * 2:  # based on the num of train triplets we need
                    break
            if len(apn_node_triplets) == len(train_node_list) * 2:
                break

        print("complete: sample")
        print(len(apn_time_triplets))
        print(apn_node_triplets[0])
        pickle.dump(apn_node_triplets, open(str(config["path_node_triplets"]), 'wb'))
        pickle.dump(apn_time_triplets, open(str(config["path_time_triplets"]), 'wb'))
        pickle.dump(apn_d2vec_triplets, open(str(config["path_d2vec_triplets"]), 'wb'))

    def return_triplets_num(self):
        apn_node_triplets = pickle.load(open(str(config["path_node_triplets"]), 'rb'))
        return len(apn_node_triplets)
    
def triplet_groud_truth():
    apn_node_triplets = pickle.load(open(str(config["path_node_triplets"]),'rb'))
    apn_time_triplets = pickle.load(open(str(config["path_time_triplets"]),'rb'))
    com_max_s = []
    com_max_t = []
    for i in range(len(apn_time_triplets)):
        # multiple p/n samples
        inter_s = []
        inter_t = []
        for j in range(3):
            if str(config["distance_type"]) == "TP":
                # change the number of positive/negative samples
                ap_s = spatial_com.TP_dis(apn_node_triplets[i][0],apn_node_triplets[i][1][j])
                an_s = spatial_com.TP_dis(apn_node_triplets[i][0],apn_node_triplets[i][2][j])
                inter_s.append([ap_s, an_s])
                ap_t = temporal_com.TP_dis(apn_time_triplets[i][0], apn_time_triplets[i][1][j])
                an_t = temporal_com.TP_dis(apn_time_triplets[i][0], apn_time_triplets[i][2][j])
                inter_t.append([ap_t, an_t])
            elif str(config["distance_type"]) == "DITA":
                ap_s = spatial_com.DITA_dis(apn_node_triplets[i][0], apn_node_triplets[i][1])
                an_s = spatial_com.DITA_dis(apn_node_triplets[i][0], apn_node_triplets[i][2])
                com_max_s.append([ap_s, an_s])
                ap_t = temporal_com.DITA_dis(apn_time_triplets[i][0], apn_time_triplets[i][1])
                an_t = temporal_com.DITA_dis(apn_time_triplets[i][0], apn_time_triplets[i][2])
                com_max_t.append([ap_t, an_t])
            elif str(config["distance_type"]) == "discret_frechet":
                ap_s = spatial_com.frechet_dis(apn_node_triplets[i][0], apn_node_triplets[i][1])
                an_s = spatial_com.frechet_dis(apn_node_triplets[i][0], apn_node_triplets[i][2])
                com_max_s.append([ap_s, an_s])
                ap_t = temporal_com.frechet_dis(apn_time_triplets[i][0], apn_time_triplets[i][1])
                an_t = temporal_com.frechet_dis(apn_time_triplets[i][0], apn_time_triplets[i][2])
                com_max_t.append([ap_t, an_t])
            elif str(config["distance_type"]) == "LCRS":
                ap_s = spatial_com.LCRS_dis(apn_node_triplets[i][0], apn_node_triplets[i][1])
                an_s = spatial_com.LCRS_dis(apn_node_triplets[i][0], apn_node_triplets[i][2])
                com_max_s.append([ap_s, an_s])
                ap_t = temporal_com.LCRS_dis(apn_time_triplets[i][0], apn_time_triplets[i][1])
                an_t = temporal_com.LCRS_dis(apn_time_triplets[i][0], apn_time_triplets[i][2])
                com_max_t.append([ap_t, an_t])
            elif str(config["distance_type"]) == "NetERP":
                ap_s = spatial_com.NetERP_dis(apn_node_triplets[i][0], apn_node_triplets[i][1])
                an_s = spatial_com.NetERP_dis(apn_node_triplets[i][0], apn_node_triplets[i][2])
                com_max_s.append([ap_s, an_s])
                ap_t = temporal_com.NetERP_dis(apn_time_triplets[i][0], apn_time_triplets[i][1])
                an_t = temporal_com.NetERP_dis(apn_time_triplets[i][0], apn_time_triplets[i][2])
                com_max_t.append([ap_t, an_t])
        com_max_s.append(inter_s)
        com_max_t.append(inter_t)

    com_max_s = np.array(com_max_s)
    com_max_t = np.array(com_max_t)

    com_max_s = com_max_s/np.max(com_max_s)*8
    com_max_t = com_max_t/np.max(com_max_t)*8

    train_triplets_dis = (com_max_s+com_max_t)/2

    np.save(str(config["path_triplets_truth"]), train_triplets_dis)
    print("complete: triplet groud truth")
    print(train_triplets_dis[0])

class batch_list():
    def __init__(self, batch_size):
        self.apn_node_triplets = np.array(pickle.load(open(str(config["path_node_triplets"]), 'rb')))
        self.apn_d2vec_triplets = np.array(pickle.load(open(str(config["path_d2vec_triplets"]), 'rb')))
        self.batch_size = batch_size
        self.start = len(self.apn_node_triplets)    # ordered is '0' ; reverse is 'maxsize'

    def getbatch_one(self):
        '''
        # batch random
        index = list(range(len(self.apn_node_triplets)))
        random.shuffle(index)
        batch_index = random.sample(index, self.batch_size)

        # batch ordered
        if self.start + self.batch_size > len(self.apn_node_triplets):
            self.start = 0
        batch_index = list(range(self.start, self.start + self.batch_size))
        self.start += self.batch_size
        '''

        # batch reverse
        if self.start - self.batch_size < 0:
            self.start = len(self.apn_node_triplets)
        batch_index = list(range(self.start - self.batch_size, self.start))
        self.start -= self.batch_size

        node_list = self.apn_node_triplets[batch_index]
        time_list = self.apn_d2vec_triplets[batch_index]

        a_node_batch = []
        a_time_batch = []
        p_node_batch = []
        p_time_batch = []
        n_node_batch = []
        n_time_batch = []
        for tri1 in node_list:
            a_node_batch.append(tri1[0])
            p_node_batch.append(tri1[1])
            n_node_batch.append(tri1[2])
        for tri2 in time_list:
            a_time_batch.append(tri2[0])
            p_time_batch.append(tri2[1])
            n_time_batch.append(tri2[2])

        return a_node_batch, a_time_batch, p_node_batch, p_time_batch, n_node_batch, n_time_batch, batch_index

if __name__ == "__main__":
    data = DataLoader()
    data.get_triplets() # get triplets
    triplet_groud_truth() # compute the ground truth of triplets