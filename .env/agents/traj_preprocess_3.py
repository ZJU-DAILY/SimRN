# by dlhu, 05/2024
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from agents.date2vec import Date2VecConvert
from torch_geometric.nn import Node2Vec
import pickle
import time
import datetime
import random
import os

random.seed(1953)
#config = yaml.load(open('config.yaml'))

# Extract the spatial and temporal information
def prepare_dataset(trajfile, timefile, kseg=5):
    """
    :param trajfile: map-matching result
    :param timefile: raw coor-timestamp file
    :param kseg: Simplify the trajectory to kseg
    """
    node_list = pd.read_csv(trajfile)
    node_list = node_list.Node_list
    time_list = pd.read_csv(timefile)
    time_list = time_list.Time_list

    node_list_int = []
    for nlist in node_list:
        tmp_list = []
        nlists = nlist[1:-1].replace('[', '').replace(']', '').replace(' ', ',').replace('\n', ',').split(',')
        for n in nlists:
            if n != '':
                tmp_list.append(int(n))
        node_list_int.append(tmp_list)
    node_list_int = np.array(node_list_int)

    time_list_int = []
    for tlist in time_list:
        tmp_list = []
        tlist = tlist[1:-1].replace('[', '').replace(']', '').replace(' ', ',').replace('\n', ',').split(',')
        for t in tlist:
            if t != '':
                tmp_list.append(int(t))
        time_list_int.append(tmp_list)
    time_list_int = np.array(time_list_int)

    df = pd.read_csv(trajfile)
    trajs = df.Coor_list

    coor_trajs = []
    for traj in trajs:
        traj = traj[1:-1].replace('[', '').replace(']', '').replace(' ', ',').replace('\n', ',').split(',')
        ts = []
        for s in traj:
            if s != '':
                ts.append(float(s))
        traj = np.reshape(ts, [-1, 2], order='C')
        coor_trajs.append(traj)

    kseg_coor_trajs = []
    for t in coor_trajs:
        kseg_coor = []
        seg = len(t) // kseg
        t = np.array(t)
        for i in range(kseg):
            if i == kseg - 1:
                kseg_coor.append(np.mean(t[i * seg:], axis=0))
            else:
                kseg_coor.append(np.mean(t[i * seg:i * seg + seg], axis=0))
        kseg_coor_trajs.append(kseg_coor)
    kseg_coor_trajs = np.array(kseg_coor_trajs)

    shuffle_index = list(range(len(node_list_int)))
    random.shuffle(shuffle_index)
    shuffle_index = shuffle_index[:10000] #修改

    coor_trajs = np.array(coor_trajs)
    coor_trajs = coor_trajs[shuffle_index]

    kseg_coor_trajs = kseg_coor_trajs[shuffle_index]
    time_list_int = time_list_int[shuffle_index]
    node_list_int = node_list_int[shuffle_index]

    file1 = str(config["shuffle_coor_file"])
    os.makedirs(os.path.dirname(file1), exist_ok=True)
    pickle.dump(coor_trajs, open(file1, 'wb'))

    file2 = str(config["shuffle_node_file"])
    os.makedirs(os.path.dirname(file2), exist_ok=True)
    pickle.dump(node_list_int, open(file2, 'wb'))

    file3 = str(config["shuffle_time_file"])
    os.makedirs(os.path.dirname(file3), exist_ok=True)
    pickle.dump(time_list_int, open(file3, 'wb'))

    file4 = str(config["shuffle_kseg_file"])
    os.makedirs(os.path.dirname(file4), exist_ok=True)
    pickle.dump(kseg_coor_trajs, open(file4, 'wb'))

# Time embedding: Prepare for temporal embedding
class Date2vec(tf.keras.Model):
    def __init__(self):
        super(Date2vec, self).__init__()
        self.d2v = Date2VecConvert(model_path="./d2v_model/d2v_98291_17.169918439404636.pth")

    def call(self, time_seq):
        all_list = []
        for one_seq in time_seq:
            one_list = []
            for timestamp in one_seq:
                t = datetime.datetime.fromtimestamp(timestamp)
                t = [t.hour, t.minute, t.second, t.year, t.month, t.day]
                x = tf.constant(t, dtype=tf.float32)
                embed = self.d2v(x)
                one_list.append(embed)

            one_list = tf.concat(one_list, axis=0)
            one_list = tf.reshape(one_list, [-1, 64])

            all_list.append(one_list.numpy().tolist())

        all_list = np.array(all_list)

        return all_list
    
# Obtain the reconstructed road network
def read_graph(dataset):
    """
    Read network edages from text file and return networks object
    :param file: input dataset name
    :return: edage index with shape (n,2)
    """
    dataPath = "./data/" + dataset
    edge = dataPath + "/road/edge_weight_p.csv"
    node = dataPath + "/road/node_p.csv"

    df_edge = pd.read_csv(edge, sep=',')
    df_node = pd.read_csv(node, sep=',')

    edge_index = df_edge[["s_node", "e_node"]].to_numpy()
    num_node = df_node["node"].size

    print("{0} road netowrk has {1} edges.".format(config["dataset"], edge_index.shape[0]))
    print("{0} road netowrk has {1} nodes.".format(config["dataset"], num_node))

    return edge_index, num_node

def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def train_epoch(model, loader, optimizer):
    # Training with epoch iteration
    last_loss = 1
    print("Training node embedding with node2vec...")
    for i in range(100):
        loss = train(model, loader, optimizer)
        print('Epoch: {0} \tLoss: {1:.4f}'.format(i, loss))
        if abs(last_loss - loss) < 1e-5:
            break
        else:
            last_loss = loss

# Save the node embeddings
def save_embeddings(model, num_nodes, dataset, device):
    model.eval()
    node_features = model(tf.range(num_nodes, device=device)).cpu().numpy() # torch.arange
    np.save("./data/" + dataset + "/node_features.npy", node_features)
    print("Node embedding saved at: ./data/" + dataset + "/node_features.npy")
    return

if __name__ == "__main__":
    config = yaml.load(open('config.yaml'))

    edge_index, num_node = read_graph(str(config["dataset"]))

    device = "cuda:" + str(config["cuda"])
    feature_size = config["feature_size"]
    walk_length = config["node2vec"]["walk_length"]
    context_size = config["node2vec"]["context_size"]
    walks_per_node = config["node2vec"]["walks_per_node"]
    p = config["node2vec"]["p"]
    q = config["node2vec"]["q"]

    edge_index = tf.constant(edge_index, dtype=tf.int64).t().contiguous().to(device) # torch.LongTensor

    # Node embedding: Prepare for spatial embedding
    model = Node2Vec(
        edge_index,
        embedding_dim=feature_size,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=1,
        p=p,
        q=q,
        sparse=True,
        num_nodes=num_node
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = tf.kears.optimizers.Adam(model.parameters(), lr=0.01) # torch.optim.SparseAdam(model.parameters(), lr=0.01)

    # Train until delta loss has been reached
    train_epoch(model, loader, optimizer)

    save_embeddings(model, num_node, str(config["dataset"]), device)

    prepare_dataset(trajfile=str(config["traj_file"]), timefile=str(config["time_file"]), kseg=config["kseg"])
    
    # Time embedding
    d2vec = Date2vec()
    timelist = np.load(str(config["shuffle_time_file"]), allow_pickle=True)
    d2v = d2vec(timelist)
    print(len(d2v))
    np.save(str(config["shuffle_d2vec_file"]), d2v)