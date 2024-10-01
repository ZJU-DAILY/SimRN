from agents.STmatching_distribution_ver import network_data
from multiprocessing import Pool
import pickle
import numpy as np
import math
import networkx as nx
import time
import numba
import sys
import random
import pandas as pd
import yaml
import collections

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

random.seed(1998)
config = yaml.load(open('config.yaml'))
dataset = str(config["dataset"])
dataset_point = config["pointnum"][str(config["dataset"])]

def find_longest_trajectory():
    longest_traj = 0
    node_list_int = np.load(str(config["shuffle_node_file"]), allow_pickle=True)
    for node_list in node_list_int:
        if len(node_list)>longest_traj:
            longest_traj = len(node_list)
    return longest_traj

longest_traj_len = find_longest_trajectory()

def batch_Point_distance():

    pool = Pool(processes=1)
    # pool = Pool(processes=20)
    for i in range(dataset_point + 1):
        if i != 0 and i % 1000 == 0:
            pool.apply_async(parallel_point_com, (i, list(range(i - 1000, i))))
    pool.close()
    pool.join()

def merge_Point_distance():
    res = []
    for i in range(dataset_point + 1):
        if i != 0 and i % 1000 == 0:
            res.append(np.load('./ground_truth/{}/Point_dis_matrix_{}.npy'.format(dataset, str(i))))
    res = np.concatenate(res, axis=0)
    np.save('./ground_truth/{}/Point_dis_matrix.npy'.format(dataset), res)

def parallel_point_com(i, id_list = []):
    batch_list = []

    for k in id_list:
        one_list = []
        if k in roadnetwork.nodes():
            length_list = nx.shortest_path_length(roadnetwork, source=k, weight='distance')
            for j in range(dataset_point):
                if (j in length_list.keys()) == True:
                    one_list.append(length_list[j])
                else:
                    one_list.append(-1)
            batch_list.append(np.array(one_list,dtype=np.float32))
        else:
            one_list = [-1 for j in range(dataset_point)]
            batch_list.append(np.array(one_list,dtype=np.float32))

    batch_list = np.array(batch_list,dtype=np.float32)
    np.save('./ground_truth/{}/Point_dis_matrix_{}.npy'.format(dataset, str(i)), batch_list)

def generate_point_matrix():
    # res = np.load('./ground_truth/{}/Point_dis_matrix.npy'.format(dataset),mmap_mode='r')
    res = np.load('/media/yilin/LENOVO/lab/Point_dis_matrix.npy',mmap_mode='r')
    return res

def generate_node_edge_interation():
    node_edge_dict = collections.defaultdict(set)
    edge = pd.read_csv('./data/{}/road/edge_weight.csv'.format(dataset))
    node_s, node_e = edge.s_node, edge.e_node

    for idx, (n_s, n_e) in enumerate(zip(node_s, node_e)):
        node_edge_dict[int(n_s)].add(idx)
        node_edge_dict[int(n_e)].add(idx)

    return node_edge_dict

def batch_similarity_ground_truth(valiortest = None):
    node_list_int = np.load(str(config["shuffle_node_file"]), allow_pickle=True)
    if valiortest == 'vali':
        node_list_int = node_list_int[2000:2800]  # based dataset and "validation or test"  (train:validation:test = 1w:4k:1.6w)
    elif valiortest == 'test':
        node_list_int = node_list_int[2800:6000]

    sample_list = node_list_int[:1000]  # m*n matrix distance, m and n can be set by yourself

    pool = Pool(processes=1)
    # pool = Pool(processes=19)
    for i in range(len(sample_list)+1):
        if i!=0 and i%50==0:
            # pool.apply_async(Traj_distance, (i,sample_list[i-50:i],node_list_int, valiortest))
            pool.apply(Traj_distance, (i,sample_list[i-50:i],node_list_int, valiortest))

    pool.close()
    pool.join()
    return len(sample_list)

def merge_similarity_ground_truth(sample_len, valiortest):
    res = []
    for i in range(sample_len+1):
        if i!=0 and i%50==0:
            res.append(np.load('./ground_truth/{}/{}/{}_batch/{}_spatial_distance_{}.npy'.format(dataset, str(config["distance_type"]),valiortest, str(config["distance_type"]), str(i))))
    res = np.concatenate(res, axis=0)
    np.save('./ground_truth/{}/{}/{}_spatial_distance.npy'.format(dataset, str(config["distance_type"]), valiortest), res)

def Traj_distance(k, sample_list = [[]], test_list = [[]], valiortest = None):
    all_dis_list = []
    i = 0
    print(len(sample_list))
    print(len(test_list))
    for sample in sample_list:
        i += 1
        print(f'i-{i}')
        # j = 19
        one_dis_list = []
        # for idx in range(19,len(test_list)):
        for traj in test_list:
            # print(traj)
            if str(config["distance_type"]) == 'TP':
                one_dis_list.append(TP_dis(sample, traj))
            elif str(config["distance_type"]) == 'DTW':
                one_dis_list.append(DTW_dis(sample, traj))
            elif str(config["distance_type"]) == 'LCRS':
                one_dis_list.append(LCRS_dis(sample, traj))
            elif str(config["distance_type"]) == 'NetERP':
                one_dis_list.append(NetERP_dis(sample, traj))
            elif str(config["distance_type"]) == 'NetEDR':
                one_dis_list.append(NetEDR_dis(sample, traj))
            elif str(config["distance_type"]) == 'LORS':
                one_dis_list.append(LORS_dis(sample, traj))

        all_dis_list.append(np.array(one_dis_list))

    all_dis_list = np.array(all_dis_list)
    np.save('./ground_truth/{}/{}/{}_batch/{}_spatial_distance_{}.npy'.format(dataset, str(config["distance_type"]), valiortest, str(config["distance_type"]), str(k)), all_dis_list)

    print('complete: ' + str(k))

distance_matrix = generate_point_matrix()  # This code is required when computing triplets truth and commented out the rest of the time


# TP
@numba.njit()
def TP_dis(list_a = [] , list_b = []):
    tr1 = np.array(list_a)
    tr2 = np.array(list_b)
    M, N = len(tr1), len(tr2)
    max1 = -1
    for i in range(M):
        mindis = np.inf
        for j in range(N):
            if distance_matrix[tr1[i]][tr2[j]] != -1:
                temp = distance_matrix[tr1[i]][tr2[j]]
                if temp < mindis:
                    mindis = temp
            else:
                return -1
        if mindis != np.inf and mindis > max1:
            max1 = mindis

    max2 = -1
    for i in range(N):
        mindis = np.inf
        for j in range(M):
            if distance_matrix[tr2[i]][tr1[j]] != -1:
                temp = distance_matrix[tr2[i]][tr1[j]]
                if temp < mindis:
                    mindis = temp
            else:
                return -1
        if mindis != np.inf and mindis > max2:
            max2 = mindis

    return int(max(max1,max2))

# DTW
def DTW_dis(list_a, list_b):
    len1, len2 = len(list_a), len(list_b)

    dtw_matrix = np.full((len1 + 1, len2 + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = distance_matrix[list_a[i - 1], list_b[j - 1]]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],
                                          dtw_matrix[i, j - 1],
                                          dtw_matrix[i - 1, j - 1])

    return dtw_matrix[len1, len2]

# LCRS
node_edge_dict = generate_node_edge_interation()
def LCRS_dis(list_a = [], list_b = []):
    lena = len(list_a)
    lenb = len(list_b)
    c = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    for i in range(lena):
        for j in range(lenb):
            if len(node_edge_dict[list_a[i]] & node_edge_dict[list_b[j]]) >= 1:
                c[i + 1][j + 1] = c[i][j] + 1
            elif c[i + 1][j] > c[i][j + 1]:
                c[i + 1][j + 1] = c[i + 1][j]
            else:
                c[i + 1][j + 1] = c[i][j + 1]
    if c[-1][-1]==0:
        return longest_traj_len*2
    else:
        return (lena + lenb - c[-1][-1])/float(c[-1][-1])

# NetERP
def hot_node():
    max_num = 0
    max_idx = 0
    for idx, nodes_interaction in enumerate(distance_matrix):
        nodes_interaction = np.array(nodes_interaction)
        x = len(nodes_interaction[nodes_interaction != -1])
        if x > max_num:
            max_num = x
            max_idx = idx
    print(max_num, max_idx)
    return max_idx

hot_node_id = hot_node()

@numba.jit(nopython=True, fastmath=True)
def NetERP_dis(list_a = [], list_b = []):
    lena = len(list_a)
    lenb = len(list_b)

    edit = np.zeros((lena + 1, lenb + 1))
    for i in range(1, lena + 1):
        tp = distance_matrix[hot_node_id][list_a[i-1]]
        if tp == -1:
            return -1
        edit[i][0] = edit[i-1][0] + tp
    for i in range(1, lenb + 1):
        tp = distance_matrix[hot_node_id][list_b[i-1]]
        if tp == -1:
            return -1
        edit[0][i] = edit[0][i-1] + tp

    for i in range(1, lena + 1):
        for j in range(1, lenb + 1):
            tp1 = distance_matrix[hot_node_id][list_a[i-1]]
            tp2 = distance_matrix[hot_node_id][list_b[j-1]]
            tp3 = distance_matrix[list_a[i-1]][list_b[j-1]]
            if tp1 == -1 or tp2 == -1 or tp3 == -1:
                return -1
            edit[i][j] = min(edit[i - 1][j] + tp1, edit[i][j - 1] + tp2, edit[i - 1][j - 1] + tp3)

    return edit[-1][-1]

# LORS
def grid_mapping(point):
    point = vertice_dict[point]
    # 这里的grid_mapping逻辑需要根据实际的网格划分或区域定义
    return int(point[0] // 0.1), int(point[1] // 0.1)  # 简单网格化

def LORS_dis(list_a=[], list_b=[]):
    # 将轨迹点映射到网格区域ID
    region_a = [grid_mapping(p) for p in list_a]
    region_b = [grid_mapping(p) for p in list_b]

    M, N = len(region_a), len(region_b)
    cost = np.zeros((M + 1, N + 1))

    for i in range(1, M + 1):
        cost[i, 0] = i
    for j in range(1, N + 1):
        cost[0, j] = j

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if region_a[i - 1] == region_b[j - 1]:
                cost[i, j] = cost[i - 1, j - 1]
            else:
                cost[i, j] = 1 + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return cost[M, N]

# NetEDR
def point_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# @numba.jit(nopython=True, fastmath=True)
def NetEDR_dis(list_a=[], list_b=[], epsilon=1.0):
    M, N = len(list_a), len(list_b)
    cost = np.zeros((M + 1, N + 1))

    for i in range(1, M + 1):
        cost[i, 0] = i
    for j in range(1, N + 1):
        cost[0, j] = j

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if point_distance(list_a[i - 1], list_b[j - 1]) <= epsilon:
                cost[i, j] = cost[i - 1, j - 1]
            else:
                cost[i, j] = 1 + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return cost[M, N]

def generate_edge_direction():
    edge = pd.read_csv('./data/{}/road/edge_weight.csv'.format(dataset))
    node_s, node_e = edge.s_node, edge.e_node
    node_pairs = list(zip(node_s, node_e))
    directions = []
    for s,e in node_pairs:
        direction = 0
        if (e,s) in node_pairs:
            direction += 1
        directions.append(direction)
        print(direction)
    edge['direction'] = directions

    edge.to_csv('./data/{}/road/edge_weight_direction.csv'.format(dataset), index=False)


if __name__ == '__main__':
    nx_vertice, nx_edge, vertice_dict, edge_dict, edge_dist, edge_dist_dict, roadnetwork = network_data()
            # batch_Point_distance()
            # merge_Point_distance()
    distance_matrix = generate_point_matrix()

    node_edge_dict = generate_node_edge_interation()
    sample_len = batch_similarity_ground_truth(valiortest='vali')
    merge_similarity_ground_truth(sample_len=sample_len, valiortest='vali')
    sample_len = batch_similarity_ground_truth(valiortest='test')
    merge_similarity_ground_truth(sample_len=sample_len, valiortest='test')

    # generate_edge_direction()




