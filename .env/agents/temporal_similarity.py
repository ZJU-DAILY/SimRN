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
import datetime


import os

config = yaml.load(open('config.yaml'))
dataset = str(config["dataset"])
dataset_point = config["pointnum"][str(config["dataset"])]

def find_trajtimelist():
    longest_traj = 0
    smallest_time = np.inf
    time_list_int = np.load(str(config["shuffle_time_file"]), allow_pickle=True)
    for time_list in time_list_int:
        if len(time_list)>longest_traj:
            longest_traj = len(time_list)
        for t in time_list:
            if t<smallest_time:
                smallest_time = t
    return longest_traj, smallest_time

longest_trajtime_len, smallest_trajtime = find_trajtimelist()

def batch_timelist_ground_truth(valiortest = None):
    time_list_int = np.load(str(config["shuffle_time_file"]), allow_pickle=True)
    if valiortest == 'vali':
        time_list_int = time_list_int[2000:2800]   # based dataset and "validation or test" (train:validation:test = 1w:4k:1.6w)
    elif valiortest == 'test':
        time_list_int = time_list_int[2800:6000]

    sample_list = time_list_int[:1000]  # m*n matrix distance, m and n can be set by yourself

    pool = Pool(processes=1)
    # pool = Pool(processes=19)
    for i in range(len(sample_list)+1):
        if i!=0 and i%50==0:
            pool.apply_async(timelist_distance, (i, sample_list[i-50:i], time_list_int, valiortest))
    pool.close()
    pool.join()

    return len(sample_list)

def merge_timelist_ground_truth(sample_len, valiortest):
    res = []
    for i in range(sample_len+1):
        if i!=0 and i%50==0:
            res.append(np.load('./ground_truth/{}/{}/{}_batch/{}_temporal_distance_{}.npy'.format(dataset, str(config["distance_type"]), valiortest, str(config["distance_type"]), str(i))))
    res = np.concatenate(res, axis=0)
    np.save('./ground_truth/{}/{}/{}_temporal_distance.npy'.format(dataset, str(config["distance_type"]), valiortest), res)


def get_period_idx(timestamp):
    period = config["period"]
    interval = int(24/period)
    dt = datetime.datetime.fromtimestamp(timestamp)
    hour = dt.hour
    return int(hour/interval)


def batch_extra_time_feature():
    node_list_int = np.load(str(config["shuffle_node_file"]), allow_pickle=True)
    time_list_int = np.load(str(config["shuffle_time_file"]), allow_pickle=True)

    for i in range(dataset_point+1):
        if i!=0 and i%500==0:
            cal_time_feature(i, list(range(i - 500, i)), node_list_int, time_list_int)

def merge_extra_time_feature():
    res = []
    for i in range(dataset_point + 1):
        if i != 0 and i % 500 == 0:
            print(i)
            res.append(np.load('/media/yilin/LENOVO/lab/temporal/extra_time_feature_{}.npy'.format(str(i))))
    res = np.concatenate(res, axis=0)
    np.save('/media/yilin/LENOVO/lab/extra_time_feature.npy'.format(dataset), res)

def cal_time_feature(i, id_list = [], node_list_int = [[]], time_list_int = [[]]):

    s1 = time.time()

    period = config["period"]
    n = len(id_list)
    print(f'i is {i}, n is {n}')
    offset = 500*int(i/500-1)
    print(offset)
    # matrix = [[[[] for _ in range(period)] for _ in range(dataset_point)] for _ in range(n)]
    matrix = []

    for id in id_list:
        print(id)
        curr = [[[] for _ in range(period)] for _ in range(dataset_point)]
        # list related to start node
        # curr = matrix[id-offset]
        for idx, node_list in enumerate(node_list_int):
            if id in node_list and node_list.index(id) != len(node_list)-1:
                # time_list related to the curr trajectory
                time_list = time_list_int[idx]
                # get start time and its period index
                timestamp = time_list[node_list.index(id)]
                period_idx = get_period_idx(timestamp)
                # get end and interval time
                end = node_list[node_list.index(id)+1]
                interval = time_list[node_list.index(id)+1]-timestamp
                curr[end][period_idx].append(interval)


        for end, periods in enumerate(curr):
            sum_p = 0.0
            count_p = 0
            for p in periods:
                if p:
                    sum_p += sum(p)
                    count_p += len(p)
            if count_p != 0:
                curr[end] = sum_p / count_p
            else:
                curr[end] = sum_p

        matrix.append(curr)

    matrix = np.array(matrix,dtype=np.float32)
    np.save('./ground_truth/{}/extra_time_feature_{}.npy'.format(dataset, str(i)), matrix)
    s2 = time.time()
    print("time: ",s2-s1)
    print("finish")


def demo():
    matrix = np.load('/media/yilin/LENOVO/lab/temporal/extra_time_feature_500.npy')
    period = config["period"]
    m = len(matrix)
    n = len(matrix[0])
    print(m)
    print(n)

    for i in range(m):
        count = 0
        for j in range(n):
            f = False
            for x in range(period):
                if matrix[i][j][x] != 0.0:
                    f = True
                    break
            if f:
                count += 1
                print(matrix[i][j])
        print(f'{i}\'s count is {count}')

def generate_time_feature():
    edge = pd.read_csv('./data/{}/road/edge_weight_direction_interval1.csv'.format(dataset),sep=',')

    node_s, node_e, intervals = edge.s_node, edge.e_node, edge.interval
    step = 500
    for i in range(1):
        node_pairs = list(zip(node_s, node_e))

        for x,pair in enumerate(node_pairs):
            s = pair[0]
            e = pair[1]
            offset = int(s/500)
            idx = (offset+1)*500
            print(f's is {s}, offset is {offset}, idx is {idx}')
            matrix = np.load('./ground_truth/tdrive/extra_time_feature_{}.npy'.format(str(idx)),mmap_mode='r')
            interval = matrix[(s%500)][e]
            intervals[x] = interval
            print(interval)

        edge['interval'] = intervals
        edge.to_csv('./data/{}/road/edge_weight_direction_interval1.csv'.format(dataset), index=False)


def timelist_distance(k, sample_list = [[]], test_list = [[]], valiortest=None):

    all_dis_list = []
    i = 0
    for sample in sample_list:
        i += 1
        print(f'i-{i}')
        j = 0
        one_dis_list = []
        for traj in test_list:
            if str(config["distance_type"]) == 'TP':
                one_dis_list.append(TP_dis(sample, traj))
            elif str(config["distance_type"]) == 'DTW':
                one_dis_list.append(DITA_dis(sample, traj))
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
    filename = './ground_truth/{}/{}/{}_batch/{}_temporal_distance_{}.npy'.format(dataset, str(config["distance_type"]), valiortest, str(config["distance_type"]), str(k))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, all_dis_list)

    print('complete: ' + str(k))

@numba.jit(nopython=True, fastmath=True)
def TP_dis(list_a = [] , list_b = []):
    tr1 = np.array(list_a)
    tr2 = np.array(list_b)
    M, N = len(tr1), len(tr2)
    max1 = -1
    for i in range(M):
        mindis = np.inf
        for j in range(N):
            temp = abs(tr1[i]-tr2[j])
            if temp < mindis:
                mindis = temp
        if mindis != np.inf and mindis > max1:
            max1 = mindis

    max2 = -1
    for i in range(N):
        mindis = np.inf
        for j in range(M):
            temp = abs(tr2[i]-tr1[j])
            if temp < mindis:
                mindis = temp
        if mindis != np.inf and mindis > max2:
            max2 = mindis

    return int(max(max1,max2))

@numba.jit(nopython=True, fastmath=True)
def DITA_dis(list_a = [], list_b = []):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    cost = np.zeros((M, N))
    cost[0, 0] = abs(tr1[0]-tr2[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + abs(tr1[i]-tr2[0])
    for i in range(1, N):
        cost[0, i] = cost[0, i - 1] + abs(tr1[0]-tr2[i])
    for i in range(1, M):
        for j in range(1, N):
            small = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(small) + abs(tr1[i]-tr2[j])
    return int(cost[M - 1, N - 1])

@numba.jit(nopython=True, fastmath=True)
def LCRS_dis(list_a = [], list_b = []):
    lena = len(list_a)
    lenb = len(list_b)
    c = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    for i in range(lena):
        for j in range(lenb):
            if abs(list_a[i] - list_b[j]) <= 3600:
                c[i + 1][j + 1] = c[i][j] + 1
            elif c[i + 1][j] > c[i][j + 1]:
                c[i + 1][j + 1] = c[i + 1][j]
            else:
                c[i + 1][j + 1] = c[i][j + 1]
    if c[-1][-1] == 0:
        return longest_trajtime_len*2
    else:
        return (lena + lenb - c[-1][-1])/float(c[-1][-1])

@numba.jit(nopython=True, fastmath=True)
def NetERP_dis(list_a = [], list_b = []):
    lena = len(list_a)
    lenb = len(list_b)

    edit = np.zeros((lena + 1, lenb + 1))
    for i in range(lena + 1):
        edit[i][0] = i * smallest_trajtime
    for i in range(lenb + 1):
        edit[0][i] = i * smallest_trajtime

    for i in range(1, lena + 1):
        for j in range(1, lenb + 1):
            edit[i][j] = min(edit[i - 1][j] + list_a[i-1] - smallest_trajtime, edit[i][j - 1] + list_b[j-1] - smallest_trajtime, edit[i - 1][j - 1] + abs(list_a[i-1] - list_b[j-1]))

    return edit[-1][-1]


# def grid_mapping(point):
#     # 这里的grid_mapping逻辑需要根据实际的网格划分或区域定义
#     return int(point[0] // 1), int(point[1] // 1)  # 简单网格化

@numba.jit(nopython=True, fastmath=True)
def LORS_dis(list_a=[], list_b=[], delta=1.0):
    M, N = len(list_a), len(list_b)
    cost = np.zeros((M + 1, N + 1))

    for i in range(1, M + 1):
        cost[i, 0] = i
    for j in range(1, N + 1):
        cost[0, j] = j

    for i in range(1, M + 1):
        for j in range(1, N + 1):
            if (list_a[i - 1] - list_b[j - 1]) <= delta:
                cost[i, j] = cost[i - 1, j - 1]
            else:
                cost[i, j] = 1 + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return cost[M, N]

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


if __name__ == '__main__':
    sample_len = batch_timelist_ground_truth(valiortest='vali')
    merge_timelist_ground_truth(sample_len=sample_len, valiortest='vali')
    sample_len = batch_timelist_ground_truth(valiortest='test')
    merge_timelist_ground_truth(sample_len=sample_len, valiortest='test')

    # batch_extra_time_feature()
    # merge_extra_time_feature()
    # generate_time_feature()

