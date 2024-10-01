# by dlhu, 07/2024
# obtain the reconstructed road network

import numpy as np
import random
import pandas as pd
import yaml
import time
import datetime

random.seed(1998)
config = yaml.load(open('config.yaml'))
dataset = str(config["dataset"])
dataset_point = config["pointnum"][str(config["dataset"])]

def generate_direction():
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
    print("direction finish")

def batch_time_interval():
    node_list_int = np.load(str(config["shuffle_node_file"]), allow_pickle=True)
    time_list_int = np.load(str(config["shuffle_time_file"]), allow_pickle=True)

    # pool = Pool(processes=1)
    # for i in range(dataset_point + 1):
    #     if i != 0 and i % 500 == 0:
    #         pool.apply_async(cal_time_feature, (i, list(range(i - 500, i)), node_list_int, time_list_int))
    # pool.close()
    # pool.join()

    for i in range(dataset_point+1):
        if i!=0 and i%500==0:
            cal_time_interval(i, list(range(i - 500, i)), node_list_int, time_list_int)


# def merge_time_interval():
#     res = []
#     for i in range(dataset_point + 1):
#         if i != 0 and i % 500 == 0:
#             print(i)
#             res.append(np.load('./data/{}/time/interval_{}.npy'.format(dataset,str(i))))
#     res = np.concatenate(res, axis=0)
#     np.save('./data/{}/interval.npy'.format(dataset), res)

def cal_time_interval(i, id_list = [], node_list_int = [[]], time_list_int = [[]]):

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
    np.save('./data/{}/time/interval_{}.npy'.format(dataset, str(i)), matrix)
    s2 = time.time()
    print("time: ",s2-s1)
    print("finish")

def generate_interval():
    batch_time_interval()
    # merge_time_interval()
    edge = pd.read_csv('./data/{}/road/edge_weight_direction.csv'.format(dataset),sep=',')

    # intervals = []
    # title = ["s_node", "e_node", "direction","interval"]
    # # intervals_title = [f'interval_{i}' for i in range(period)]
    # for i in df_edge.interval:
    #     interval = []
    #     interval_str = i[1:-1]
    #     res = interval_str.split()
    #     # for idx, p in enumerate(res):
    #     #     intervals[idx].append(float(p))
    #     for j in res:
    #         interval.append(float(j))
    #     intervals.append(interval)
    # # for i in range(period):
    # #     df_edge[f'interval_{i}'] = intervals[i]
    # # edge_index = df_edge[title + intervals_title].to_numpy()
    # df_edge['interval'] = intervals
    # edge_index = df_edge[title].to_numpy()
    # # period = config["period"]
    # # intervals = [[] for _ in range(period)]
    # # title = ["length", "direction"]
    # # intervals_title = [f'interval_{i}' for i in range(period)]
    # # for i in edge.interval:
    # #     interval = i[1:-1]
    # #     res = interval.split()
    # #     for idx, p in enumerate(res):
    # #         intervals[idx].append(float(p))
    # # for i in range(period):
    # #     edge[f'interval_{i}'] = intervals[i]
    # #
    # # edge_attr = edge[title+intervals_title].to_numpy()
    # # edge_attr = torch.tensor(edge_attr).t().contiguous()
    # edge_index = torch.tensor(edge_index).t().contiguous()
    # print(edge_index.shape)



    #
    intervals = []
    node_s, node_e = edge.s_node, edge.e_node

    node_pairs = list(zip(node_s, node_e))
    for pair in node_pairs:
        s = pair[0]
        e = pair[1]
        offset = int(s / 500)
        idx = (offset + 1) * 500
        print(f's is {s}, offset is {offset}, idx is {idx}')
        matrix = np.load('./data/{}/time/interval_{}.npy'.format(dataset, str(idx)), mmap_mode='r')
        interval = matrix[(s % 500)][e]
        intervals.append(interval)
        print(interval)

    edge['interval'] = intervals
    edge.to_csv('./data/{}/road/edge_weight_direction_interval.csv'.format(dataset), index=False)
    print("interval finish")

def get_period_idx(timestamp):
    period = config["period"]
    interval = int(24/period)
    dt = datetime.datetime.fromtimestamp(timestamp)
    hour = dt.hour
    return int(hour/interval)

if __name__ == '__main__':
    # generate_direction()
    generate_interval()
