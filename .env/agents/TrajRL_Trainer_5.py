import numpy as np
import torch
import yaml
import os
import pickle
import time
import json
import torch.distributed as dis

from sympy.physics.units import action

import get_triplets_4 as data_utils
import test_method as test_method
from model_network import STTrajSimEncoder
from lossfun import LossFun


class STsim_Trainer: 
    def __init__(self, actions=None):
        config = yaml.load(open('config.yaml'))

        self.feature_size = config["feature_size"]
        self.embedding_size = config["embedding_size"]
        self.date2vec_size = config["date2vec_size"]
        self.hidden_size = actions["hidden_size"] if actions is not None else config["hidden_size"]
        self.num_layers = actions["layer_num"] if actions is not None else config["num_layers"]
        self.dropout_rate = config["dropout_rate"]
        self.concat = config["concat"]
        self.device = "cuda:" + str(config["cuda"])
        self.learning_rate = actions["learning_rate"] if actions is not None else config["learning_rate"]
        self.epochs = actions["training_epoch"] if actions is not None else config["epochs"]

        self.train_batch = actions["batch_size"] if actions is not None else config["train_batch"]
        self.test_batch = config["test_batch"]
        self.traj_file = str(config["traj_file"])
        self.time_file = str(config["time_file"])

        self.dataset = str(config["dataset"])
        self.distance_type = str(config["distance_type"])
        self.early_stop = config["early_stop"]

        self.best_model = None
        self.communication_stats_list = [] 

    # def set_best_model(self, best_model):
    #     self.best_model = best_model

    def ST_eval(self, load_model=None):
        net = STTrajSimEncoder(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size=self.date2vec_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               concat=self.concat,
                               device=self.device)

        if load_model:
            net.load_state_dict(torch.load(load_model))
            net.to(self.device)

        dataload = data_utils.DataLoader()
        road_network = data_utils.load_netowrk(self.dataset).to(self.device)

        with torch.no_grad():
            vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='test')
            embedding_vali = test_method.compute_embedding(road_network=road_network, net=net,
                                                           test_traj=list(vali_node_list),
                                                           test_time=list(vali_d2vec_list),
                                                           test_batch=self.test_batch)
            acc = test_method.test_model(embedding_vali, isvali=False)
            print(acc)

    def ST_train(self, load_model=None, load_optimizer=None):
        net = STTrajSimEncoder(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size=self.date2vec_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               concat=self.concat,
                               device=self.device)

        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=self.learning_rate,
                                     weight_decay=0.0001)
        lossfunction = LossFun(self.train_batch, self.distance_type)

        net.to(self.device)
        lossfunction.to(self.device)

        road_network = data_utils.load_netowrk(self.dataset).to(self.device)

        dataload = data_utils.DataLoader()
        batch_l = data_utils.batch_list(batch_size=self.train_batch)
        bt_num = int(dataload.return_triplets_num() / self.train_batch)

        best_epoch = 0
        best_hr10 = 0
        lastepoch = '0'
        if load_model != None:
            net.load_state_dict(torch.load(load_model))
            optimizer.load_state_dict(torch.load(load_optimizer))
            lastepoch = load_model.split('/')[-1].split('_')[3]
            best_epoch = int(lastepoch)

        i = 0
        best_model_path = None
        for epoch in range(int(lastepoch), self.epochs):
            print(f'i-{i}')
            i += 1
            net.train()
            s1 = time.time()
            j = 0
            epoch_transfer_size = 0
            epoch_transfer_time = 0
            
            for bt in range(bt_num):
                print(f'j-{j}')
                j += 1
                a_node_batch, a_time_batch, p_node_batch, p_time_batch, n_node_batch, n_time_batch, batch_index = batch_l.getbatch_one()

                a_embedding = net(road_network, a_node_batch, a_time_batch)
                p_embedding = []
                n_embedding = []
                for x in range(3):
                    p_embedding.append(net(road_network, [sample[x] for sample in p_node_batch], [sample[x] for sample in p_time_batch]))
                    n_embedding.append(net(road_network, [sample[x] for sample in n_node_batch], [sample[x] for sample in n_time_batch]))

                loss = lossfunction(a_embedding, p_embedding, n_embedding, batch_index)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            s5 = time.time()
            print("Training Time: ", s5 - s1)
            
            comm_stats = net.get_communication_stats()
            print(f"Transmitted Data Amount: {comm_stats['total_transfer_mb']:.2f} MB")
            print(f"Transmission Speed: {comm_stats['avg_speed_mbs']:.2f} MB/s")
            
            epoch_stats = {
                "epoch": epoch,
                "total_transfer_mb": comm_stats['total_transfer_mb'],
                "avg_speed_mbs": comm_stats['avg_speed_mbs'],
                "transfer_details": comm_stats['transfer_details']
            }
            self.communication_stats_list.append(epoch_stats)
            
            if epoch % 2 == 0:
                net.eval()
                with torch.no_grad():
                    s6 = time.time()
                    vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='vali')
                    embedding_vali = test_method.compute_embedding(road_network=road_network, net=net,
                                                                   test_traj=list(vali_node_list),
                                                                   test_time=list(vali_d2vec_list),
                                                                   test_batch=self.test_batch)
                    acc = test_method.test_model(embedding_vali, isvali=True)
                    s7 = time.time()
                    print("Testing Time: ", s7 - s6)
                    print(f"HR@10: {acc[0]}, HR@50: {acc[1]}, HR@10-50: {acc[2]}, Loss: {loss.item()}")

                    save_modelname = './model/TP/{}_{}_epoch_{}_HR10_{}_HR50_{}_HR1050_{}_Loss_{}.pkl'.format(
                        self.dataset, self.distance_type, str(epoch), acc[0], acc[1], acc[2], loss.item())
                    torch.save(net.state_dict(), save_modelname)
                    best_model_path = save_modelname
                    if acc[0] > best_hr10:
                        best_hr10 = acc[0]
                        best_epoch = epoch
                    if epoch - best_epoch >= self.early_stop:
                        final_stats = net.get_communication_stats()
                        print("\nTraining Ends:")
                        print(f"Transmitted Data Amount: {final_stats['total_transfer_mb']:.2f} MB")
                        print(f"Transmission Speed: {final_stats['avg_speed_mbs']:.2f} MB/s")
                        print(f"Transmission Reports:")
                        for i, transfer in enumerate(final_stats['transfer_details']):
                            print(f"Transfer {i+1}: {transfer['size']:.2f} MB, "
                                  f"Time: {transfer['time']:.4f} s, "
                                  f"Speed: {transfer['speed']:.2f} MB/s")
                        break


                    '''
                    save_optname = './optimizer/{}/tdrive_TP_2w_ST/{}_{}_epoch_{}.pkl'.format(self.dataset, self.dataset,
                                                                                              self.distance_type,
                                                                                              str(epoch))
                    torch.save(optimizer.state_dict(), save_optname)
                    '''

        self.best_model = best_model_path

        with open("communication_stats.json", "w") as file:
            json.dump(self.communication_stats_list, file, ensure_ascii=False, indent=4)

    def traj_triplet_to_embedding(self):
        
        batch_l = data_utils.batch_list(self.test_batch)
        a_node_batch, a_time_batch, p_node_batch, p_time_batch, n_node_batch, n_time_batch, batch_index = batch_l.getbatch_one()

        net = STTrajSimEncoder(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size=self.date2vec_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               concat=self.concat,
                               device=self.device)

        if self.best_model:
            net.load_state_dict(torch.load(self.best_model))
            net.to(self.device)

            road_network = data_utils.load_netowrk(self.dataset).to(self.device)

            with torch.no_grad():
                a_embedding = net(road_network, a_node_batch, a_time_batch).cpu().numpy()
                print(a_embedding.shape)
                # p_embedding = net(road_network, p_node_batch, p_time_batch)
                # n_embedding = net(road_network, n_node_batch, n_time_batch)
                p_embedding = []
                n_embedding = []
                res = []
                for x in range(3):
                    print(x)
                    p_embedding.append(net(road_network, [sample[x] for sample in p_node_batch],
                                           [sample[x] for sample in p_time_batch]).cpu().numpy())
                    n_embedding.append(net(road_network, [sample[x] for sample in n_node_batch],
                                           [sample[x] for sample in n_time_batch]).cpu().numpy())

                res.append(a_embedding)
                res.append(p_embedding)
                res.append(n_embedding)
                res_arr = np.array(res)
                np.save("traj_triplet_embedding.npy", res_arr)

    def cal_traj_embedding_dis(self):
        dataload = data_utils.DataLoader()
        node_list_int, _, d2vec_list_int = dataload.load(load_part='all')

        N = len(node_list_int)
        res_matrix = np.zeros((N + 1, N + 1))

        net = STTrajSimEncoder(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size=self.date2vec_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               concat=self.concat,
                               device=self.device)

        if self.best_model:
            net.load_state_dict(torch.load(self.best_model))
            net.to(self.device)


            road_network = data_utils.load_netowrk(self.dataset).to(self.device)

            with torch.no_grad():
                total_embedding = test_method.compute_embedding(road_network=road_network, net=net,
                                                                test_traj=list(node_list_int),
                                                                test_time=list(d2vec_list_int),
                                                                test_batch=self.test_batch)
                print(total_embedding.shape)
                for i in range(N):
                    if i % 1000 == 0:
                        print(f'traj_embedding-{i}')
                    for j in range(i, N):
                        curr = np.linalg.norm((total_embedding[i] - total_embedding[j]).cpu())
                        res_matrix[i, j] = curr
                        res_matrix[j, i] = curr
                np.save("traj_embedding_dis.npy", res_matrix)

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())

    # train and test

    STsim = STsim_Trainer()

    # load_model_name = None
    load_model_name = './model/NetEDR_new/porto_NetEDR_epoch_48_HR10_0.110375_HR50_0.263875_HR1050_0.334875_Loss_0.002631083130836487.pkl'
    # load_model_name = "./model/tdrive_TP_2w_ST/tdrive_TP_epoch_66_HR10_0.5038273824813768_HR50_0.5989416902132032_HR1050_0.8521448754174159_Loss_0.004272001795470715.pkl"
    load_optimizer_name = None

    # STsim.ST_train(load_model = load_model_name,load_optimizer= load_optimizer_name)
    #
    s_time = time.time()
    STsim.ST_eval(load_model=load_model_name)
    print(time.time() - s_time)