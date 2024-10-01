import tensorflow as tf
import numpy as np
import yaml
import os
import pickle
import agents.get_triplets_4 as data_utils
import agents.test_method as test_method
from agents.model_network import STTrajSimEncoder
from agents.lossfun import LossFun

# by dlhu, 05/2024

class STsim_Trainer: # /dlhu
    def __init__(self, actions):
        config = yaml.load(open('config.yaml'))

        self.feature_size = config["feature_size"]
        self.embedding_size = config["embedding_size"]
        self.date2vec_size = config["date2vec_size"]
        self.hidden_size = actions["hidden_size"]
        self.num_layers = actions["layer_num"]
        self.dropout_rate = config["dropout_rate"]
        self.concat = config["concat"]
        self.device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
        self.learning_rate = actions["learning_rate"]
        self.epochs = actions["training_epoch"]

        self.train_batch = config["train_batch"]
        self.test_batch = config["test_batch"]
        self.traj_file = str(config["traj_file"])
        self.time_file = str(config["time_file"])

        self.dataset = str(config["dataset"])
        self.distance_type = str(config["distance_type"])
        self.early_stop = config["early_stop"]

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
            net.load_weights(load_model)

            dataload = data_utils.DataLoader()
            road_network = data_utils.load_netowrk(self.dataset, self.device)

            vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='test')
            embedding_vali = test_method.compute_embedding(road_network, net, vali_node_list, self.test_batch)
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

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        lossfunction = LossFun(self.train_batch, self.distance_type)
        net.to(self.device)
        lossfunction.to(self.device)

        road_network = data_utils.load_netowrk(self.dataset, self.device)

        dataload = data_utils.DataLoader()
        batch_l = data_utils.batch_list(batch_size=self.train_batch)

        best_epoch = 0
        best_hr10 = 0
        last_epoch = 0

        if load_model and load_optimizer:
            net.load_weights(load_model)
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)  #需要修改 这里用actions['optim_method']
            optimizer.load_weights(load_optimizer)
            last_epoch = int(load_model.split('/')[-1].split('_')[3])
            best_epoch = last_epoch

        for epoch in range(last_epoch, self.epochs):
            for bt in range(int(dataload.return_triplets_num() / self.train_batch)):
                a_node_batch, p_node_batch, n_node_batch, batch_index = batch_l.getbatch_one() 

                with tf.GradientTape() as tape:
                    a_embedding = net(road_network, a_node_batch) 
                    p_embedding = net(road_network, p_node_batch)
                    n_embedding = net(road_network, n_node_batch)
                    loss = lossfunction(a_embedding, p_embedding, n_embedding, batch_index)

                gradients = tape.gradient(loss, net.trainable_variables)
                optimizer.apply_gradients(zip(gradients, net.trainable_variables))

            if epoch % 2 == 0:
                vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='vali')
                embedding_vali = test_method.compute_embedding(road_network, net, vali_node_list, self.test_batch)
                acc = test_method.test_model(embedding_vali, isvali=True)
                print(epoch, acc[1], loss.numpy())

                save_modelname = './model/tdrive_{}_2w_ST/{}_{}_epoch_{}_HR10_{}_HR50_{}_HR1050_{}_Loss_{}.pkl'.format(self.distance_type, self.dataset, self.distance_type, str(epoch), acc[0], acc[1], acc[2], loss.numpy())
                os.makedirs(os.path.dirname(save_modelname), exist_ok=True)
                net.save_weights(save_modelname)

                if acc[0] > best_hr10:
                    best_hr10 = acc[0]
                    best_epoch = epoch
                if epoch - best_epoch >= self.early_stop:
                    break

if __name__ == '__main__':
    print(tf.__version__)
    print(tf.cuda.device_count())
    print(tf.cuda.is_available())

    # train and test

    STsim = STsim_Trainer()

    # load_model_name = None
    load_model_name = './model/st2vec_1w/tdrive_TP_epoch_148_HR10_0.40258732212160414_HR50_0.591849935316947_HR1050_0.7420439844760672_Loss_0.002503135008737445.pkl'
    # load_model_name = "./model/tdrive_TP_2w_ST/tdrive_TP_epoch_66_HR10_0.5038273824813768_HR50_0.5989416902132032_HR1050_0.8521448754174159_Loss_0.004272001795470715.pkl"
    load_optimizer_name = None

    # STsim.ST_train(load_model = load_model_name,load_optimizer= load_optimizer_name)
    #
    STsim.ST_eval(load_model=load_model_name)