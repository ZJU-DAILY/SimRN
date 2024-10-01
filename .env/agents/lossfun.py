import numpy as np
import tensorflow as tf
import math
import yaml
import pandas as pd

class LossFun(tf.Module):
    def __init__(self,train_batch,distance_type):
        super(LossFun, self).__init__()
        self.train_batch = train_batch
        self.distance_type = distance_type
        config = yaml.load(open('config.yaml'))
        self.triplets_dis = np.load(str(config["path_triplets_truth"]))

    def forward(self,embedding_a,embedding_p,embedding_n,batch_index):

        batch_triplet_dis = self.triplets_dis[batch_index]
        batch_loss = 0.0

        for i in range(self.train_batch):

            ap_loss = 0.0
            an_loss = 0.0
            for j in range(len(embedding_p)):

                D_ap = math.exp(-batch_triplet_dis[i][j][0])
                v_ap = np.exp(-tf.norm(embedding_a[i], embedding_p[j][i], p=2)) # torch.dist
                ap_loss += D_ap * ((D_ap - v_ap) ** 2)

                D_an = math.exp(-batch_triplet_dis[i][j][1])
                v_an = np.exp(-tf.norm(embedding_a[i], embedding_n[j][i], p=2)) # torch.dist
                an_loss += D_an * ((D_an - v_an) ** 2)

            oneloss = ap_loss + an_loss
            batch_loss += oneloss

        mean_batch_loss = batch_loss / self.train_batch
        sum_batch_loss = batch_loss

        return mean_batch_loss
