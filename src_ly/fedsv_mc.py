#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import random

import numpy as np
from tqdm import tqdm, trange

import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from src.utils.options import args_parser
from src.utils.update import LocalUpdate, test_inference
from src.utils.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from src.util import get_dataset,average_weights, exp_details, SVAtt_weights,avgSV_weights
from src.utils.plot import draw
from heapq import nlargest

def get_weights(j, idx, local_ws):
    test_weight = []
    for i in range(j):
        current_weight = local_ws[idx[i]]
        test_weight.append(current_weight)
    return test_weight

def solver():
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    # device = 'cuda' if args.gpu else 'cpu'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset and user groups
    train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(args)


    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()
    original_weights = copy.deepcopy(global_weights)

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    allAcc_list = []
    m = max(int(args.frac * args.num_users), 1)
    beta_parameter = np.ones([args.num_users,2])


    for epoch in tqdm(range(args.epochs)):

        local_weights, local_losses = [], []
        clients_acc, clients_losses = [], []

        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()

        shapley = np.zeros(m)
        # beta_rand = np.zeros(len(beta_parameter))
        # for i in range(len(beta_parameter)):
        #     beta_rand[i] = np.random.beta(beta_parameter[i][0], beta_parameter[i][1])
        # idxs_users = nlargest(m,range(len(beta_rand)),beta_rand.__getitem__)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        index = [i for i in range(m)]
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # 计算每个客户端shapley值
        for step in trange(60):
            index = np.random.permutation(len(index))
            original_acc = 0
            for j in range(1, len(index)):
                if j >= 5:
                    break
                test_weights = get_weights(j, index, local_weights)
                test_weight = average_weights(test_weights)
                global_model.load_state_dict(test_weight)
                global_model.eval()
                current_acc, current_loss = test_inference(args, global_model, valid_dataset)
                shapley[index[j - 1]] += current_acc - original_acc
                original_acc = current_acc

        shapley = [shap / 100 for shap in shapley]

        # for i in range(len(shapley)):
        #     if shapley[i] > np.mean(np.array(shapley),axis=None):
        #         beta_parameter[idxs_users[i]][0] += 1
        #     else:
        #         beta_parameter[idxs_users[i]][1] += 1
        # shapley = F.softmax(torch.tensor(shapley),dim=0)
        #
        # # update global weights
        #
        # global_weights = SVAtt_weights(local_weights, shapley, original_weights, 1-0.005*epoch, epoch)
        global_weights = avgSV_weights(local_weights, shapley)
        original_weights = copy.deepcopy(global_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        allAcc_list.append(test_acc)
        print(" \nglobal accuracy:{:.2f}%".format(100 * test_acc))

    #draw(args.epochs, allAcc_list, "SV 10 100")

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

if __name__ == '__main__':
    test_acc, train_acc = 0, 0
    for _ in range(5):
        print("|---- 第「{}」次 ----|".format(_ + 1))
        solver()
