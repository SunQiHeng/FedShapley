#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from src.utils.sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, cifar_iid, cifar_noniid, \
    FashionMnist_noniid
from src.utils.options import args_parser
import ssl


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    ssl._create_default_https_context = ssl._create_unverified_context

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset_all = datasets.CIFAR10(data_dir, train=False, download=True,
                                            transform=apply_transform)
        test_dataset, valid_dataset = torch.utils.data.random_split(test_dataset_all, [5000, 5000])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist':
        data_dir = '../data/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset_all = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

        test_dataset, valid_dataset = torch.utils.data.random_split(test_dataset_all, [5000, 5000])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'fmnist':
        data_dir = '../data/fmnist'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                              transform=apply_transform)

        test_dataset_all = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                 transform=apply_transform)

        test_dataset, valid_dataset = torch.utils.data.random_split(test_dataset_all, [5000, 5000])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = FashionMnist_noniid(train_dataset, args.num_users)

    indices = []
    for i in range(len(valid_dataset)):
        data, label = valid_dataset[i]
        if label != 0 and label != 5 and label != 9:
            indices.append(i)
    new_valid_dataset = torch.utils.data.Subset(valid_dataset, indices)
    indices = []
    for i in range(len(test_dataset)):
        data, label = test_dataset[i]
        if label != 0 and label != 5 and label != 9:
            indices.append(i)
    new_test_dataset = torch.utils.data.Subset(test_dataset, indices)
    return train_dataset, new_valid_dataset, new_test_dataset, user_groups
    #return train_dataset, valid_dataset, test_dataset, user_groups
    # return train_dataset, test_dataset, user_groups

def average_weights(w):
    """
    最正常的平均
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def avgSV_weights(w, shapley):
    """
        Shapley权值平均
        Returns the average of the weights.
    """
    s = 0
    first = 0
    Flag = True
    for i in range(len(shapley)):
        if shapley[i] > 0:
            if Flag:
                first = i
                Flag = False
            s += shapley[i]
    w_avg = copy.deepcopy(w[first])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * shapley[first] / s
        for i in range(1, len(w)):
            if shapley[i] > 0:
                w_avg[key] += w[i][key] * shapley[i] / s
    return w_avg


def avgSVAtt_weights(w, shapley):
    """
    最正常的平均
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * shapley[0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * shapley[i]
    return w_avg


def SVAtt_weights(w, shapley, original_weights, learning_rate, epoch):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if epoch == 0:
            w_avg[key] = w_avg[key] * shapley[0]
        else:
            w_avg[key] = w_avg[key] * shapley[0] * learning_rate + (1 - learning_rate) * original_weights[key]
        for i in range(1, len(w)):
            if epoch == 0:
                w_avg[key] += w[i][key] * shapley[i]
            else:
                w_avg[key] += w[i][key] * shapley[i] * learning_rate
    return w_avg


def SVAtt2_weights(w, original_weights, learning_rate, att):
    w_avg = copy.deepcopy(w[0])
    temp_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        temp_avg[key] = att[0] * (original_weights[key] - temp_avg[key])
        for i in range(1, len(w)):
            temp_avg[key] += att[i] * (original_weights[key] - w[i][key])
        w_avg[key] = original_weights[key] - learning_rate * temp_avg[key]

    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset   : {args.dataset}')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
