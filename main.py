import argparse
import collections
import random
import time

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data_util import get_data
from model import HGNNModel
from nn_util import get_init_embd, clustering, split_data

start_time = time.time()


def train(epoch, args, model, optimizer, train_data_full, class_weights):
    model.train()
    loss_accum = 0
    new_train_data = []

    for train_data in train_data_full:
        train_size = len(train_data)
        # if epoch < 25 and train_size > 1000:
        #     continue
        idx_train = np.random.permutation(train_size)

        for i in range(0, train_size, args.batch_size):
            selected_idx = idx_train[i:i + args.batch_size]
            # batch_data = [train_data[idx] for idx in selected_idx]
            new_train_data.append([train_data[idx] for idx in selected_idx])

    sz = 0
    # sss = 0
    optimizer.zero_grad()
    loss = torch.zeros(1, device=class_weights.device)
    idx_train = np.random.permutation(len(new_train_data))
    for i in idx_train:
        batch_data = new_train_data[i]
        if len(batch_data) <= 1:
            continue
        t_idxs = []
        for j, d in enumerate(batch_data):
            if d.d_type == 0:
                t_idxs.append(j)
        if len(t_idxs) == 0:
            continue
        sz += len(t_idxs)
        # optimizer.zero_grad()
        output, targets, _ = model(batch_data)

        t_idxs = torch.tensor(t_idxs, device=output.device).long()
        output = output[t_idxs]
        targets = targets[t_idxs]
        # sss += len(output)
        loss += F.cross_entropy(output, targets)
        # if epoch > 3:
        #     loss += F.cross_entropy(output, targets)
        # else:
        #     loss += F.cross_entropy(output, targets, class_weights)

        if sz >= args.batch_size:
            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()

            sz = 0
            optimizer.zero_grad()
            loss = torch.zeros(1, device=class_weights.device)

    if sz > 0:
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()

    # print(sss)
    return loss_accum


def pass_data_iteratively(model, data_full, minibatch_size=128):
    pooled_h_ls = []
    data_new = []

    outputs = [[], [], []]
    targets = [[], [], []]

    for data in data_full:
        data_size = len(data)
        # full_idx = np.arange(data_size)
        full_idx = np.random.permutation(data_size)
        for i in range(0, data_size, minibatch_size):
            selected_idx = full_idx[i:i + minibatch_size]
            if len(selected_idx) == 0:
                continue
            batch_data = [data[idx] for idx in selected_idx]
            with torch.no_grad():
                output, target, pooled_h = model(batch_data)
            for j, d in enumerate(batch_data):
                outputs[d.d_type].append(output[j])
                targets[d.d_type].append(target[j])

            pooled_h_ls.append(pooled_h)
            data_new.extend(batch_data)

    output_train, target_train = torch.stack(outputs[0]), torch.stack(targets[0]).detach().cpu().numpy()
    output_dev, target_dev = torch.stack(outputs[1]), torch.stack(targets[1]).detach().cpu().numpy()
    output_test, target_test = torch.stack(outputs[2]), torch.stack(targets[2]).detach().cpu().numpy()

    pred_train = output_train.max(1, keepdim=True)[1].squeeze().detach().cpu().numpy()
    pred_dev = output_dev.max(1, keepdim=True)[1].squeeze().detach().cpu().numpy()
    pred_test = output_test.max(1, keepdim=True)[1].squeeze().detach().cpu().numpy()

    # print("train", target_train, pred_train)
    # print("dev", target_dev, pred_dev)
    # print("test", target_test, pred_test)

    acc_train = accuracy_score(target_train, pred_train)
    acc_dev = accuracy_score(target_dev, pred_dev)
    acc_test = accuracy_score(target_test, pred_test)

    return acc_train, acc_dev, acc_test, data_new, pooled_h_ls


def test(args, model, data_full):
    model.eval()

    acc_train, acc_dev, acc_test, data_full, pooled_h_ls = pass_data_iteratively(model, data_full, args.batch_size)

    pooled_h_full = torch.cat(pooled_h_ls, dim=0).detach().cpu().numpy()

    return acc_train, acc_dev, acc_test, data_full, pooled_h_full


def main():
    print('\n\ndate and time : ', time.ctime())
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--dataset', type=str, default="R8",
                        help='dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers INCLUDING the input one (default: 3)')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of layers for MLP EXCLUDING the input one (default: 1). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=200,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout (default: 0.5)')
    parser.add_argument('--filename', type=str, default="",
                        help='output file')
    parser.add_argument('--early_stop', type=int, default=30,
                        help='early_stop')
    parser.add_argument('--random_vec', action="store_true",
                        help='run in debug mode')
    parser.add_argument('--lda', action="store_true",
                        help='lda')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    args = parser.parse_args()

    if args.seed == -1:
        args.seed = random. randint(0, 1000)
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print('device : ', device, flush=True)

    train_data, dev_data, test_data, vocab_dic, labels_dic, class_weights, word_vectors \
        = get_data(dataset=args.dataset, lda=args.lda, seed=args.seed)

    num_classes = len(labels_dic)
    # num_clusters = (num_classes + 1) // 2
    num_clusters = 1
    # train_size, dev_size, test_size = len(train_data), len(dev_data), len(test_data)
    data_full = train_data + dev_data + test_data

    init_embed = get_init_embd(data_full, word_vectors).numpy()

    data_full_split_test = cluster_data(data_full, num_clusters, init_embed)
    data_full_split_train = data_full_split_test
    # data_full_split_train = [data_full]
    plot_tsne(init_embed, args.dataset + str(0))

    class_weights = torch.from_numpy(class_weights).float().to(device)
    input_dim = word_vectors.shape[1]

    model = HGNNModel(args, input_dim, num_classes, word_vectors, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.5)

    # print(model)
    print('')

    acc_test = 0
    max_acc_epoch, max_val_accuracy, test_accuracy = 0, 0, 0
    for epoch in range(1, args.epochs + 1):

        loss_accum = train(epoch, args, model, optimizer, data_full_split_train, class_weights)
        print('Epoch : ', epoch, 'loss training: ', loss_accum, 'Time : ', int(time.time() - start_time))

        acc_train, acc_dev, acc_test, data_full, embed = test(args, model, data_full_split_test)
        print("accuracy train: %f val: %f test: %f" % (acc_train, acc_dev, acc_test))
        if acc_dev > max_val_accuracy:
            max_val_accuracy = acc_dev
            max_acc_epoch = epoch
            test_accuracy = acc_test

        print('max validation accuracy : %f max acc epoch : %d test accuracy : %f'
              % (max_val_accuracy, max_acc_epoch, test_accuracy))

        plot_tsne(init_embed, args.dataset + str(epoch))
        if epoch == 20:
            model.word_embeddings.weight.requires_grad = True

        if epoch % 2 == 0:
            data_full_split_test = cluster_data(data_full, num_clusters, embed)
            # data_full_split_train = data_full_split_test

        # if epoch > 60:
        #     num_clusters = num_classes

        if epoch < 15:
            scheduler.step()
            print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        print('', flush=True)
        if epoch > max_acc_epoch + args.early_stop:
            break

    print('=' * 200)
    print('max acc epoch : ', max_acc_epoch)
    print('max validation accuracy : ', max_val_accuracy)
    print('test accuracy : ', test_accuracy)
    print('last test_accuracy : ', acc_test)
    print('=' * 200 + '\n')


def plot_tsne(embed, filename):
    tsne = TSNE()
    h = tsne.fit_transform(embed)
    plt.figure()
    plt.scatter(h[:, 0], h[:, 1])
    plt.savefig('tsne' + filename + '.png')


def cluster_data(data_full, num_clusters, embed):

    if num_clusters == 1:
        return [data_full]

    clusters = clustering(embed, num_clusters)
    elements_count = collections.Counter(clusters)
    print(elements_count)
    data_full_split = split_data(data_full, clusters)
    return data_full_split


if __name__ == '__main__':
    main()
