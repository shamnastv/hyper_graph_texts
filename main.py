import argparse
import collections
import time

import torch
import numpy as np
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
    optimizer.zero_grad()
    loss = torch.zeros(1, device=class_weights.device)
    idx_train = np.random.permutation(len(new_train_data))
    for i in idx_train:
        batch_data = new_train_data[i]
        # optimizer.zero_grad()
        output, targets, _ = model(batch_data)

        # loss = F.cross_entropy(output, targets)
        # loss = F.cross_entropy(output, targets, class_weights)
        loss += F.cross_entropy(output, targets)
        # loss += F.cross_entropy(output, targets, class_weights)
        sz += len(batch_data)

        if sz >= 2 * args.batch_size:
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

    return loss_accum


def pass_data_iteratively(model, data_full, minibatch_size=128):
    outputs = []
    targets = []
    pooled_h_ls = []
    data_new = []

    for data in data_full:
        data_size = len(data)
        full_idx = np.arange(data_size)
        for i in range(0, data_size, minibatch_size):
            selected_idx = full_idx[i:i + minibatch_size]
            if len(selected_idx) == 0:
                continue
            batch_data = [data[idx] for idx in selected_idx]
            with torch.no_grad():
                output, target, pooled_h = model(batch_data)
            outputs.append(output)
            targets.append(target)
            pooled_h_ls.append(pooled_h)
            data_new.extend(batch_data)

    outputs, targets = torch.cat(outputs, 0), torch.cat(targets, 0)

    pred = outputs.max(1, keepdim=True)[1].squeeze().detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    acc = accuracy_score(targets, pred)
    # print(classification_report(targets, pred))

    return acc, data_new, pooled_h_ls


def test(args, model, train_data, dev_data, test_data):
    model.eval()

    data_full, pooled_h_full = [], []
    acc_train, data, pooled_h_ls = pass_data_iteratively(model, train_data, args.batch_size)
    data_full += data
    pooled_h_full += pooled_h_ls

    acc_dev, data, pooled_h_ls = pass_data_iteratively(model, dev_data, args.batch_size)
    data_full += data
    pooled_h_full += pooled_h_ls

    acc_test, data, pooled_h_ls = pass_data_iteratively(model, test_data, args.batch_size)
    data_full += data
    pooled_h_full += pooled_h_ls

    pooled_h_full = torch.cat(pooled_h_full, dim=0).detach().cpu().numpy()
    return acc_train, acc_dev, acc_test, data_full, pooled_h_full


def main():
    print('date and time : ', time.ctime())
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--dataset', type=str, default="R8",
                        help='dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.0008,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers INCLUDING the input one (default: 3)')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of layers for MLP EXCLUDING the input one (default: 1). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=100,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout (default: 0.5)')
    parser.add_argument('--filename', type=str, default="",
                        help='output file')
    parser.add_argument('--early_stop', type=int, default=100,
                        help='early_stop')
    parser.add_argument('--debug', action="store_true",
                        help='run in debug mode')
    parser.add_argument('--random_vec', action="store_true",
                        help='run in debug mode')
    parser.add_argument('--lda', action="store_true",
                        help='lda')
    parser.add_argument('--weight_decay', type=float, default=1e-7,
                        help='weight decay')
    args = parser.parse_args()

    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print('device : ', device, flush=True)

    train_data, dev_data, test_data, vocab_dic, labels_dic, class_weights, word_vectors \
        = get_data(args.dataset, args.lda)

    num_classes = len(labels_dic)
    num_clusters = num_classes
    train_size, dev_size, test_size = len(train_data), len(dev_data), len(test_data)
    data_full = train_data + dev_data + test_data

    init_embed = get_init_embd(data_full, word_vectors).numpy()

    dev_data, test_data, train_data = cluster_data(data_full, num_clusters, init_embed, dev_size, train_size, test_size)

    class_weights = torch.from_numpy(class_weights).float().to(device)
    input_dim = word_vectors.shape[1]

    model = HGNNModel(args, input_dim, num_classes, word_vectors, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=.5)

    print(model)

    acc_test = 0
    max_acc_epoch, max_val_accuracy, test_accuracy = 0, 0, 0
    for epoch in range(1, args.epochs + 1):

        loss_accum = train(epoch, args, model, optimizer, train_data, class_weights)
        print('Epoch : ', epoch, 'loss training: ', loss_accum, 'Time : ', int(time.time() - start_time))

        acc_train, acc_dev, acc_test, data_full, embed = test(args, model, train_data, dev_data, test_data)
        print("accuracy train: %f val: %f test: %f" % (acc_train, acc_dev, acc_test))
        if acc_dev > max_val_accuracy:
            max_val_accuracy = acc_dev
            max_acc_epoch = epoch
            test_accuracy = acc_test

        print('max validation accuracy : %f max acc epoch : %d test accuracy : %f'
              % (max_val_accuracy, max_acc_epoch, test_accuracy), flush=True)

        if epoch % 1 == 0:
            dev_data, test_data, train_data = cluster_data(data_full, num_clusters, embed,
                                                           dev_size, train_size, test_size)
        if epoch > 60:
            num_clusters = num_classes
        # scheduler.step()
        print('')
        if epoch > max_acc_epoch + args.early_stop:
            break

    print('=' * 200)
    print('max acc epoch : ', max_acc_epoch)
    print('max validation accuracy : ', max_val_accuracy)
    print('test accuracy : ', test_accuracy)
    print('last test_accuracy : ', acc_test)
    print('=' * 200 + '\n')


def cluster_data(data_full, num_classes, embed, dev_size, train_size, test_size):
    clusters = clustering(embed, num_classes)
    cluster_train = clusters[:train_size]
    cluster_dev = clusters[train_size:train_size + dev_size]
    cluster_test = clusters[train_size + dev_size:train_size + dev_size + test_size]
    elements_count = collections.Counter(clusters)
    print(elements_count)
    train_data = split_data(data_full[:train_size], cluster_train)
    dev_data = split_data(data_full[train_size:train_size + dev_size], cluster_dev)
    test_data = split_data(data_full[train_size + dev_size:train_size + dev_size + test_size], cluster_test)
    return dev_data, test_data, train_data


if __name__ == '__main__':
    main()
