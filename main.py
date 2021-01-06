import argparse
import time

import torch
import numpy as np
from torch import optim
import torch.nn.functional as F

from data_util import get_data
from model import HGNNModel
from preprocess import get_embedding


start_time = time.time()


def train(args, model, optimizer, train_data):
    model.train()

    train_size = len(train_data)
    idx_train = np.random.permutation(train_size)

    loss_accum = 0
    for i in range(0, train_size, args.batch_size):
        selected_idx = idx_train[i:i + args.batch_size]
        batch_data = [train_data[idx] for idx in selected_idx]
        output, targets = model(batch_data)

        optimizer.zero_grad()
        loss = F.cross_entropy(output, targets)
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

    return loss_accum


def pass_data_iteratively(model, data, minibatch_size=128):
    outputs = []
    targets = []

    data_size = len(data)
    full_idx = np.arange(data_size)
    for i in range(0, data_size, minibatch_size):
        selected_idx = full_idx[i:i + minibatch_size]
        if len(selected_idx) == 0:
            continue
        batch_data = [data[idx] for idx in selected_idx]
        with torch.no_grad():
            output, target = model(batch_data)
        outputs.append(output)
        targets.append(target)

    outputs, targets = torch.cat(outputs, 0), torch.cat(targets, 0)

    pred = outputs.max(1, keepdim=True)[1]
    correct = pred.eq(targets.view_as(pred)).sum().cpu().item()
    acc = correct / float(len(data))

    return acc


def test(model, train_data, dev_data, test_data):
    model.eval()

    acc_train = pass_data_iteratively(model, train_data)
    acc_dev = pass_data_iteratively(model, dev_data)
    acc_test = pass_data_iteratively(model, test_data)

    return acc_train, acc_dev, acc_test


def main():
    print('date and time : ', time.ctime())
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=200,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout (default: 0.5)')
    parser.add_argument('--dataset', type=str, default="R52",
                        help='dataset')
    parser.add_argument('--filename', type=str, default="",
                        help='output file')
    parser.add_argument('--early_stop', type=int, default=30,
                        help='early_stop')
    parser.add_argument('--debug', action="store_true",
                        help='run in debug mode')
    args = parser.parse_args()

    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    print('device : ', device, flush=True)

    train_data, dev_data, test_data, vocab_dic, labels_dic, class_weights, word_vectors = get_data(args.dataset)

    # word_vectors = get_embedding(vocab_dic)
    input_dim = word_vectors.shape[1]
    num_classes = len(labels_dic)

    model = HGNNModel(args, input_dim, num_classes, word_vectors, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    acc_test = 0
    max_acc_epoch, max_val_accuracy, test_accuracy = 0, 0, 0
    for epoch in range(1, args.epochs + 1):

        loss_accum = train(args, model, optimizer, train_data)
        print('Epoch : ', epoch, 'loss training: ', loss_accum, 'Time : ', int(time.time() - start_time))

        acc_train, acc_dev, acc_test = test(model, train_data, dev_data, test_data)
        print("accuracy train: %f val: %f test: %f" % (acc_train, acc_dev, acc_test), flush=True)
        if acc_dev > max_val_accuracy:
            max_val_accuracy = acc_dev
            max_acc_epoch = epoch
            test_accuracy = acc_test

        print('max validation accuracy :', max_val_accuracy, 'max acc epoch :', max_acc_epoch,
              'test accuracy :', test_accuracy, flush=True)

        print('')
        if epoch > max_acc_epoch + args.early_stop:
            break

    print('=' * 200)
    print('max acc epoch : ', max_acc_epoch)
    print('max validation accuracy : ', max_val_accuracy)
    print('test accuracy : ', test_accuracy)
    print('latest_test_accuracy : ', acc_test)
    print('=' * 200 + '\n')


if __name__ == '__main__':
    main()
