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
max_acc_epoch, max_val_accuracy, test_accuracy = 0, 0, 0


def train(args, epoch, model, optimizer, train_data):
    model.train()

    train_size = len(train_data)
    idx_train = np.random.permutation(train_size)

    loss_accum = 0
    for i in range(0, train_size, args.batch_size):
        selected_idx = idx_train[i:i + args.batch_size]
        batch_data = [train_data[idx] for idx in selected_idx]
        output, targets = model(batch_data)

        loss = F.cross_entropy(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

    print('Epoch : ', epoch, 'loss training: ', loss_accum, 'Time : ', int(time.time() - start_time))
    return loss_accum


def pass_data_iteratively(model, data, minibatch_size=128):
    outputs = []
    targets = []

    full_idx = np.arange(len(data))
    for i in range(0, len(data), minibatch_size):
        selected_idx = full_idx[i:i + minibatch_size]
        if len(selected_idx) == 0:
            continue
        batch_data = [data[idx] for idx in selected_idx]
        with torch.no_grad():
            output, target = model(batch_data)
        outputs.append(output)
        targets.append(target)

    return torch.cat(outputs, 0), torch.cat(targets, 0)


def test(epoch, model, train_data, dev_data, test_data):
    model.eval()

    output_train, target_train = pass_data_iteratively(model, train_data)
    pred_train = output_train.max(1, keepdim=True)[1]
    correct = pred_train.eq(target_train.view_as(pred_train)).sum().cpu().item()
    acc_train = correct / float(len(train_data))

    output_dev, target_dev = pass_data_iteratively(model, dev_data)
    pred_dev = output_dev.max(1, keepdim=True)[1]
    correct = pred_dev.eq(target_dev.view_as(pred_dev)).sum().cpu().item()
    acc_dev = correct / float(len(dev_data))

    output_test, target_test = pass_data_iteratively(model, test_data)
    pred_test = output_test.max(1, keepdim=True)[1]
    correct = pred_test.eq(target_test.view_as(pred_test)).sum().cpu().item()
    acc_test = correct / float(len(test_data))

    print("accuracy train: %f val: %f test: %f" % (acc_train, acc_dev, acc_test), flush=True)
    global max_acc_epoch, max_val_accuracy, test_accuracy
    if acc_dev > max_val_accuracy:
        max_val_accuracy = acc_dev
        max_acc_epoch = epoch
        test_accuracy = acc_test

    print('max validation accuracy : ', max_val_accuracy, 'max acc epoch : ', max_acc_epoch, flush=True)


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
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.5,
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

    # set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    print('device : ', device, flush=True)

    train_data, dev_data, test_data, vocab_dic, labels_dic, class_weights = get_data(args.dataset)

    word_vectors = get_embedding(vocab_dic)
    input_dim = word_vectors.shape[1]
    num_classes = len(labels_dic)

    model = HGNNModel(args, input_dim, num_classes, word_vectors, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, epoch, model, optimizer, train_data)
        test(epoch, model, train_data, dev_data, test_data)


if __name__ == '__main__':
    main()