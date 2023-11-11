import numpy as np
import torch
import os
import torch.utils.data as Data
from tqdm import tqdm
import argparse
import Model
import time
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    parser = argparse.ArgumentParser(description='Transformer-csi')
    parser.add_argument('--model', type=str, default='HARTrans',
                        help='model')
    parser.add_argument('--dataset', type=str, default='53001_npy',
                        help='dataset')
    parser.add_argument('--sample', type=int, default=1,
                        help='sample length on temporal side [default:4]')
    parser.add_argument('--batch', type=int, default=8,
                        help='batch size [default: 16]')
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='learning rate [default: 0.001]')
    parser.add_argument('--epoch', type=int, default=50,
                        help='number of epoch [default: 20]')
    parser.add_argument('--hlayers', type=int, default=7,
                        help='horizontal transformer layers [default: 6]')
    parser.add_argument('--hheads', type=int, default=10,
                        help='horizontal transformer head [default: 9]')
    parser.add_argument('--vlayers', type=int, default=1,
                        help='vertical transformer layers [default: 1]')
    parser.add_argument('--vheads', type=int, default=50,
                        help='vertical transformer head [default: 200]')
    parser.add_argument('--category', type=int, default=5,
                        help='category [default: 7]')
    parser.add_argument('--com_dim', type=int, default=50,
                        help='compressor vertical transformer layers [default: 50]')
    parser.add_argument('--K', type=int, default=10,
                        help='number of Gaussian distributions [default: 10]')
    parser.add_argument('--input_length', type=int, default=350,
                        help='input_length for avg pooling in raw data [default: 2000]')
    parser.add_argument('--output_model_dir', type=str, default='./1p_NOISY_runs',
                    help='directory to store finetuned models')
    args = parser.parse_args()
    return args

args = get_args()


def get_model_class(model_name, args):
    model_list = ['TransCNN', 'Transformer', 'HARTrans']
    for x in model_list:
        if x.find(model_name) != -1:
            AClass = getattr(Model, x)(args)
    return AClass


def load_data(root):
    data = torch.load("/home/cynthia/transformer_har/Data_noisy_1p_w350.pt")
    aclist = ["jump", "run", "sit", "stand", "walk"]
    return data, aclist


def data_loader(data):
    loader = Data.DataLoader(
        dataset=data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=1,
    )
    return loader


def gen_conf_matrix(pred, truth, conf_matrix):
    p = pred.cpu().tolist()
    l = truth.cpu().tolist()
    for i in range(len(p)):
        conf_matrix[l[i]][p[i]] += 1
    return conf_matrix


def write_to_file(conf_matrix):
    f = open("conf_matrix_1p_noisy.txt", mode='w+', encoding='utf-8')
    for x in range(len(conf_matrix)):
        base = sum(conf_matrix[x])
        for y in range(len(conf_matrix[0])):
            value = str(format(conf_matrix[x][y]/base, '.2f'))
            f.write(value+'&')
        f.write('\n')
    f.flush()
    f.close()


def c_main():
    # Set up logging
    logging = True
    if logging and not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)
    logfile_name = os.path.join(args.output_model_dir, "THAT_1p_noisy_test19.txt")

    if logging:
        logfile = open(logfile_name, "w")

    if logging:
        logfile.write(f"model_name = {args.model}, lr = {args.lr}, sample = {args.sample}, batch_size = {args.batch}, epochs = {args.epoch}\n")
        logfile.write(f"input_length = {args.input_length}, hlayers = {args.hlayers}, hheads = {args.hheads}, vlayers = {args.vlayers}, vheads = {args.vheads}, com_dim = {args.com_dim}, K = {args.K}\n")
        logfile.write("\n")
        logfile.write("\n")

    dataset, aclist = load_data(args.dataset)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_data = data_loader(train_dataset)
    test_data = data_loader(test_dataset)
    model = get_model_class(args.model, args)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = args.epoch
    best = 0.0
    best_conf_matrix = [[0 for _ in range(len(aclist))] for _ in range(len(aclist))]

    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        tr_acc = 0.
        total_num = 0
        print("\nEpoch{}/{}".format(epoch, n_epochs))
        print("-" * 10)
        steps = len(train_data)
        model.train()
        time_start = time.time()
        for batch in tqdm(train_data):
            X_train, Y_train = batch
            Y_train = Y_train.long()
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            outputs = model(X_train)
            pred = torch.max(outputs, 1)[1]
            loss = criterion(outputs, Y_train)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
            optimizer.step()
            running_loss += loss.item()
            running_correct = (pred.cpu() == Y_train.cpu()).sum()
            tr_acc += running_correct.item()
            total_num += len(batch[0])
        print(f"Model loss after {epoch+1} epochs = {criterion(outputs, Y_train)}")
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        running_loss = 0.0
        running_correct = 0
        tr_acc = 0.
        total_num = 0
        print("\nStart validation")
        print("-" * 10)
        steps = len(train_data)
        model.eval()
        conf_matrix = [[0 for _ in range(len(aclist))] for _ in range(len(aclist))] ## ADD ME

        time_start = time.time()
        for batch in tqdm(test_data):
            X_train, Y_train = batch
            Y_train = Y_train.long()
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            outputs = model(X_train)
            pred = torch.max(outputs, 1)[1]
            running_correct = (pred.cpu() == Y_train.cpu()).sum()
            conf_matrix = gen_conf_matrix(pred, Y_train, conf_matrix)
            tr_acc += running_correct.item()
            total_num += len(batch[0])
            acc = tr_acc/total_num
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        print("\nAccuracy is", tr_acc/total_num)
        if best < acc:
            best = acc
            best_conf_matrix = conf_matrix
        print("\nBest is", best)
        
        if logging:
            logfile.write(f"Model loss after {epoch+1} epochs = {criterion(outputs, Y_train)}, accuracy = {tr_acc/total_num}, best_accuracy = {best}, time_cost = {time_end - time_start}s \n")

    if logging:
        logfile.write(f"\n Final confusion matrix:\n")
        for x in range(len(best_conf_matrix)):
            base = sum(best_conf_matrix[x])
            for y in range(len(best_conf_matrix[0])):
                value = str(format(best_conf_matrix[x][y]/base, '.2f'))
                logfile.write(value+'&')
            logfile.write('\n')

        logfile.close()


def cross_validation():
    dataset, aclist = load_data("args.dataset")
    k = 5
    results = {} # fold results
    torch.manual_seed(42) # set random seed
    kfold = KFold(n_splits=k, shuffle=True)
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = Data.SubsetRandomSampler(train_ids)
        test_subsampler = Data.SubsetRandomSampler(test_ids)
        # Define data loaders for training and testing data in this fold
        trainloader = Data.DataLoader(dataset, batch_size=args.batch, sampler=train_subsampler)
        testloader = Data.DataLoader(dataset, batch_size=args.batch, sampler=test_subsampler)

        model = get_model_class(args.model, args)
        if torch.cuda.is_available():
            model = model.cuda()

        # change loss function to account for multilabel
        criterion = torch.nn.NLLLoss() # original single label loss

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # added weight decay
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_epochs = args.epoch
        best = 0.0

        for epoch in range(n_epochs):
            conf_matrix = [[0 for _ in range(len(aclist))] for _ in range(len(aclist))]
            running_loss = 0.0
            running_correct = 0
            tr_acc = 0.
            total_num = 0
            print("\nEpoch{}/{}".format(epoch, n_epochs))
            print("-" * 10)
            #print("\n")
            steps = len(train_ids)
            model.train()
            time_start = time.time()
            for batch in tqdm(trainloader):
                X_train, Y_train = batch
                Y_train = Y_train.long()
                X_train, Y_train = X_train.to(device), Y_train.to(device)

                # prediction
                outputs = model(X_train) # size = batch_size x 5
                pred = torch.max(outputs, 1)[1]

                loss = criterion(outputs, Y_train)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item()
                running_correct = (pred.cpu() == Y_train.cpu()).sum()
                tr_acc += running_correct.item()
                total_num += len(batch[0])

            time_end = time.time()
            print(f"Model loss after {epoch+1} epochs = {criterion(outputs, Y_train)}")
            print('time cost', time_end - time_start, 's')
            running_loss = 0.0
            running_correct = 0
            tr_acc = 0.
            total_num = 0
            print("\nStart validation")
            print("-" * 10)
            steps = len(test_ids)
            model.eval()

            time_start = time.time()
            cm = [[0 for _ in range(len(aclist))] for _ in range(len(aclist))] # initialise confusion matrix
            # validation
            for batch in tqdm(testloader):
                X_train, Y_train = batch
                Y_train = Y_train.long()
                X_train, Y_train = X_train.to(device), Y_train.to(device)
                outputs = model(X_train)
                pred = torch.max(outputs, 1)[1]
                running_correct = (pred.cpu() == Y_train.cpu()).sum()
                conf_matrix = gen_conf_matrix(pred, Y_train, conf_matrix)
                tr_acc += running_correct.item()
                total_num += len(batch[0])
                acc = tr_acc/total_num

                tr_acc += running_correct.item()
                total_num += len(batch[0])

                acc = tr_acc/total_num
            time_end = time.time()
            print('time cost', time_end - time_start, 's')
            print("\nAccuracy is", tr_acc/total_num)
            print("confusion matrix is\n", cm)
            if best < acc:
                best = acc
                best_cm = conf_matrix
            print("\nBest is", best)
        results[fold] = (best, best_cm)

    logfile = open("1p_noisy_cross_validation.txt", "w")
    for key in results.keys():
        logfile.write(f"this is fold {key}\n")
        logfile.write(f"the accuracy is {results[key][0]}")
        logfile.write(f"\n Final confusion matrix:\n")
        conf_matrix = results[key][1]
        for x in range(len(conf_matrix)):
            base = sum(conf_matrix[x])
            for y in range(len(conf_matrix[0])):
                value = str(format(conf_matrix[x][y]/base, '.2f'))
                logfile.write(value+'&')
            logfile.write('\n')
    logfile.close()


if __name__=="__main__":
    try:
        # c_main()
        cross_validation()
    except KeyboardInterrupt:
        print("error")
