import numpy as np
import torch
import os
import torch.utils.data as Data
from tqdm import tqdm
import argparse
import Model_2p_multilabel
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
    parser.add_argument('--sample', type=int, default=4,
                        help='sample length on temporal side')
    parser.add_argument('--batch', type=int, default=4,
                        help='batch size [default: 16]')
    parser.add_argument('--lr', type=float, default=5e-05,
                        help='learning rate [default: 0.001]')
    parser.add_argument('--wd', type=float, default=1e-05,
                        help='weight decay [default: 0.0]')
    parser.add_argument('--epoch', type=int, default=200,
                        help='number of epoch [default: 20]')
    parser.add_argument('--hlayers', type=int, default=9,
                        help='horizontal transformer layers [default: 6]')
    parser.add_argument('--hheads', type=int, default=54,
                        help='horizontal transformer head [default: 9]')
    parser.add_argument('--vlayers', type=int, default=6,
                        help='vertical transformer layers [default: 1]')
    parser.add_argument('--vheads', type=int, default=350,
                        help='vertical transformer head [default: 200]')
    parser.add_argument('--category', type=int, default=6,
                        help='category [default: 7]')
    parser.add_argument('--com_dim', type=int, default=50, #not use in har
                        help='compressor vertical transformer layers [default: 50]')
    parser.add_argument('--K', type=int, default=7,
                        help='number of Gaussian distributions [default: 10]')
    parser.add_argument('--filter_size', type=list, default= [10, 35, 70, 115],
                        help='temporal filter sizes in encoder. default: [10, 40]')
    parser.add_argument('--filter_size_v', type=list, default=[2, 8, 12],
                        help='channel filter sizes in encoder. default: [2, 4]')
    parser.add_argument('--kernel_num', type=int, default=256,
                        help='temporal nb of filter in encoder [default: 128]')
    parser.add_argument('--kernel_num_v', type=int, default=16,
                        help='channel nb of filter in encoder  [default: 16]')
    parser.add_argument('--input_length', type=int, default=700,
                        help='input_length for avg pooling in raw data [default: 2000]')
    parser.add_argument('--dropout_rate', type=int, default=0.5,
                        help='dropout_rate [default: 0.5]')
    parser.add_argument('--clip_gradient', type=int, default=0.8,
                        help='clip_gradient [default: 1.0]')
    parser.add_argument('--output_model_dir', type=str, default='./2p_NOISY_runs', ## CHANGE DEFAULT
                    help='directory to store finetuned models') ## ADD ME
    args = parser.parse_args()
    return args

args = get_args()


def get_model_class(model_name, args):
    model_list = ['TransCNN', 'Transformer', 'HARTrans']
    for x in model_list:
        if x.find(model_name) != -1:
            AClass = getattr(Model_2p_multilabel, x)(args)
    return AClass


def load_data(root):
    data = torch.load("/home/cynthia/transformer_har/Data_noisy_2p_w700.pt")
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
    f = open("conf_matrix_2p_clean.txt", mode='w+', encoding='utf-8')
    for x in range(len(conf_matrix)):
        base = sum(conf_matrix[x])
        for y in range(len(conf_matrix[0])):
            value = str(format(conf_matrix[x][y]/base, '.2f'))
            f.write(value+'&')
        f.write('\n')
    f.flush()
    f.close()


def get_multilabel_conf_matrix(pred, truth, conf_matrix):
    cm = multilabel_confusion_matrix(pred, truth)
    if conf_matrix is not None:
        new_cm = conf_matrix + cm
    else: 
        new_cm = cm

    return new_cm


def c_main():
    # Set up logging
    logging = True
    if logging and not os.path.exists(args.output_model_dir):
        os.makedirs(args.output_model_dir)
    logfile_name = os.path.join(args.output_model_dir, "THAT_2p_noisy_test111.txt") ## CHANGE ME

    if logging:
        logfile = open(logfile_name, "w")

    if logging:
        logfile.write(f"model_name = {args.model}, lr = {args.lr}, wd = lr = {args.wd} sample = {args.sample}, batch_size = {args.batch}, epochs = {args.epoch}\n")
        logfile.write(f"input_length = {args.input_length}, hlayers = {args.hlayers}, hheads = {args.hheads}, vlayers = {args.vlayers}, vheads = {args.vheads}\n")
        logfile.write(f"filter_size = {args.filter_size}, filter_size_v = {args.filter_size_v}, kernel_num = {args.kernel_num}, kernel_num_v = {args.kernel_num_v}\n")
        logfile.write(f"dropout_rate = {args.dropout_rate}, com_dim = {args.com_dim}, K = {args.K}, clip_gradient = {args.clip_gradient}\n")
        logfile.write("\n")
        logfile.write("\n")

    dataset, aclist = load_data("args.dataset")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_data = data_loader(train_dataset)
    test_data = data_loader(test_dataset)
    model = get_model_class(args.model, args)
    if torch.cuda.is_available():
        model = model.cuda()

    # change loss function to account for multilabel
    #criterion = torch.nn.NLLLoss() # original single label loss
    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) # added weight decay
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = args.epoch
    best = 0.0

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

            # prediction
            outputs = model(X_train) # size = batch_size x 5
            pred = outputs>0.5

            loss = criterion(outputs.float(), Y_train.float())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
            optimizer.step()
            running_loss += loss.item()
            running_correct = (pred.cpu() == Y_train.cpu()).sum()
            tr_acc += running_correct.item()
            total_num += len(batch[0])
        print("training accuracy = ", tr_acc/total_num)
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
        conf_matrix = [[0 for _ in range(len(aclist))] for _ in range(len(aclist))]

        time_start = time.time()
        for batch in tqdm(test_data):
            X_train, Y_train = batch
            Y_train = Y_train.long()
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            outputs = model(X_train)
            
            # switch prediction to multilabel
            pred = outputs>0.5

            running_correct = torch.all(pred.cpu() == Y_train.cpu(), dim=1).sum() # multilabel

            tr_acc += running_correct.item()
            total_num += len(batch[0])

            acc = tr_acc/total_num
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        print("\nAccuracy is", tr_acc/total_num)
        if best < acc:
            best = acc
            cm = multilabel_confusion_matrix(Y_train.cpu(), pred.cpu())

        print(f"Model loss after {epoch+1} epochs = {criterion(outputs.float(), Y_train.float())}")
        print("\nBest is", best)
        if logging:
            logfile.write(f"Model loss after {epoch+1} epochs = {criterion(outputs.float(), Y_train.float())}, accuracy = {tr_acc/total_num}, best_accuracy = {best}, time_cost = {time_end - time_start}s \n")

    if logging:
        logfile.write(f"\n Final confusion matrix:\n")
        for i in range(len(cm)):
            logfile.write(f"\nActivity {i+1}:\n")
            for j in range(len(cm[i])):
                logfile.write(str(cm[i, j]) + "\n")
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
        criterion = torch.nn.BCELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) # added weight decay
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_epochs = args.epoch
        best = 0.0

        for epoch in range(n_epochs):
            running_loss = 0.0
            running_correct = 0
            tr_acc = 0.
            total_num = 0
            print("\nEpoch{}/{}".format(epoch, n_epochs))
            print("-" * 10)
            steps = len(train_ids)
            model.train()
            time_start = time.time()
            for batch in tqdm(trainloader):
                X_train, Y_train = batch
                Y_train = Y_train.long()
                X_train, Y_train = X_train.to(device), Y_train.to(device)

                # prediction
                outputs = model(X_train) # size = batch_size x 5
                pred = outputs>0.5

                loss = criterion(outputs.float(), Y_train.float())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
                optimizer.step()
                running_loss += loss.item()
                running_correct = (pred.cpu() == Y_train.cpu()).sum()
                tr_acc += running_correct.item()
                total_num += len(batch[0])

            time_end = time.time()
            print(f"Model loss after {epoch+1} epochs = {criterion(outputs.float(), Y_train.float())}")
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
            cm = None # initialise confusion matrix
            # validation
            for batch in tqdm(testloader):
                X_train, Y_train = batch
                Y_train = Y_train.long()
                X_train, Y_train = X_train.to(device), Y_train.to(device)
                outputs = model(X_train)
                
                # switch prediction to multilabel
                pred = outputs>0.5

                running_correct = torch.all(pred.cpu() == Y_train.cpu(), dim=1).sum() # multilabel

                tr_acc += running_correct.item()
                total_num += len(batch[0])

                acc = tr_acc/total_num
                cm = get_multilabel_conf_matrix(Y_train.cpu(), pred.cpu(), cm)
            time_end = time.time()
            print('time cost', time_end - time_start, 's')
            print("\nAccuracy is", tr_acc/total_num)
            print("confusion matrix is\n", cm)
            if best < acc:
                best = acc
                best_cm = cm
            print("\nBest is", best)
        results[fold] = (best, best_cm)

    logfile = open("2p_noisy_cross_validation3.txt", "w")
    logfile.write(f"model_name = {args.model}, lr = {args.lr}, wd = lr = {args.wd} sample = {args.sample}, batch_size = {args.batch}, epochs = {args.epoch}\n")
    logfile.write(f"input_length = {args.input_length}, hlayers = {args.hlayers}, hheads = {args.hheads}, vlayers = {args.vlayers}, vheads = {args.vheads}\n")
    logfile.write(f"filter_size = {args.filter_size}, filter_size_v = {args.filter_size_v}, kernel_num = {args.kernel_num}, kernel_num_v = {args.kernel_num_v}\n")
    logfile.write(f"dropout_rate = {args.dropout_rate}, com_dim = {args.com_dim}, K = {args.K}, clip_gradient = {args.clip_gradient}\n")

    logfile.write("\n")
    logfile.write("\n")
    for key in results.keys():
        logfile.write(f"this is fold {key}\n")
        logfile.write(f"the accuracy is {results[key][0]}")
        logfile.write(f"\n Final confusion matrix:\n")
        cm = results[key][1]
        for i in range(len(cm)):
            logfile.write(f"\nActivity {i+1}:\n")
            for j in range(len(cm[i])):
                logfile.write(str(cm[i, j]) + "\n")
    logfile.close()


if __name__=="__main__":
    try:
        # c_main()
        cross_validation()
    except KeyboardInterrupt:
        print("error")


