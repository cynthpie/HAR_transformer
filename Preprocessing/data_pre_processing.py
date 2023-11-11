import csv
import os
import torch
import torch.utils.data as Data
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


def average_list(d_list):
    sum = [0.0 for _ in range(len(d_list[0]))]
    for j in range(len(d_list[0])):
        for i in range(len(d_list)):
            sum[j] += d_list[i][j]
        sum[j] /= len(d_list)
    return sum


def merge_timestamp(data, time_stamp):
    intervel = (time_stamp[len(time_stamp)-1] - time_stamp[0]) / 2000
    cur_range = time_stamp[0] + intervel
    temp_list = []
    new_data = []
    for i in range(len(time_stamp)):
        if time_stamp[i] > cur_range:
            if len(temp_list) != 0:
                new_data.append(average_list(temp_list))
            else:
                new_data.append(data[i])
            temp_list = []
            cur_range = cur_range + intervel
        temp_list.append(data[i])
    if len(temp_list) != 0:
        new_data.append(average_list(temp_list))
    if len(new_data) < 2000:
        new_data.append(data[len(time_stamp)-1])
        print("!!!!")
    return new_data[:2000]


def load_annotation(root):
    root = root + "/"
    file_list = os.listdir(root)

    # handle normal filename
    annotation_list = [file_name for file_name in file_list if ('annotation' in file_name) and ('sankalp' not in file_name) and ('siamak' not in file_name)]
    annotation_list.sort(key=lambda annotation: str(annotation[10:]))

    # handle filename contain sankalp or siamak
    annotation_list2 = [file_name for file_name in file_list if ('annotation' in file_name) and (('sankalp' in file_name) or ('siamak' in file_name))]
    annotation_list2.sort(key=lambda annotation: str(annotation[10:]))

    # concatinate two list
    annotation_list = annotation_list + annotation_list2
    print("number of annotation file is", len(annotation_list))
    print(annotation_list[501:510])
    label = []
    
    aclist = ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']
    for file in tqdm(annotation_list):
        for j in range(len(aclist)):
            if file.find(aclist[j]) != -1:
                label.append(j)
                break
    label = torch.tensor(label)
    print(label)
    return label


def load_input(root):
    root = root + "/"
    file_list = os.listdir(root)

    # handle normal filename
    input_list = [file_name for file_name in file_list if ('input' in file_name) and ('sankalp' not in file_name) and ('siamak' not in file_name)]
    input_list.sort(key=lambda input: str(input[5:]))

    # handle filename contain sankalp or siamak
    input_list2 = [file_name for file_name in file_list if ('input' in file_name) and (('sankalp' in file_name) or ('siamak' in file_name))]
    input_list2.sort(key=lambda input: str(input[13:]))

    # concatinate two list
    input_list = input_list + input_list2
    print("number of input file is", len(input_list))
    print(input_list[501:510])

    data = []
    for file in tqdm(input_list):
            with open(root + file, encoding='utf-8') as f:
                #print(root + file)
                reader = csv.reader(f)
                record = []
                time_stamp = []
                for r in reader:
                    record.append([float(str_d) for str_d in r[1:91]])
                    time_stamp.append(float(r[0]))
                record = merge_timestamp(record, time_stamp)
                float_data = torch.tensor(record, dtype=torch.float32, requires_grad=False)
                # print(float_data.shape) 2000, 90
                data.append(float_data.unsqueeze(0))
    data = torch.cat(data, dim=0).reshape(2000, 90)
    return data


def load_data(root):
    label = load_annotation(root)
    data = load_input(root)
    data = torch.cat(data, dim=0)
    print((data[0].shape), (data[1].shape))
    print(label.shape)
    data = Data.TensorDataset(data, label)
    return data


if __name__ == "__main__":
    batch_size = 16
    data = torch.load("Data.pt")
    print(data)
    loader = DataLoader(data, batch_size, shuffle=False)
    for batch_inx, (data, y) in enumerate(loader):
        print("this is batch",batch_inx )
        print(data.shape) # torch.size([1, 90])
        print(y.shape) # torch.size([1])
