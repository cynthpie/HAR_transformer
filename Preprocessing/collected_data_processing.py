import numpy as np
import pickle
import scipy.io as scio

import csv
import os
import torch
import torch.utils.data as Data
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy import signal
import sys

input_length = 350
input_dim = 270


with open("/home/joanna/collected_data_preprocessing/noisy_data_1.pk1", "rb") as file:
    data = pickle.load(file)
x_full = data[0]
y_full = data[1]
print(len(data[0]), len(data[1]))
print(x_full[0])
print(y_full[0])

x_full_resampled = []
for i in range(len(x_full)):
    x_original = x_full[i]
    x_original = np.reshape(x_original, (-1, input_dim))
    x_full_resampled.append(x_original)

print(len(x_full_resampled))
print(len(y_full))

# filter out data length <input_length
for i in range(len(x_full_resampled)):
    if len(x_full_resampled[i])<input_length:
        print(i, len(x_full_resampled[i]))

x_full_resampled = [x for x in x_full_resampled if len(x)>input_length]
# y_full.pop(303) 
# y_full.pop(304) 
# y_full.pop(583)
print(len(x_full_resampled), len(y_full))


def average_list(d_list):
    sum = [0.0 for _ in range(len(d_list[0]))]
    for j in range(len(d_list[0])):
        for i in range(len(d_list)):
            sum[j] += d_list[i][j]
        sum[j] /= len(d_list)
    return sum


def merge_timestamp(data, time_stamp):
    intervel = (time_stamp[len(time_stamp)-1] - time_stamp[0]) / input_length
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
    if len(new_data) < input_length:
        new_data.append(data[len(time_stamp)-1])
        print("!!!!")
    return new_data[:input_length]


def load_annotation(y_full):
    label = []
    aclist = ["jump", "run", "sit", "stand", "walk"] 
    for l in y_full:
        for j in range(len(aclist)):
            if l[j] == 1:
                label.append(j)
                break
    label = torch.tensor(label)
    print(label)
    return label


def load_input(x_full_resampled):
    data = []
    for samples in x_full_resampled:
        record = []
        time_stamp = []
        print(samples.shape)
        for row in samples:
            record.append(row)
        time_stamp = list(range(samples.shape[0]))
        record = merge_timestamp(record, time_stamp)
        float_data = torch.tensor(record, dtype=torch.float32, requires_grad=False)
        print(float_data.shape)
        data.append(float_data.unsqueeze(0))
    return data


def load_data(y_full, x_full_resampled):
    label = load_annotation(y_full)
    data = load_input(x_full_resampled)
    data = torch.cat(data, dim=0)
    print(data.size())
    print((data[0].shape), (data[1].shape))
    print(label.shape)
    data = Data.TensorDataset(data, label)
    print(data)
    torch.save(data, "Data_noisy_1p_w350.pt")
    return data
    

load_data(y_full, x_full_resampled)