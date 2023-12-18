import os
import sys

import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
from torchsummary import summary

from generate_pic_label import get_all_train_drawn
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(23040, 2),
        )
        self.softmax = nn.Softmax(dim=1)
       
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(-1,23040)
        x = self.fc1(x)
        x = self.softmax(x)
        return x


"""
saved images&labels:
images_5_5_wo_vb_wo_ma.npy
labels_5_5_wo_vb_wo_ma.npy
images_20_5_wo_vb_wo_ma.npy
labels_20_5_wo_vb_wo_ma.npy
images_jump_5_5_wo_vb_wo_ma.npy
labels_jump_5_5_wo_vb_wo_ma.npy
images_20_5_w_vb_w_ma.npy
labels_20_5_w_vb_w_ma.npy
images_10_5_wo_vb_wo_ma.npy
labels_10_5_wo_vb_wo_ma.npy
"""

# # create train data
images, labels = get_all_train_drawn("/home/ttzhang/00-code/00-deep-learing/project/to_csv_ret3.zip",10,5, "wo_vb_wo_ma") # 手动调整
images = np.concatenate(images, axis=0)
labels = np.concatenate(labels, axis=0)
print(images.shape)
print(labels.shape)
np.save("images_10_5_wo_vb_wo_ma.npy", images) # 手动调整
np.save("labels_10_5_wo_vb_wo_ma.npy", labels) # 手动调整

# load train data
images = np.load("images_10_5_wo_vb_wo_ma.npy") # 手动调整
labels = np.load("labels_10_5_wo_vb_wo_ma.npy") # 手动调整
print(images.shape, labels.shape)
images = np.expand_dims(images, axis=1)


# build dataset
class MyDataset(Dataset):
    
    def __init__(self, img, label):
        self.img = torch.Tensor(img.copy())
        self.label = torch.Tensor(label)
        self.len = len(img)
  
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]


# split data and get dataloader
train_val_ratio = 0.7
split_idx = int(images.shape[0] * 0.7)
train_dataset = MyDataset(images[:split_idx], labels[:split_idx])
val_dataset = MyDataset(images[split_idx:], labels[split_idx:])

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, pin_memory=True)

# 设置超参数
learning_rate = 0.00001
num_epochs = 100

# 初始化模型,损失函数和优化器
model = Net().to(device)

try:
    model.load_state_dict(torch.load("baseline_epoch_5_train_0.704243_val_0.695599.pt", map_location=device))
except:
    pass

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)


# 训练模型： 注意别过拟合 如果准确率一直下降就停掉
for epoch in range(num_epochs):
    for images, labels in tqdm(train_dataloader):
        # 将数据转换为模型所需的类型
        images = images.float().to(device)
        labels = labels.long().to(device)
        # labels_one_hot = nn.functional.one_hot(labels.squeeze(),num_classes=2).long().to(device)

        # 前向传递和反向传递
        optimizer.zero_grad() # 将优化器缓存梯度清零
        output = model(images)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step() # 更新模型权重参数

    # 每个epoch结束后在测试集上进行评估
    correct = 0
    total = 0
    threshold = 0.8
    with torch.no_grad(): # 语句块内的计算不会被PyTorch自动跟踪梯度，从而提高计算效率；
        for images, labels in val_dataloader:
            # 将数据转换为模型所需的类型
            images = images.float().to(device)
            labels = labels.long().to(device)

            # 前向传递并计算准确率
            output = model(images)
            # wrong = torch.count_nonzero(output-labels).item()
            prob, predicted = torch.max(output.data, 1)   
            # print(prob,prob.shape,predicted.shape)    
            mask = prob>threshold
            new_prob = torch.masked_select(prob, mask)
            if len(new_prob)>0:
                total += len(new_prob)
                # print(predicted.size(), labels.size())
                correct += (predicted[mask] == labels[mask]).sum().item()
        if total>0:
            print('Epoch [{}/{}], Loss: {:.4f}, Test Accuracy: {:.2f}%'
                .format(epoch+1, num_epochs, loss.item(), 100 * correct / total))
    torch.save(model.state_dict(), 'cnn_stock.pt')



# 得到每个股票每五天特定日期的看涨概率的表格
from generate_pic_label import DrawOHLC
import zipfile

model.load_state_dict(torch.load('cnn_stock.pt'))

datedate = pd.read_excel("datedate.xlsx")
adjust_date = []
for index, row in datedate.iterrows():
    date = str(row["date"])
    adjust_date.append(date[:10])


def get_bullish_prob(window_df, window_size, predict_period):
    demo = DrawOHLC(window_size, predict_period, window_df)
    image,_,ret = demo.wo_vb_wo_ma() # 手动调整

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=1)
    image = torch.Tensor(image.copy())
    image = image.float().to(device)

    output = model(image)
    prob = output.tolist()
    return prob[0][1], ret


def get_test_df(folder_zip_path:str, window_size:int, predict_period:int):
    dataframe = pd.DataFrame(columns=['Stock Name', 'AdjDate', 'Bullish Probability', "Ret"])
    with zipfile.ZipFile(folder_zip_path, "r") as zip_file:
        for filename in tqdm(zip_file.namelist()[1:]): #手动check
            with zip_file.open(filename) as file:
                # 拿到一张Excel里2018年以后的部分
                df = pd.read_csv(file)
                df["Date"] = pd.to_datetime(df["Date"])
                df["year"] = df["Date"].dt.year
                df.set_index("Date", inplace=True)
                df = df[df["year"]>=2018]
                
                # 五天五天生成图片计算概率
                count = -1
                for index, row in df.iterrows():
                    count += 1
                    date = str(row["Time"])
                    if date in adjust_date:
                        if count-9<0:  # 手动调整
                            continue
                        window_df = df.iloc[count-9:count+1, :] # 手动调整
                        bullish_prob, ret = get_bullish_prob(window_df, window_size, predict_period)

                        stockname = filename[12:]
                        stockname = stockname[:-4]
                        dataframe.loc[len(dataframe.index)] = stockname, date, bullish_prob, ret
    return dataframe


bullish_prob_df = get_test_df("/home/ttzhang/00-code/00-deep-learing/project/to_csv_ret3.zip", 10, 5) # 手动调整
print(bullish_prob_df)
bullish_prob_df.to_excel("bull_I10R5.xlsx") # 手动调整

