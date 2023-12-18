import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
# from cnn_stock import Net

import zipfile
import csv


image_width = {5:15, 10:30, 20:60, 60:180}
image_height = {5:32, 10:36, 20:64, 60:96}
price_height = {5:25, 10:28, 20:51, 60:76}
volume_height = {5:6, 10:7, 20:12, 60:19}


class DrawOHLC:
    """
    given a dataset with arbitrary asked size(5/20/60),
    draw four kinds of OHLC
    """
    def __init__(self, sample_size:int, predict_period, df):
        """
        :param
            df: columns=['Open', 'High', 'Low', 'Close' ...]; index: datetime
        """
        self.sample_size = sample_size
        self.predict_period = predict_period
        self.df = df

        self.image_width = image_width[self.sample_size]
        self.image_height = image_height[self.sample_size]
        self.price_height = price_height[self.sample_size]
        self.volume_height = volume_height[self.sample_size]

        self.fig = np.zeros((self.image_height, self.image_width))
        # index_values = [i for i in range(self.image_height-1,-1,-1)]
        # self.fig = np.flipud(self.fig)

        self.max_price = np.max(self.df["High"])
        self.min_price = np.min(self.df["Low"])
        self.max_ma = np.max(self.df[f"MA{self.sample_size}"])
        self.min_ma = np.min(self.df[f"MA{self.sample_size}"])
        self.max_vb = np.max(self.df["Volume"])
        self.min_vb = np.min(self.df["Volume"])

        self.ret = self.df[f"Ret_{self.predict_period}d"][-1]
        if self.ret>0:
            self.label = 1
        else:
            self.label = 0
    
    def _price(self, wo_vb=True):
        # 获取open， high，low， close四列
        to_be_drawn = pd.concat([
            self.df.loc[:,"Open"], self.df.loc[:,"Close"],
            self.df.loc[:, "High"], self.df.loc[:,"Low"]
        ], axis=1)
        # 将价格数值缩放为图片大小 对应索引
        if self.max_price == self.min_price:
            self.min_price = self.max_price - 0.0001
        if wo_vb:
            scale = (self.image_height - 1) / (self.max_price - self.min_price)
        else:
            scale = (self.price_height - 1) / (self.max_price - self.min_price)
        to_be_drawn = to_be_drawn.sub(self.min_price).mul(scale)
        # 四舍五入价格数值 方便后面画图
        to_be_drawn = to_be_drawn.round(0)
        # 逐列（天）画图
        count = 0
        for index, row in to_be_drawn.iterrows():
            open_price, high_price, low_price, close_price = row["Open"], row["High"], row["Low"], row["Close"]
            # print(index,open_price, high_price, low_price, close_price)
            if not np.isnan(open_price):
                self.fig[int(open_price), count] = 255
                # self.fig[open_price-1, count] = 200
            if not np.isnan(low_price) and not np.isnan(high_price):
                self.fig[int(low_price):int(high_price) + 1, count + 1] = 255
            if not np.isnan(close_price):
                self.fig[int(close_price), count + 2] = 255
            count += 3

    def _ma(self, wo_vb=True):
        to_be_drawn = self.df[[f"MA{self.sample_size}"]]
        if self.max_ma == self.min_ma:
            self.min_ma = self.max_ma - 0.0001
        if wo_vb:
            scale = (self.image_height - 1) / (self.max_ma - self.min_ma)
        else:
            scale = (self.price_height - 1) / (self.max_ma - self.min_ma)
        to_be_drawn = to_be_drawn.sub(self.min_ma).mul(scale/3)
        to_be_drawn = to_be_drawn.round(0)
        # # replace nan as mean
        # mean = np.nanmean(to_be_drawn)
        # to_be_drawn[np.isnan(to_be_drawn)] = mean

        count = 1
        mas = []
        for index, row in to_be_drawn.iterrows():
            ma = row[f"MA{self.sample_size}"]
            if np.isnan(ma):
                count += 3
                continue
            ma = int(ma)
            mas.append(ma)
            self.fig[ma, count] = 255
            if count>3 and len(mas)>1:
                last_ma = mas[-2]
                gap = round((ma-last_ma)/3, 0)
                ma_1 = int(last_ma + gap)
                ma_2 = int(ma_1 + gap)
                self.fig[ma_1, count-2] = 255
                self.fig[ma_2, count-1] = 255
            count += 3

    def _vb(self):
        to_be_drawn = self.df[["Volume"]]
        if self.max_vb == self.min_vb:
            self.min_vb = self.max_vb - 0.0001
        scale = (self.volume_height - 1)/(self.max_vb - self.min_vb)
        to_be_drawn = to_be_drawn.sub(self.min_vb).mul(scale)
        to_be_drawn = to_be_drawn.round(0)
        count = 1
        for index, row in to_be_drawn.iterrows():
            if not np.isnan(row["Volume"]):
                vb = int(row["Volume"])
                self.fig[:vb+1, count] = 255
            count += 3

    def wo_vb_wo_ma(self):
        self._price()
        self.fig = np.flipud(self.fig)
        image = self.fig
        # plt.imshow(self.fig, cmap="gray")
        # plt.axis("off")
        # plt.savefig(f"./A/wo_vb_wo_ma/5days/{figname}.pdf")
        # plt.clf()
        self.fig = np.zeros((self.image_height, self.image_width))
        return image, self.label, self.ret

    def wo_vb_w_ma(self):
        self._price()
        self._ma()
        self.fig = np.flipud(self.fig)
        image = self.fig
        # mas = self._ma()
        # plt.plot([ma[0] for ma in mas], [self.image_height-ma[1] for ma in mas], color="white")
        # plt.imshow(self.fig, cmap="gray")
        # plt.axis("off")
        # plt.savefig(fig_name)
        # plt.clf()
        self.fig = np.zeros((self.image_height, self.image_width))
        return image, self.label, self.ret

    def w_vb_wo_ma(self):
        self._vb()
        self.fig = np.flipud(self.fig)
        self._price(False)
        self.fig[:self.price_height, :] = np.flipud(self.fig[:self.price_height, :])
        image = self.fig
        # plt.imshow(self.fig, cmap="gray")
        # plt.axis("off")
        # plt.savefig(fig_name)
        # plt.clf()
        self.fig = np.zeros((self.image_height, self.image_width))
        return image, self.label, self.ret

    def w_vb_w_ma(self):
        self._vb()
        self.fig = np.flipud(self.fig)
        self._price(False)
        self._ma()
        self.fig[:self.price_height, :] = np.flipud(self.fig[:self.price_height, :])
        image = self.fig
        # mas = self._ma()
        # plt.plot([ma[0] for ma in mas], [self.price_height - ma[1] for ma in mas],color="white")
        # plt.imshow(self.fig, cmap="gray")
        # plt.axis("off")
        # plt.savefig(fig_name)
        # plt.clf()
        self.fig = np.zeros((self.image_height, self.image_width))
        return image, self.label, self.ret


# total_folder = "./"


# draw all IXRX pictures and return images and labels of one excel(one company)
def draw(file, window_size:int, predict_period:int, image_type:str, train=True):
    """
    :param file_name: stock information to be read
    :param window_size: 5/20/60
    :param predict_period: 5/20/60
    :param image_type: wo_vb_wo_ma/wo_vb_w_ma/w_vb_wo_ma/w_vb_w_ma
    :return: image(nparray), label(0/1)
    """
    # read the whole file and set date as index
    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df.set_index("Date", inplace=True)

    # os.makedirs(os.path.join(total_folder, f"{image_type}/{window_size}_{predict_period}"), exist_ok=True)
    labels = []
    images = []
    rets = []

    # sliding windows and draw OHLC through the file
    if train is True:
        df = df[df["year"]<2018]
        if len(df)>window_size:
            for i in range(0, len(df)+1-window_size): # 训练集是否按滑动窗口取，手动调整(0, len(df)+1-window_size, window_size)
                window_df = df.iloc[i:i+window_size, :]
                demo = DrawOHLC(window_size, predict_period, window_df)
                image, label, ret = demo.wo_vb_wo_ma() # 手动调整
                labels.append(label)
                images.append(image)
                rets.append(ret)
        else:
            return False, False, False
    else:
        df = df[df["year"]>=2018]
        for i in range(0, len(df)+1-window_size,window_size):
            window_df = df.iloc[i:i+window_size, :]
            demo = DrawOHLC(window_size, predict_period, window_df)
            image, label, ret = demo.wo_vb_wo_ma() # 手动调整
            labels.append(label)
            images.append(image)
            rets.append(ret)

    return images, labels, rets


# generate one type of model's images and labels of all companies
def get_all_train_drawn(folder_zip_path:str, window_size:int, predict_period:int, image_type:str):
    with zipfile.ZipFile(folder_zip_path, "r") as zip_file:
        images_all = []
        labels_all = []
        for filename in tqdm(zip_file.namelist()[1:]):  # 手动检查
            with zip_file.open(filename) as file:            
                images, labels,_ = draw(file,window_size,predict_period,image_type)
                if images is False:
                    continue
                images_all.append(images)
                labels_all.append(labels)
    
    return images_all, labels_all
