from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import torch
from config import *
from util import *
from collections import defaultdict

class ODDataset(Dataset):

    def __init__(self, od_data, flow_data, hist_len, num_grids, train_prop, val_prop, mode):

        od_shape = od_data.shape
        od_num_train = int(od_shape[0] * train_prop)
        od_num_val = int(od_shape[0] * val_prop)

        flow_shape = flow_data.shape
        flow_num_train = int(flow_shape[0] * train_prop)
        flow_num_val = int(flow_shape[0] * val_prop)

        if mode == 'train':
           self.od_data = od_data[:od_num_train]
           self.flow_data = flow_data[:flow_num_train]
        elif mode == 'validate':
            self.od_data = od_data[od_num_train:od_num_train+od_num_val]
            self.flow_data = flow_data[flow_num_train:flow_num_train+flow_num_val]
        elif mode == 'test':
            self.od_data = od_data[od_num_train+od_num_val:]
            self.flow_data = flow_data[flow_num_train+flow_num_val:]

        self.hist_len = hist_len
        self.num_grids = num_grids

    def __len__(self):
        return self.od_data.shape[0]//(self.num_grids*self.hist_len)

    def __getitem__(self, index):
        x_od = self.od_data[index: index + (self.hist_len*self.num_grids)]
        x_od = x_od.reshape(-1,self.num_grids, self.num_grids)
        y_od = self.od_data[index + (self.hist_len*self.num_grids): index + (self.hist_len*self.num_grids) + self.num_grids]
        y_od = y_od.reshape(-1,self.num_grids,self.num_grids)

        x_flow = self.flow_data[index:index+self.hist_len]
        y_flow = self.flow_data[index+self.hist_len:index+self.hist_len+1]

        od_input = torch.tensor(x_od, dtype=torch.float)
        od_label = torch.tensor(y_od, dtype=torch.float)
        flow_input = torch.tensor(x_flow,dtype=torch.float)
        flow_label = torch.tensor(y_flow,dtype=torch.float)

        return {
            'od_input':od_input,
            'od_label':od_label,
            'flow_input':flow_input,
            'flow_label':flow_label
        }

def get_dataloader(hist_len, num_grids, batch_size, train_prop, val_prop):
    def create_loader(od_data, flow_data, hist_len, num_grids, batch_size, train_prop, val_prop, mode, shuffle=False):
        dataset = ODDataset(od_data, flow_data, hist_len, num_grids, train_prop, val_prop, mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    #1.构建数据集
    #(190464,256)
    # od_data = np.random.randn(190464,256)
    # flow_data = np.random.randn(744,256,2)

    od_data = pd.read_csv(od_input_file, sep=',', header=None)
    od_data = od_data.values

    #(744,256,2)
    flow_data = np.load(flow_input_file)['data']

    # 2.归一化
    od_train_data = od_data[:int(od_data.shape[0]*train_prop)]
    od_scaler = MinMaxScalar(np.min(od_train_data), np.max(od_train_data))
    flow_train_data = flow_data[:int(flow_data.shape[0]*train_prop)]
    flow_scaler = MinMaxScalar(np.min(flow_train_data),np.max(flow_train_data))

    od_data = od_scaler.transform(od_data)
    flow_data = flow_scaler.transform(flow_data)

    #3.构建dataloader
    train_loader = create_loader(od_data, flow_data, hist_len, num_grids, batch_size, train_prop, val_prop, mode='train', shuffle = True)
    val_loader = create_loader(od_data, flow_data, hist_len, num_grids, batch_size, train_prop, val_prop, mode='validate', shuffle = False)
    test_loader = create_loader(od_data, flow_data, hist_len, num_grids, batch_size, train_prop, val_prop, mode='test', shuffle = False)

    return {
        'train' : train_loader,
        'validate' : val_loader,
        'test' : test_loader,
        'scaler' : od_scaler
    }

def convert_geo_adj(row_num, col_num):

    m_size = row_num * col_num

    adj_ = np.zeros((m_size, m_size))

    geo_neighbors = defaultdict(set)

    # Formulate the neighbor set for each grid
    for i in range(0, m_size):

        grid_no = i
        gn_grids = []
        # gn_grid2dis = {}
        if i == 0:
            gn_grids = [i + 1, i + col_num, i + col_num + 1]
            # print("0::::[i + 1, i + col_num, i + col_num + 1]"

        elif i == col_num - 1:
            gn_grids = [i - 1, i + col_num - 1, i + col_num]
            # print("2::::[i - 1, i + col_num - 1, i + col_num]"

        elif i == row_num * col_num - col_num:
            gn_grids = [i - col_num, i - col_num + 1, i + 1]
            # print("6::::[i - col_num, i - col_num + 1, i + 1]"

        elif i == row_num * col_num - 1:
            gn_grids = [i - col_num - 1, i - col_num, i - 1]
            # print("8::::[i - col_num - 1, i - col_num, i - 1]"

        elif i in range(1, col_num - 1):
            gn_grids = [i - 1, i + 1, i + col_num - 1, i + col_num, i + col_num + 1]
            # print("1::::[i - 1, i + 1, i + col_num - 1, i + col_num, i + col_num + 1]"

        elif i in range(row_num * col_num - col_num + 1, row_num * col_num - 1):
            gn_grids = [i - col_num - 1, i - col_num, i - col_num + 1, i - 1, i + 1]
            # print("7::::[i - col_num - 1, i - col_num, i - col_num + 1, i - 1, i + 1]"

        elif i in range(0 + col_num, row_num * col_num - col_num, col_num):
            gn_grids = [i - col_num, i - col_num + 1, i + 1, i + col_num, i + col_num + 1]
            # print("3::::[i - col_num, i - col_num + 1, i + 1, i + col_num, i + col_num + 1]"

        elif i in range(col_num - 1 + col_num, row_num * col_num - 1, col_num):
            gn_grids = [i - col_num - 1, i - col_num, i - 1, i + col_num - 1, i + col_num]
            # print("5::::[i - col_num - 1, i - col_num, i - 1, i + col_num - 1, i + col_num]"

        else:
            gn_grids = [i - col_num - 1, i - col_num, i - col_num + 1, i - 1, i + 1, i + col_num - 1, i + col_num,
                        i + col_num + 1]
            # print("4::::[i - col_num - 1, i - col_num, i - col_num + 1, i - 1, i + 1, i + col_num - 1, i + col_num, i + col_num + 1]"

        # print("gn_grids=" + "\n", gn_grids
        for k in range(len(gn_grids)):

            adj_[grid_no][gn_grids[k]] = 1
            adj_[gn_grids[k]][grid_no] = 1
            geo_neighbors[grid_no].add(gn_grids[k])

    return geo_neighbors,adj_

def load_node_embedding():
    with open(embedding_file,'r') as fr:
        lines = fr.readlines()[1:]
        df = pd.DataFrame(list(map(lambda line: line.strip().split(' '),lines)))
        df.columns = ['node'] + ['emb_%d' % i for i in range(10)]
        df[['emb_%d' % i for i in range(10)]] = df[['emb_%d' % i for i in range(10)]].astype('float')
        df['node'] = df['node'].astype('int')

    df = df.sort_values(by='node',ascending=True)
    node_embedding = df.drop('node',axis=1).values
    return node_embedding


if __name__ == '__main__':
    hist_len = 8
    num_grids = 256
    batch_size = 32
    train_prop = 0.8
    val_prop = 0.1
    skip = False

    load_node_embedding()

    #test dataloader
    # dataloaders = get_dataloader(hist_len, num_grids, batch_size, train_prop, val_prop,skip)
    #
    #
    # print(len(dataloaders['train']))
    # for input,label in dataloaders['train']:
    #     print(input.shape,label.shape)
    #
    # print(len(dataloaders['validate']))
    # for input,label in dataloaders['validate']:
    #     print(input.shape,label.shape)
    #
    # print(len(dataloaders['test']))
    # for input,label in dataloaders['test']:
    #     print(input.shape,label.shape)

    '''
    train_prop = 0.8
    val_prop = 0.1

    data = np.random.rand(190464,256)

    num_samples = data.shape[0]
    num_train = int(num_samples * train_prop)
    num_val = int(num_samples * val_prop)
    num_test = num_samples - num_train - num_val

    
    train_dataset = ODDataset(data, split_point_start=0, split_point_end=num_train, hist_len=8, num_grids=256)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    print('train_length sample',len(train_dataloader))

    for input, label in train_dataloader:
        print('train:',input.shape,label.shape)

    val_dataset = ODDataset(data, split_point_start=num_train, split_point_end=num_train+num_val, hist_len=8, num_grids=256)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print('val_length sample',len(val_dataloader))

    for input, label in val_dataloader:
        print('val:',input.shape,label.shape)


    test_dataset = ODDataset(data, split_point_start=num_train+num_val, split_point_end=num_samples, hist_len=8, num_grids=256)
    val_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print('test_length sample',len(test_dataset))

    for input, label in val_dataloader:
        print('test:',input.shape,label.shape)
    '''

