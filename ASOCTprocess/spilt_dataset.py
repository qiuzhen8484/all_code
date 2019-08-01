# coding: utf-8
"""
    划分训练、测试集
"""
import pickle as pkl
from utils import *

def GetTrainDict(data_path, save_path, split_idx=0.7):
    train_dict = {}
    with open(os.path.join(save_path, 'train_dict_C.pkl'), 'wb+') as f:
        train_list, val_list = split_dataset(data_path, split_idx=split_idx)
        train_dict['train_list'] = train_list
        train_dict['val_list'] = val_list
        # train_dict['test_list'] = train_list + val_list
        pkl.dump(train_dict, f)
    print('Step 2: success split the dataset, train:%s, val:%s' % (len(train_list), len(val_list)))

if __name__ == '__main__':
    data_path = '/data/qiuzhen/cataract_classifi/Origin_LGS_C'    # 数据路径
    save_path = '/home/intern1/qiuzhen/Works/test/sick_classifi/data'   # 保存train_dict的路径
    split_idx = 0.7      # 训练集所占比例，剩余则为测试集
    GetTrainDict(data_path, save_path)
