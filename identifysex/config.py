import os,torch

from pro.allyears.basicfunc import *
from pro.deeplearn.trainmodel.utils.vocab import *
from pro.deeplearn.trainmodel.utils.ttspl import *


class Config(object):

    def __init__(self):
        self.run = True
        self.device = torch.device('cuda:0')
        """
        模型参数
        """
        self.num_classes = 2                        # 分类数，默认1是回归问题，非1是分类问题
        self.embed = 128                        # shape的最内部数据的长度
        self.normal = True                       #  是否批量归一化
        self.hidden_size = 32                    #  LSTM单元隐藏层个数
        self.num_layers = 1                    #  LSTM层个数
        self.dropout = 0.2                      # 随机丢弃
        if self.run:
            self.dropout = 0
        self.batch_size = 256 if self.run == False else 1  # 批数据大小，如果显存溢出则调小，原则为2**n
        self.mapping = True                   #  是否做映射，加relu函数，只用作model_num==4的时候
