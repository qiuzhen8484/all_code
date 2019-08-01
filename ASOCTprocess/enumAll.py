
from enum import Enum

class TrainMode(Enum):
    train = 0
    test = 1
    segment = 2
    trainfc = 3
    testfc = 4

# 数据模式类别
class DataMode(Enum):
    DATA_8_MODE = 0
    DATA_16_LBO = 1
    DATA_16_LGS = 2
    DATA_16_LRS = 3
