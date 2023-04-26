import numpy as np

# 创建一个具有指定形状的随机数组
data = np.random.rand(300, 535, 535)

# 对数据进行最大最小归一化
data_min = np.min(data)
data_max = np.max(data)

normalized_data = (data - data_min) / (data_max - data_min)
origin = normalized_data * (data_max - data_min) + data_min
print("原始数据的最小值和最大值:", data_min, data_max)
print("归一化后的数据的最小值和最大值:", np.min(normalized_data), np.max(normalized_data))
