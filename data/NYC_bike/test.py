import numpy as np

# 创建一个简单的二维数组
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 沿着第一个轴（axis=0，行）向下滚动1个元素
b = np.roll(a, shift=1, axis=0)
print(b)
