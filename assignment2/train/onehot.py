from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 创建 OneHotEncoder 对象
encoder = OneHotEncoder()

# 准备数据
data = np.array([['cat', 'small'],
                 ['dog', 'medium'],
                 ['fish', 'large'],
                 ['dog', 'small']])

# 拟合编码器并指定要编码的列
encoder.fit(data[:, [0, 1]])

# 转换数据为 One-Hot 编码
encoded_data = encoder.transform(data[:, [0, 1]])

# 转换为密集矩阵
dense_encoded_data = encoded_data.toarray()

print(dense_encoded_data)
