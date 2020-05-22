# 1.1创建一个向量 1.2创建一个矩阵
import numpy as np

vector_row = np.array([1,2,3])

vetor_column = np.array([[1],
                        [2],
                        [3]])

matrix = np.array([[1,2],
                   [1,2],
                   [1,2]])

matrix_object = np.mat([[1,2],
                       [1,2],
                       [1,2]])

# matrix([[1,2],
#         [1,2],
#         [1,2]])

# 不推荐使用 矩阵数据结构。 1.数组才是Numpy的标准数据结构。
#                        2.绝大多数的NumPy 返回的是 数组而不是矩阵对象

# 1.3创建一个稀疏矩阵
import numpy as np
from scipy import sparse

matrix = np.array([[0,0],
                   [0,1],
                   [3,0]])

# 创建一个压缩的稀疏行 （Compressed Sparse Row, CSR)矩阵 只保存非零值
matrix_sparse = sparse.csr_matrix(matrix)
print(matrix_sparse)

# 1.4选择元素
import numpy as np
vector = np.array([1,2,3,4,5,6])
matrix = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])

vector[2]   #3 选择向量的第三个元素
matrix[1,1] #5 选择第二行第二列
# numpy 遵从0开始的索引编号

vector[:]   #array([1, 2, 3, 4, 5, 6])

vector[:3]  #array([1, 2, 3])

vector[3:]  #array([4, 5, 6])

vector[-1]  #6

matrix[:2,:]    #array([[1, 2, 3],
                # [4, 5, 6]])

matrix[:,1:2]   #array([[2],
                #[5],
                #[8]])

# 1.5展示一个矩阵的属性
import numpy as np
matrix = np.array([[1,2,3,4],
                   [5,6,7,8],
                   [9,10,11,12]])
matrix.shape    #查看行列数 （3,4)

matrix.size     #查看元素数量 12

matrix.ndim     #查看维数 2

# 1.6 对多个元素同时应用某个操作
import numpy as np

matrix = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])

add_100 = lambda i:i+100 # 创造公式
vectorized_add_100 = np.vectorize(add_100)  # 创建向量化函数
vectorized_add_100(matrix)      #应用函数
                                # array([[101, 102, 103],
                                #        [104, 105, 106],
                                #        [107, 108, 109]])

# 1.7找到最大值最小值   1.8 计算平均值，方差和标准差
import numpy as np
matrix = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])

np.max(matrix)      #9
np.min(matrix)      #1

#特定某一坐标值
np.max(matrix,axis=0)  #找到每一列最大的元素 array([7, 8, 9])
np.max(matrix,axis=1)  #找到每一行的最大的元素 array([3, 6, 9])

np.mean(matrix)     #平均值 5.0
np.var(matrix)      #方差 6.666666666667
np.std(matrix)      #标准差 2.581988897471611
# 特定某一坐标值 用法与max 相同 axis

# 1.9矩阵变形
import numpy as np
matrix = np.array([[1,2,3,4],
                   [5,6,7,8],
                   [9,10,11,12]])
matrix.reshape(2,6) #必须元素数量相同时 size
matrix.reshape(1,-1) #参数-1.表示可以根据需要补充元素。
matrix.reshape(12)  #如果给出一个整数做参数，会返回长度等于该整数的一维数组

# 1.10转置向量或矩阵 1.11 展开一个矩阵
import numpy as np
matrix = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])
matrix.T

# 向量严格意义上是不能被转置的，因为他是值的集合

# 转置一个 行向量
np.array([[1,2,3,4,5,6]]).T
# array([[1],
#        [2],
#        [3],
#        [4],
#        [5],
#        [6]]) 转变为一个列向量

matrix.flatten()
# array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 1.12计算矩阵的秩
import numpy as np

matrix = np.array([[1,1,1],
                   [1,1,10],
                   [1,1,15]])

np.linalg.matrix_rank(matrix) #2
# 秩 是由它的行或者列展开的向量空间的维度

# 1.13 计算行列式    1.14 获取矩阵对角线元素     1.15 计算矩阵的迹
import numpy as np
matrix = np.array([[1,2,3],
                   [2,4,6],
                   [3,8,9]])
np.linalg.det(matrix)       #0

matrix.diagonal() # array([1, 4, 9])
matrix.diagonal(offset=1) #array([2, 6]) 主对角线向上偏移量为1的对角线元素
matrix.diagonal(offset=-1) #array([2, 8]) 主对角线向下偏移量为1的对角线元素

matrix.trace()  #14 迹 为矩阵主对角线元素之和

# 1.16 计算特征值和特征向量
import numpy as np
matrix = np.array([[1,-1,3],
                   [1,1,6],
                   [3,8,9]])
eigenvalues, eigenvectors = np.linalg.eig(matrix)
eigenvalues #array([13.55075847,  0.74003145, -3.29078992])
eigenvectors
            # array([[-0.17622017, -0.96677403, -0.53373322],
            #        [-0.435951  ,  0.2053623 , -0.64324848],
            #        [-0.88254925,  0.15223105,  0.54896288]])

# 1.17计算点积
import numpy as np
vector_a = np.array([1,2,3])
vector_b = np.array([4,5,6])

np.dot(vector_a,vector_b)   #32
vector_a @ vector_b         #32

# 1.18矩阵的相加和相减
import numpy as np
matrix_a = np.array([[1,1,1],
                     [1,1,1],
                     [1,1,2]])
matrix_b = np.array([[1,3,1],
                     [1,3,1],
                     [1,3,8]])
np.add(matrix_a,matrix_b)
np.subtract(matrix_a,matrix_b)

matrix_a+matrix_b #也可简单用 + ，- 符号来表示

# 1.19矩阵乘法
import numpy as np
matrix_a = np.array([[1,1],
                     [1,2]])
matrix_b = np.array([[1,3],
                     [1,2]])
np.dot(matrix_a,matrix_b) #array([[2, 5],
                          #       [3, 7]])
matrix_a @ matrix_b
matrix_a * matrix_b       #对应元素相乘
                          # array([[1, 3],
                          #        [1, 4]])

# 1.20计算矩阵的逆
import numpy as np
matrix = np.array([[1,4],
                   [2,5]])

np.linalg.inv(matrix)
# array([[-1.66666667,  1.33333333],
#        [ 0.66666667, -0.33333333]])

matrix @ np.linalg.inv(matrix)
# array([[1., 0.],
#        [0., 1.]])

# 1.21生成随机数
import numpy as np
np.random.seed(0)
np.random.random(3) #生成三个0.0-1.0之间的随机浮点数
# array([0.5488135 , 0.71518937, 0.60276338])

np.random.randint(0,11,3) #生成三个0-10 之间的随机整数
# array([3, 7, 9])

np.random.normal(0.0,1.0,3) #从平均值为0，标准差为1的正态分布中抽取3个数
# array([-1.42232584,  1.52006949, -0.29139398])

np.random.logistic(0.0,1.0,3) #从平均值为0，且散布程度为1的logistic分布中抽取3个数
# array([-0.98118713, -0.08939902,  1.46416405])

np.random.uniform(1.0,2.0,3) # 从大于或等于1.0并且小于2.0范围中抽取3个数
# array([1.47997717, 1.3927848 , 1.83607876])