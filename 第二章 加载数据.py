# 2.1 加载样本数据集
from sklearn import datasets
digits = datasets.load_digits() #加载手写数字数据集
features = digits.data #创建特征矩阵
target = digits.target #创建目标向量
features[0]
# 比较流行的数据集（toys）load_boston(波士顿房价), load_iris(鸢尾花尺寸),load_digits(手写数字图片）

# 2.2 创建仿真数据集
# 仿真数据集做线性回归
from sklearn.datasets import make_regression
features,target,coefficients = make_regression(n_samples = 100, n_features= 3, n_informative=3,
                                               n_targets=1,noise=0.0,coef=True,random_state=1)
print('Feature Matrix\n',features[:3])  #特征矩阵
print('Target Vector\n',target[:3])     #目标向量

# 仿真数据集做分类
from sklearn.datasets import make_classification
features,target = make_classification(n_samples=100,n_features=3,n_informative=3,n_redundant=0,
                                      n_classes=2,weights=[.25,.75],random_state=1)
print('Feature Matrix\n',features[:3])  #特征矩阵
print('Target Vector\n',target[:3])     #目标向量

# 仿真数据做聚类分析
from sklearn.datasets import make_blobs
features,target = make_blobs(n_samples=100,n_features=2,centers=3,
                             cluster_std=0.5,shuffle=True,random_state=1)
print('Feature Matrix\n',features[:3])  #特征矩阵
print('Target Vector\n',target[:3])     #目标向量

# 在 make_regression 和 make_classification中，n_informative 确定了用于生成目标向量的特征的数量,
# n_informative 的值比总的特征数(n_features)小，则生成的数据集将包含多余的特征，这些特征可以通过特征选择技术识别出来

# make_classification 包含了一个weights 参数，可以利用它生成不均衡的仿真数据集。例如weights=[.25,.75],那么生成
# 的数据集中，25%的观察值属于第一个分类，75%的观察值属于第二个分类

# make_blobs中，centers 的参数决定了要生成多少个聚类，可以用matplotlib 将，make_blobs 生成的聚类显示出来

import matplotlib.pyplot as plt
plt.scatter(features[:,0],features[:,1],c=target)
plt.show()


# 2.3加载CSV 文件   2.4加载Excel 文件    2.5加载Json 文件
import pandas as pd
url = 'https://tinyurl.com/simulated_data'
dataframe= pd.read_csv(url)
dataframe.head(2)

url = 'https://tinyurl.com/simulated_excel'
dataframe=pd.read_excel(url,sheetname=0,header=1)
dataframe.head(2)

url= 'https://tinyurl.com/simulated_json'
dataframe=pd.read_json(url,orient='columns')
dataframe.head(2)

# 2.6查询SQL数据库
import pandas as pd
from sqlalchemy import create_engine

database_connection =create_engine('sqlite:///sample.db')
dataframe=pd.read_sql_query('SELECT * FROM data',database_connection)
dataframe.head(2)