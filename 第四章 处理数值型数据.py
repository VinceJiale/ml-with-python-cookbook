# 4.1 特征的缩放
import numpy as np
from sklearn import preprocessing

feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])

minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))  #创建缩放器
scaled_feature = minmax_scale.fit_transform(feature)    #缩放特征的值
scaled_feature

# 机器学习中，缩放是一个很常见的预处理任务。本书后面所讲的很多算法都假设所有的特征是在同一取值范围中的，最常见的范围是[0,1],[-1,1]。用于
# 缩放的方法有很多，，最简单的是 min-max 缩放。 公式: Xi' = (Xi-min(X))/(max(X)-min(X))

# 4.2 特征的标准化
import numpy as np
from sklearn import preprocessing

x = np.array([[-1000.1],
              [-200.2],
              [500.5],
              [600.6],
              [9000.9]])

scaler = preprocessing.StandardScaler() #创建缩放器
standardized = scaler.fit_transform(x)  #转换特征
standardized
# min-max缩放有个常见的替代方案，即将特征缩放为大致符合标准正态分布，这样的缩放，叫做标准化。（standardization) 这样数据能有一个等于
# 0的平均值，为1的标准差。xi'=(xi-avg(x))/σ
print('Mean: ',round(standardized.mean()))          #0
print('Standard deviation: ',standardized.std())    #1

# 如果数据中存在很大的异常值，可能会影响特征的平均值和方差，也会对标准化有影响。 这种情况下使用
# 中位数和四分数间距进行缩放会更有效果。scikit-learn 中可以调用 RobustScaler()

robust_scaler = preprocessing.RobustScaler()
robust_scaler.fit_transform(x)

# 4.3 归一观察值
import numpy as np
from sklearn.preprocessing import Normalizer

features = np.array([[0.5,0.5],
                    [1.1,3.4],
                    [1.5,20.2],
                    [1.63,34.4],
                    [10.9,3.3]])

normalizer=Normalizer(norm='l2')    #创建归一化器
normalizer.transform(features)      #转化特征矩阵
# 很多缩放方法是对特征进行操作的,但其实对观察值也可以进行缩放操作。Normalizer 可以对
# 单个观察值进行缩放，使其拥有一致的范围（总长度为1）。当一个观察值有多个相等的特征时（例如，做
# 文本分类时，每一个词或每几个词就是一个特征），经常使用这种类型的缩放。
# Normalizer 提供三个范数选项，默认值是欧式范数（Euclidean norm 常被称为L2范数）

features_l1_norm = Normalizer(norm='l1').transform(features)    #manhatten distance
features_l1_norm
# 当使用曼哈顿范数对观察值进行缩放后，它的元素总和为1
print("Sum of the first observation\'s values:",
      features_l1_norm[0,0]+features_l1_norm[0,1])

# 4.4生成多项式和交互特征
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

features = np.array([[2,3],
                     [2,3],
                     [2,3]])
polynomial_interaction = PolynomialFeatures(degree=3, include_bias=False)   #创建PolynomialFeatures对象
polynomial_interaction.fit_transform(features)      # 创建多项式特征
# array([[2., 3., 4., 6., 9.],
#        [2., 3., 4., 6., 9.],
#        [2., 3., 4., 6., 9.]])

interaction=PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
interaction.fit_transform(features)
# array([[2., 3., 6.],
#        [2., 3., 6.],
#        [2., 3., 6.]])         #忽略与自己相乘。只包括交叉特征

# 4.5转换特征
import numpy as np
from sklearn.preprocessing import FunctionTransformer

features = np.array([[2,3],
                     [2,3],
                     [2,3]])
# 定义一个简单函数
def add_ten(x):
    return x+10

ten_transformer=FunctionTransformer(add_ten)    #创建转换器
ten_transformer.transform(features) #转换特征矩阵

import pandas as pd
df =pd.DataFrame(features,columns=['feature_1','feature_2'])
df.apply(add_ten)   #用pandas 中的apply可进行同样的转换

# 4.6识别异常值
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

features, _=make_blobs(n_samples=10,n_features=2,centers=1,random_state=1)    #创建一个模拟数据
features[0,0] =10000  # 将第一个观察值替换为极端值
features[0,1] =10000
outlier_detector = EllipticEnvelope(contamination=.1)   #创建识别器
outlier_detector.fit(features)      #拟合识别器
outlier_detector.predict(features)  #预测异常值
# contamination (污染指数）表示异常值在观察值中的比例，
# 如果你认为数据只有很少的几个异常值，可以将contamination设置得小一点

features=features[:,0]
def indicies_of_outlier(x):
    q1,q3=np.percentile(x,[25,75])
    iqr = q3-q1
    lower_bound = q1-(iqr*1.5)
    upper_bound = q3+(iqr*1.5)
    return np.where((x>upper_bound)|(x<lower_bound))

indicies_of_outlier(features)   #IQR 识别异常值

# 4.7处理异常值
import pandas as pd
houses = pd.DataFrame()
houses['Price'] = [534433,392333,293222,4322032]
houses['Bathrooms'] = [2,3.5,2,116]
houses['Square_feet']=[1500,2500,1500,48000]

houses[houses['Bathrooms']<20]  #[1]筛选异常值

houses['Outlier'] = np.where(houses["Bathrooms"]<20,0,1)    #[2]基于布尔条件语句来创建特征
houses

houses["Log_Of_Square_Feet"]=[np.log(x) for x in houses["Square_feet"]] #[3]对异常值特征进行转换，降低异常值的影响
houses

# 4.8将特征值离散化
import numpy as np
from sklearn.preprocessing import Binarizer

age = np.array([[6],
                [12],
                [20],
                [36],
                [65]])

binarizer = Binarizer(18)       #创建二值化器
binarizer.fit_transform(age)    #转化特征

np.digitize(age,bins=[20,30,64])    #将特征离散化 设置多个阈值  #bins(左边界）:左闭合右开. 使用right=True 可以包括右值

# 4.9使用聚类的方式将观察值分组
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
features ,_ = make_blobs(n_samples=50, n_features=2,centers=3,random_state=1)   #创建模拟特征矩阵
dataframe = pd.DataFrame(features,columns=["feature_1","feature_2"])
clusterer=KMeans(3,random_state=10)  #创建K-Means 聚类器
clusterer.fit(features)             # 将聚类器应用在特征上
dataframe["group"] = clusterer.predict(features)
dataframe.head(5)

# 4.10删除带有缺失值的观察值
import numpy as np
features = np.array([[1.1,11.1],
                     [2.2,22.2],
                     [3.3,33.3],
                     [4.4,44.4],
                     [np.nan,55]])
features[~np.isnan(features).any(axis=1)]   # 只保留没有(用~来表示）缺失值的观察值

import pandas as pd
dataframe = pd.DataFrame(features,columns=["feature_1","feature_2"])
dataframe.dropna()  #删除带有缺失值的观察值

# 4.11填充缺失值

import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

features,_=make_blobs(n_samples=1000,n_features=2,random_state=1)

scaler = StandardScaler()   #标准化特征
standardized_features=scaler.fit_transform(features)

true_value = standardized_features[0,0] #将第一个特征向量的第一个值替换成缺失值
standardized_features[0,0]=np.nan

# [1]KNN 算法来预测缺失值
features_knn_imputed = KNN(k=5,verbose=0).fit_transform(standardized_features)   #预测特征矩阵中的缺失值

# 对比真实值和填充值
print('True Value:',true_value)
print('Imputed Value:',features_knn_imputed[0,0])

# [2]特征值的平均值，中位数或者众数来填充
from sklearn.preprocessing import Imputer
mean_imputer=Imputer(strategy="mean", axis=0)
features_mean_imputed = mean_imputer.fit_transform(features)

print('True Value:',true_value)
print('Imputed Value:',features_mean_imputed[0,0])