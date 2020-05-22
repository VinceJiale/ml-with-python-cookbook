# 5.1对nominal型分类特征编码
# 利用 scikit-learn 的labelBinarizer 对特征进行 one-hot 编码 （独热编码）
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

feature = np.array([['Texas'],
                    ['California'],
                    ['Texas'],
                    ['Delaware'],
                    ['Texas']])
one_hot=LabelBinarizer()    # 创建one-hot 编码器
one_hot.fit_transform(feature)  #对特征进行one-hot 编码

one_hot.classes_    #array(['California', 'Delaware', 'Texas'], dtype='<U10') 可以用classes_方法输出分类

one_hot.inverse_transform(one_hot.transform(feature))   #对one-hot 编码 逆转换

import pandas as pd
pd.get_dummies(feature[:,0])    # 用pandas 对特征进行 one-hot 编码

multiclass_feature=[("Texas","Florida"),
                    ("California","Alabama"),
                    ("Texas","Florida"),
                    ("Delware","Florida"),
                    ("Texas","Alabama")]    #创建有多个分类的特征

one_hot_multiclass = MultiLabelBinarizer()  #创建能处理多个分类的 one-hot 编码器()
one_hot_multiclass.fit_transform(multiclass_feature)    #对特征进行one-hot 编码
one_hot_multiclass.classes  #输出分类

# one-hot 编码之后，最好从结果矩阵中删除一个one-hot 编码的特征，以避免线性依赖

# 5.2 对ordinal 分类特征编码
import pandas as pd
dataframe = pd.DataFrame({"Score": ["Low","Low","Medium","Medium","High"]})
scale_mapper = {"Low":1, "Medium":2, "High":3}  # 创建映射器
dataframe["Score"].replace(scale_mapper)    #使用映射器来替换特征
# Ordinal 特征用 值来表示 要注意 值与值之间的间隔。 如果分类之间的间隔有差异 需要斟酌修改 映射器的值

# 5.3的对特征字典编码
from sklearn.feature_extraction import DictVectorizer
data_dict = [{"Red":2,"Blue":4},
             {"Red":4,"Blue":3},
             {"Red":1,"Yellow":2},
             {"Red":2,"Yellow":2}]

dictvectorizer = DictVectorizer(sparse=False)   #创建字典向量化器   sparse=False 会输出一个稠密矩阵
features = dictvectorizer.fit_transform(data_dict)  #将字典转换成特征矩阵
features
feature_names=dictvectorizer.get_feature_names()
feature_names

import pandas as pd
pd.DataFrame(features,columns=feature_names)    #从特征中创建数据帧

# 在自然语言处理中,例如我们有大量的文档，对每个文档都用一个字典存放每个词在文档中出现的次数。 使用 dictvectorizer
# 可以很方便的创建特征矩阵，其中每个特征都表示一个词在某个文档中出现的次数

doc_1_word_count={"Red":2,"Blue":4}
doc_2_word_count={"Red":4,"Blue":3}
doc_3_word_count={"Red":1,"Yellow":2}
doc_4_word_count={"Red":2,"Yellow":2}

doc_word_counts=[doc_1_word_count,
                 doc_2_word_count,
                 doc_3_word_count,
                 doc_4_word_count]#创建列表
dictvectorizer=DictVectorizer(sparse=True)
dictvectorizer.fit_transform(doc_word_counts)   #将词频字典列表转换成特征矩阵
# 假设每个文档是一本书，则特征矩阵会变得很庞大，此时需要用到 sparse=True <稀疏矩阵>


# 5.4 填充缺失的分类值
# [1]最理想的方法是训练一个机器学习分类器来预测缺失值，通常使用KNN分类器
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[0,2.10,1.45],
             [1,1.18,1.33],
             [0,1.22,1.27],
             [1,-0.21,-1.19]])
X_with_nan = np.array([[np.nan,0.87,1.31],
                       [np.nan,-0.67,-1.19]])

clf = KNeighborsClassifier(3,weights='distance')    #训练KNN分类器
trained_model = clf.fit(X[:,1:],X[:,0])

imputed_values =trained_model.predict(X_with_nan[:,1:])   #预测缺失值的分类

X_with_imputed=np.hstack((imputed_values.reshape(-1,1),X_with_nan[:,1:]))  #将所预测的分类和它的其他特征连接起来

np.vstack((X_with_imputed,X))   #连接两个特征矩阵

# [2]用特征中出现次数最多的值来填充缺失值
from sklearn.preprocessing import Imputer
X_complete = np.vstack((X_with_nan,X))
imputer = Imputer(strategy='most_frequent',axis=0)
imputer.fit_transform(X_complete)

# 5.5 处理不均衡分类
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

iris = load_iris()
features = iris.data
target = iris.target
features=features[40:,:]
target = target[40:] #移除前40个观察值

target = np.where((target == 0),0,1)    #创建二元目标向量来标识观察值是否为类别0
target

weights={0:.9,1:.1} #创建权重
RandomForestClassifier(class_weight=weights)    #创建带权重的随机森林分类器

RandomForestClassifier(class_weight="balanced") #可以传入参数balanced，会自动创建与分类的频数成反比的权重。
                                                #训练一个带均衡分类权重的随机森林分类器

i_class0 = np.where(target ==0)[0]  #给每个分类的观察值打标签
i_class1 = np.where(target ==1)[0]

n_class0 = len(i_class0)
n_class1 = len(i_class1)    #确认每个分类观察值的数量

i_class1_downsampled = np.random.choice(i_class1,size=n_class0,replace=False)   #对于每个分类为0的观察值，从分类为1的数据中进行无放回的随机采样

np.hstack((target[i_class0],target[i_class1_downsampled]))  #将分类为0的目标向量和下采样的分类为1的目标向量连接起来

# 另一种选择 是对占多数的分类进行上采样   针对占多数的分类，从占小数的分类中进行有放回的随机采样
i_class0_upsampled = np.random.choice(i_class0,size=n_class1,replace=True)  #对于每个分类为1的观察值，从分类为0的数据中进行有放回的随机采样
np.concatenate((target[i_class0_upsampled],target[i_class1]))   #将向上采样得到的分类为0的目标向量和分类为1的目标向量连接起来
np.vstack((features[i_class0_upsampled],features[i_class1,:]))[0:5] #将上采样得到的分类为0的特征矩阵和分类为1的特征矩阵连接起来
