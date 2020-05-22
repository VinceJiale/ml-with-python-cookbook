# 3.0 简介
import pandas as pd
dataframe=pd.read_csv('/users/caleb/Documents/Python Scripts/python 机器学习手册/titanic.csv')
dataframe.head(5)

# 3.1 创建一个数据帧   3.2 描述数据
import pandas as pd
dataframe = pd.DataFrame()
dataframe['Name']=['Jacky Jackson','Steven Stevenson']
dataframe['Age']=[38,25]
dataframe['Driver']=[True,False]        # 增加列

dataframe

new_person= pd.Series(['Molly Mooney',40,True],index=['Name','Age','Driver']) # 创建行
dataframe.append(new_person,ignore_index=True)  #附加此行

dataframe.head(2)
dataframe.shape
dataframe.describe

# 3.3浏览数据帧
import pandas as pd
dataframe=pd.read_csv('/users/caleb/Documents/Python Scripts/python 机器学习手册/titanic.csv')

dataframe.iloc[0]   #选择第一行

dataframe.iloc[1:4] #选择三行

dataframe.iloc[:4]  #选择到第四行为止

dataframe = dataframe.set_index(dataframe['Name'])  #设置索引 以名字为索引
dataframe.loc['Allen, Miss Elisabeth Walton']   #loc 根据索引查找，iloc 根据行数查找

# 3.4根据条件语句来选择行
import pandas as pd
dataframe=pd.read_csv('/users/caleb/Documents/Python Scripts/python 机器学习手册/titanic.csv')
dataframe[dataframe['Sex'] == 'female'].head(2) #展示female的前两行

dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)]

# 3.5替换值    3.6重命名列
import pandas as pd
dataframe=pd.read_csv('/users/caleb/Documents/Python Scripts/python 机器学习手册/titanic.csv')

dataframe['Sex'].replace('female','woman').head(2)
dataframe['Sex'].replace(['female','male'],['woman','man']).head(5)
dataframe.replace(1,"One").head(2)
dataframe.replace(r"1st","First",regex=True).head(2)    #replace支持正则表达式

dataframe.rename(columns={'PClass':'Passenger Class', 'Sex':'Gender'}).head(2)

import collections
column_names= collections.defaultdict(str)  #创建字典

for name in dataframe.columns:
    column_names[name]

column_names        #查看字典

# 3.7计算最小值，最大值，总和，平均值与计数值   3.8查找唯一值
import pandas as pd
dataframe=pd.read_csv('/users/caleb/Documents/Python Scripts/python 机器学习手册/titanic.csv')

print('Maximum:',dataframe['Age'].max())
print('Minimum:',dataframe['Age'].min())
print('Mean:',dataframe['Age'].mean())
print('Sum:',dataframe['Age'].sum())
print('Count:',dataframe['Age'].count())
# 其他统计量 方差var,标准差std,峰值kurt，偏态skew，平均值标准差sem，众数mode,中文数median等
dataframe.count()

dataframe['Sex'].unique()   #筛选唯一值
dataframe['Sex'].value_counts() #显示唯一值及次数

dataframe['PClass'].value_counts()  # 有一位乘客PClass 为*。 出现异常值
dataframe['PClass'].nunique()   #显示有多少个唯一值

# 3.9处理缺失值
import pandas as pd
dataframe=pd.read_csv('/users/caleb/Documents/Python Scripts/python 机器学习手册/titanic.csv')
# isnull 和notnull 能返回布尔值来表达一个值是否缺失
dataframe[dataframe['Age'].isnull()].head(2)
# 替换缺失值
import numpy as np
dataframe['Sex']=dataframe['Sex'].replace('Man',np.nan)
dataframe = pd.read_csv('/users/caleb/Documents/Python Scripts/python 机器学习手册/titanic.csv',
                        na_values=[np.nan,'NONE',-999]) #设置缺失值

# 3.10 删除一列     3,11 删除一行
import pandas as pd
dataframe=pd.read_csv('/users/caleb/Documents/Python Scripts/python 机器学习手册/titanic.csv')

dataframe.drop('Age',axis=1).head(2)    #删除列
dataframe.drop(['Age','Sex'],axis=1).head(2)    #删除多列
dataframe.drop(dataframe.columns[1],axis=1).head(2) #指定列下标的方式删除

dataframe_name_dropped = dataframe.drop(dataframe.columns[0],axis=1)    #创建一个新的数据帧(此内改动）

dataframe[dataframe['Sex']!='male'].head(2)
dataframe_male_dropped=dataframe[dataframe['Sex']!='male'] #使用条件语句删除多行 df[]形式

# 3.12 删除重复行
import pandas as pd
dataframe=pd.read_csv('/users/caleb/Documents/Python Scripts/python 机器学习手册/titanic.csv')
dataframe.drop_duplicates().head(2) #只删除完全相同的两行
dataframe.drop_duplicates(subset=['Sex'])   #删除subset里面相同的行，保留最早出现的
dataframe.drop_duplicates(subset=['Sex'],keep='last')   #保留最后出现的

# 3.13根据值对行分组
import pandas as pd
dataframe=pd.read_csv('/users/caleb/Documents/Python Scripts/python 机器学习手册/titanic.csv')
dataframe.groupby('Sex').mean()  #根据Sex类的值进行分组，并计算每组的平均值
# groupby 需要与一些作用于组的操作配合使用 比如说计算综合统计量（平均值，中位数，总和）
dataframe.groupby('Survived')['Name'].count()   #按行分组，计算行数
dataframe.groupby(['Sex','Survived'])['Age'].mean() #二次分组，计算年龄平均值

# 3.14按时间段进行分组
import pandas as pd
import numpy as np

time_index = pd.date_range('06/06/2017',periods=100000, freq='30S') #创建日期范围
dataframe = pd.DataFrame(index=time_index)  #创建数据帧
dataframe['Sale_Amount'] = np.random.randint(1,10,100000)   #创建一列随机变量
dataframe.resample('W').sum()   #按周对行进行分组，计算每一周的总和
dataframe.resample('2W').sum()  #按两周分组，计算平均值
dataframe.resample('M').count()   #按月分组，计算行数    标签显示的右边的边界
dataframe.resample('M', label='left').count()    #按月分组，计算行数     标签显示左边的边界

# 3.15遍历一个列的数据
import pandas as pd
dataframe=pd.read_csv('/users/caleb/Documents/Python Scripts/python 机器学习手册/titanic.csv')
for name in dataframe['Name'][0:2]:
    print(name.upper())             # 以大写的形式打印前两行中的名字

[name.upper() for name in dataframe['Name'][0:2]]   # for循环，列表解析式

# 3.16对一列的所有元素应用某个函数    3.17 对所有分组应用一个函数
import pandas as pd
dataframe=pd.read_csv('/users/caleb/Documents/Python Scripts/python 机器学习手册/titanic.csv')

def uppcase(x):
    return x.upper()    #创建一个函数

dataframe['Name'].apply(uppcase)[0:2]   #应用函数，查看两行

dataframe.groupby('Sex').apply(lambda x:x.count())  #对行分组，然后在每一组上应用函数

# 3.18 连接多个数据帧
import pandas as pd
data_a = {'id':['1','2','3'],
          'first':['Alex','Amy','Allen'],
          'last':['Anderson','Ackerman','Ali']}
dataframe_a = pd.DataFrame(data_a,columns=['id','first','last'])

data_b = {'id':['4','5','6'],
          'first':['Billy','Brian','Bran'],
          'last':['Bonder','Black','Balwner']}
dataframe_b = pd.DataFrame(data_b,columns=['id','first','last'])

pd.concat([dataframe_a,dataframe_b],axis=0) #沿着行的方向连接两个数据帧
pd.concat([dataframe_a,dataframe_b],axis=1) #沿着列的方向连接两个数据帧

row = pd.Series([10,'Chris','Chillon'],index=['id','first','last'])
dataframe_a.append(row,ignore_index=True)   #附加一行

# 3.19合并两个数据帧
import pandas as pd
employee_data = {'employee_id':['1','2','3','4'],
                 'name':['Amy Jones','Allen Keys','Alice Bees','Tim Horton']}
dataframe_employees = pd.DataFrame(employee_data,columns=['employee_id','name'])

sales_data = {'employee_id':['3','4','5','6'],
              'total sales':[23456,2512,2345,1455]}
dataframe_sales = pd.DataFrame(sales_data,columns=['employee_id','total sales'])

pd.merge(dataframe_employees,dataframe_sales,on='employee_id')  #合并数据帧 默认等值连接
pd.merge(dataframe_employees,dataframe_sales,on='employee_id',how='outer')  #外连接
pd.merge(dataframe_employees,dataframe_sales,on='employee_id',how='left')   #左连接

pd.merge(dataframe_employees,dataframe_sales,left_on='employee_id',right_on='employee_id')  #按列名合并 可切换 如果是索引列 left_index=True
