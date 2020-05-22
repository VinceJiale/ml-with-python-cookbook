# 7.1 把字符串转换成日期
import numpy as np
import pandas as pd
date_strings = np.array(['03-04-2005 11:35 PM',
                         '23-05-2010 12:01 AM',
                         '04-09-2009 09:09 PM'])
[pd.to_datetime(date,format='%d-%m-%Y %I:%M %p') for date in date_strings]

[pd.to_datetime(date,format='%d-%m-%Y %I:%M %p',errors="coerce") for date in date_strings]
# errors="coerce" 可以在当转换错误时不抛出异常(默认行为），但是会导致这个错误的值设置成 NaT (也就是缺失值)

# 7.2 处理时区
import pandas as pd
pd.Timestamp('2017-05-01 06:00:00',tz='Europe/London')  # 创建datetime

date = pd.Timestamp('2017-05-01 06:00:00')  # 创建datetime
date_in_london = date.tz_localize('Europe/London')  # 为之前创建的datetime对象添加时区信息
date_in_london

date_in_london.tz_convert('Africa/Abidjan') # 改变时区

dates = pd.Series(pd.date_range('2/2/2002', periods=3, freq='M'))  # 创建3个日期对series每个对象运用tz_localize/tz_convert
dates.dt.tz.localize('Africa/Abidjan')  # 设置时区

from pytz import all_timezones  # pandas 支持两种表示时区的字符串。建议使用 pytz
all_timezones[0:2]

# 7.3 选择日期和时间
import pandas as pd
dataframe = pd.DataFrame()
dataframe['date'] = pd.date_range('1/1/2001', periods=100000, freq='H')
dataframe[(dataframe['date'] > '2002-1-1 01:00:00') &
          (dataframe['date'] <= '2002-1-1 04:00:00')]   # 筛选两个日期之间的观察值

dataframe = dataframe.set_index(dataframe['date'])  # 设置时间为索引列
dataframe.loc['2002-1-1 01:00:00':'2002-1-1 04:00:00']  # 选择两个日期之间的观察值

# 7.4 将日期数据切分成多个特征
import pandas as pd
dataframe = pd.DataFrame()
dataframe['date'] = pd.date_range('1/1/2001',periods=150, freq='W')

dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute

dataframe.head(3)

# 7.5 计算两个日期之间的时间差
import pandas as pd
dataframe = pd.DataFrame()
dataframe['Arrived'] = [pd.Timestamp('01-01-2017'),pd.Timestamp('01-04-2017')]
dataframe['Left'] = [pd.Timestamp('01-01-2017'),pd.Timestamp('01-06-2017')]

dataframe['Left'] - dataframe['Arrived']    # 我们只需要保留数值，删去days
pd.Series(delta.days for delta in (dataframe['Left'] - dataframe['Arrived']))
# 有时我们想要的只是两个时间点的时间间隔(delta)，pandas 的TimeDelta 数据类型可以让这件事情变得很简单

# 7.6 对一周内的各天进行编码
import pandas as pd
dates = pd.Series(pd.date_range("2/2/2002",periods=3, freq="M"))
dates.dt.weekday_name   # 查看星期几
dates.dt.weekday        # 用数值来表示星期几 （星期一为0）

# 7.7 创建一个滞后的特征
import pandas as pd
dataframe = pd.DataFrame()
dataframe["dates"] = pd.date_range("1/1/2001", periods=5, freq="D")
dataframe['stock_price'] = [1.1, 2.2, 3.3, 4.4, 5.5]

dataframe["previous_days_stock_price"] = dataframe["stock_price"].shift(1)  # 让值滞后一行
dataframe

# 7.8 使用滚动时间窗口
import pandas as pd
time_index = pd.date_range("01/01/2020", periods=5, freq='M')
dataframe = pd.DataFrame(index=time_index)
dataframe['Stock_Price'] = [1, 2, 3, 4, 5]
dataframe.rolling(window=2).mean()  # 计算滚动平均值   window参数指定窗口的大小     常用语对时间序列数据做平滑处理

# 7.9 处理时间序列中的缺失值
import pandas as pd
import numpy as np
time_index = pd.date_range("01/01/2010", periods=5, freq="M")
dataframe = pd.DataFrame(index=time_index)
dataframe["Sales"] = [1.0, 2.0, np.nan, np.nan, 5.0]    # 创建带缺失数据的特征

dataframe.interpolate() # 对缺失数据进行插值

dataframe.ffill()   # 向前填充

dataframe.bfill()   # 向后填充

dataframe.interpolate(method='quadratic')   # 已知两个已知点之间是非线性的，可以用method参数来指定插值的方式

dataframe.interpolate(limit=1, limit_direction="forward")
# limit 限制补充数量，limit_direction设置最后一个已知值是向前/向后插值



