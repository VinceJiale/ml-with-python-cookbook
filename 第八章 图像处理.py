# 8.1 加载图像
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane.jpg', cv2.IMREAD_GRAYSCALE)    # 把图像导入成灰度图
plt.imshow(image, cmap="gray"), plt.axis("off")   # 显示图像
plt.show()

type(image) # 显示数据类型 numpy.ndarray
image.shape # 矩阵的维度与图像分辨率完全相同   每个元素代表像素值。黑色0 -白色 255 之间的变化
image[0, 0]  # 第一个像素点的像素值

image_bgr = cv2.imread('/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane.jpg', cv2.IMREAD_COLOR)   # 以彩色模式加载图像
image_bgr[0, 0]  # array([195, 144, 111], dtype=uint8) b:blue, g:green, r:red (BGR)
plt.imshow(image_bgr), plt.axis("off")
plt.show()      # 颜色是反的, 因为 matplotlib 用的是 RGB格式

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)   # 转换成RGB 格式 OpenCV 默认使用 BGR格式。 很多图像应用程序使用 RGB格式
plt.imshow(image_rgb), plt.axis("off")
plt.show()

# 8.2 保存图像
import cv2

image = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_new.jpg",image)
# 保存的图像格式由文件扩展名决定,imwrite将直接覆盖现有文件,而不会输出一条错误消息或请求你确认。

# 8.3 调整图像大小
import cv2
from matplotlib import pyplot as plt
image = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)
image_50x50 = cv2.resize(image,(50,50))     # 将图片调整到50x50像素
plt.imshow(image_50x50, cmap="gray"), plt.axis("off")
plt.show()
# 调整图像大小是图像预处理中常见的任务。 原因有两点：
# 1. 原始图像的形状和大小各异，但是被用作特征的图像必须有相同的大小(像素数),然而图像大小的标准化会带来一些信息上的损失。
# 图像是包括信息的矩阵。当减小图像的像素时,矩阵的尺寸也会缩小，而其中包括的信息也随之减少。
# 2. 机器学习算法可能需要成百上千张图像，甚至更多。如果这些图像非常大，就会占用大量内存,通过调整图像的大小(像素数)可以大大减少
# 内存的使用量。
# 在机器学习中 常见的图像规格有: 32x32, 64x64, 96x96, 256x256

# 8.4 裁剪图像
import cv2
from matplotlib import pyplot as plt
image = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)
image_cropped = image[:,:128]   # 选择所有行和前128列
plt.imshow(image_cropped, cmap="gray"), plt.axis("off")
plt.show()
# 因为OpenCV 将图像表示为矩阵，所以选择想要保留的行和列，可以轻松裁剪图像。

# 8.5 平滑处理图像
# 平滑处理图像就是将每个像素的值变换成其相邻的平均值。 相邻像素和所执行的操作在数学上被表示为一个核。这个核的大小决定了平滑的程度，
# 核越大，产生的图像越平滑。 这里用5x5的核对每个像素周围的值取平均值。
import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)
image_blurry = cv2.blur(image,(5,5))    # 平滑处理图像
plt.imshow(image_blurry, cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

kernel = np.ones((5,5))/25.0  # 我们所使用的平滑核
kernel
# 核的中心是要处理的像素,而其余元素是该像素的相邻元素。由于所有元素具有相同的值(被归一化为1)，
# 因此每个元素对要处理的像素点有相同的权重。可以使用filter2D 在图像上手动应用核，以产生与上文类似的平滑效果
image_kernel = cv2.filter2D(image,-1, kernel)   # 应用手动制造的核
plt.imshow(image_kernel,cmap="gray"), plt.xticks([]), plt.yticks([])
plt.show()

# 8.6 图像锐化
# 创建一个能突出显示目标像素的核, 然后使用 filter2D 将其应用于图像:
import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)
kernel = np.array([[0,-1,0],
                   [-1,5,-1],
                   [0,-1,0]])   # 创建核
image_sharp = cv2.filter2D(image,-1,kernel) # 锐化图像
plt.imshow(image_sharp, cmap='gray'), plt.axis("off")
plt.show()

# 8.7 提升对比度
import cv2
from matplotlib import pyplot as plt
image = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)
image_enhanced = cv2.equalizeHist(image)    # 增强图像  灰色图
plt.imshow(image_enhanced, cmap='gray'), plt.axis("off")
plt.show()

# 如果equalizeHist 要应用彩色图需要先将格式转换为YUV格式
image_bgr = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane.jpg")    #加载图像
image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)  # 转换成YUV格式
image_yuv[:, :, 0] = cv2.cv2.equalizeHist(image_yuv[:, :, 0])   # 对图像应用直方图均衡
image_rgb = cv2.cvtColor(image_yuv,cv2.COLOR_YUV2RGB)   # 转换成RGB格式
plt.imshow(image_rgb), plt.axis("off")
plt.show()          # 增强图像 彩色
# YUV 格式： Y表示亮度，U和V表示颜色
# 如果直方图均衡能够使我们感兴趣的对象与其他对象或背景区分得更明显(并非总是如此), 那么它对我们的图像预处理流水线来说就是一个有价值的补充

# 8.8 颜色分离
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg")
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)  # 将BGR 转换成 HSV格式

lower_blue = np.array([50, 100, 50])    # 定义 HSV格式中蓝色分量的区间
upper_blue = np.array([130, 255, 255])  # HSV格式: H:色调，S：饱和度，V：亮度

mask = cv2.inRange(image_hsv, lower_blue, upper_blue)   # 创建掩模
image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)     # 应用掩模, 并将图像转换成所需的输出格式
image_rgb = cv2.cvtColor(image_bgr_masked,cv2.COLOR_BGR2RGB)    # 从BGR格式转换成RGB
plt.imshow(image_rgb), plt.axis("off")
plt.show()

plt.imshow(mask,cmap='gray'), plt.axis("off")   # 只保留掩模白色区域
plt.show()

# 8.9 图像二值化
import cv2
from matplotlib import pyplot as plt
image_grey = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)

max_output_value = 255
neighborhood_size = 99
substract_from_mean = 10
# 应用自适应阈值处理
image_binarized = cv2.adaptiveThreshold(image_grey,
                                       max_output_value,    # 用于输出像素的最大强度
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 将像素的阈值设置为相邻像素强度的加权和, 权重由高斯窗口决定，可以将该参数修改为 cv2.ADAPTIVE_THRESH_MEAN_C, 使用相邻像素的平均值作为阈值_
                                       cv2.THRESH_BINARY,
                                       neighborhood_size,   # 块大小(用于确定像素阈值的邻域大小
                                       substract_from_mean) # 调整阈值的常数(从计算中减去该常数)

plt.imshow(image_binarized,cmap="gray"), plt.axis("off")
plt.show()
# 图像二值化是指对图像进行阈值处理（thresholding), 即将强度大于某个阈值的像素设置为白色，并将小于该值的像素设置为黑色的过程，
# 还有一种技术叫做 自适应阈值处理(adaptive thresholding), 一个像素的阈值是由其相邻像素的强度决定的。当图像中不同区域光照条件
# 有异时，用这种方法处理很有帮助。

# 对图像进行阈值处理的一个主要的优点是可以对图像进行去噪---只保留最重要的元素。例如我们通常对印有文本的照片进行阈值处理，以提取
# 照片上的字母

# 8.10 移除背景
import cv2
import numpy as np
from matplotlib import pyplot as plt

image_bgr = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg")
image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)   # 先转换成RGB格式

rectangle = (0, 56, 256, 150)   # 矩阵的值: 左上角的x坐标，左上角的y坐标，宽，高

mask = np.zeros(image_rgb.shape[:2], np.uint8)  # 创建初始掩模

bgdModel = np.zeros((1,65), np.float64)     # 创建grabCut 函数所需要的临时数组
fgdModel = np.zeros((1,65),np.float64)

cv2.grabCut(image_rgb,
            mask,
            rectangle,
            bgdModel,   # 背景的临时数组   ※※
            fgdModel,   # 前景的临时数组   ※※
            5,      # 迭代次数 ※※
            cv2.GC_INIT_WITH_RECT)      # 使用定义的矩阵初始化

mask_2 = np.where((mask== 2)|(mask== 0), 0, 1).astype('uint8')  # 创建一个掩模，将确定或很有可能是背景的部分设置为0,其余设置为1

image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]    # 将图像与掩模相乘以除去背景

plt.imshow(image_rgb_nobg), plt.axis("off")
plt.show()
# grabCut 认为这个矩阵外的一切都为背景，并使用这些信息找出矩阵内可能的背景。算法最后生成一个掩模，标出确定为背景的区域，可能为背景的区域和前景区域，
# 可以看到图像仍有残留，我们可以手动将这些区域标记为背景，因为现实中有数千张图像需要处理，可以考虑保留"留有部分噪音的图像"

plt.imshow(mask, cmap="gray"), plt.axis("off")  # 显示掩模
plt.show()  # 黑色为矩阵外的区域, 灰色区域为GrabCut 认为的可能为背景的区域，白色区域则可能是前景图像

plt.imshow(mask_2,cmap="gray"), plt.axis("off") # 显示掩模，黑色区域与灰色区域合并
plt.show()

# 8.11 边缘检测
import cv2
import numpy as np
from matplotlib import pyplot as plt

image_gray = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg")

median_intensity = np.median(image_gray)    # 计算像素强度的中位值

lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))      # 设置阈值
upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold) # 应用Canny 边缘检测器
plt.imshow(image_canny,cmap="gray"), plt.axis("off") # 显示掩模，黑色区域与灰色区域合并
plt.show()

# 8.12 角点检测
import cv2
import numpy as np
from matplotlib import pyplot as plt

image_bgr = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg")
image_gray = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)

# 设置角点检测器的参数
block_size = 2
aperture = 29
free_parameter = 0.04

detector_responses = cv2.cornerHarris(image_gray, block_size, aperture, free_parameter)  # 检测角点

detector_responses = cv2.dilate(detector_responses,None)    # 放大角点标志

threshold = 0.02
image_bgr[detector_responses > threshold * detector_responses.max()]=[255,255,255]  # 只保留大于阈值的结果，并把他们标记成白色
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)     # 转换成灰度图

plt.imshow(image_gray, cmap="gray"), plt.axis("off")
plt.show()

# Harris 检测器 是检验两条边缘线条的交叉点的常用方法。 Harris 检测器会寻找一个窗口(邻域)，这个窗口的微小移动(晃动窗口)
# 会引发窗口像素值的大幅变化。    CornerHarris 中的3个重要参数， block_size 代表角点检测中的窗口尺寸，aperture 代表sobel算子尺寸
# free_parameter 用于控制对角点检测的严格程度，值越大，识别的角点越平滑
plt.imshow(detector_responses,cmap="gray"), plt.axis("off")     # 显示可能的角点 然后筛选最可能的角点
plt.show()

image_bgr = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg")
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# 待检测角点的数量
corners_to_detect = 10
minimum_quality_score = 0.05
minimum_distance = 25

corners = cv2.goodFeaturesToTrack(image_gray,       # 检测角点
                                  corners_to_detect,
                                  minimum_quality_score,
                                  minimum_distance)
corners = np.float32(corners)

for corner in corners:      # 在每个角点上画白圈
    x, y = corner[0]
    cv2.circle(image_bgr, (x, y), 10, (255, 255, 255), -1)

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

plt.imshow(image_rgb, cmap="gray"), plt.axis("off")
plt.show()
# goodFeaturesToTrack 确定一组固定数量的明显的角点
# 其与 Harris 检测器类似。 有三个参数: 待检测的角点的数量，角点的最差质量，焦点间最短的欧式距离

# 8.13 为机器学习创建特征
import cv2
import numpy as np
from matplotlib import pyplot as plt
image= cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg",cv2.IMREAD_GRAYSCALE)
image_10x10 = cv2.resize(image, (10, 10))   # 将图像的尺寸转换成10x10
image_10x10.flatten()       # 将图像数据转换成一维向量

plt.imshow(image_10x10, cmap="gray"), plt.axis("off")
plt.show()

image_10x10.shape   # (10,10) 图像维度是10X10
image_10x10.flatten().shape # (100,)    展开后,长度为100 （10*10）的向量   这是图像的特征向量，可与其他图像的特征向量结合，生成可供机器学习算法使用的数据

# 彩色图像
image_color = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg",cv2.IMREAD_COLOR)
image_color_10x10 = cv2.resize(image_color, (10, 10))
image_color_10x10.flatten().shape       # (300,) 彩色一个像素有多个值表示(常见3个), 10x10x3

# 8.14 将颜色平均值编码成特征
import cv2
import numpy as np
from matplotlib import pyplot as plt
image_bgr = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg",cv2.IMREAD_COLOR)

channels = cv2.mean(image_bgr)  # 计算每个通道的平均值
observation = np.array([(channels[2],channels[1],channels[0])])     # 交换红色和蓝色的通道，将图像从BGR 格式转换成RGB
observation
plt.imshow(observation), plt.axis("off")
plt.show()
# 本节代码的输出结果是样本的三个特征值，分别来自图像的各颜色通道。这些特征可以和其他特征一样用于机器学习算法，根据颜色对图像进行分类

# 8.15 将色彩直方图编码成特征
import cv2
import numpy as np
from matplotlib import pyplot as plt
image_bgr = cv2.imread("/Users/caleb/Documents/Python Scripts/python 机器学习手册/simulated_datasets-master/images/plane_256x256.jpg",cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)

features=[]
colors=('r','g','b')    # 为每一个颜色通道计算直方图

for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb],   # 图像
                             [i],           # 颜色通道的序号
                             None,          # 不使用掩模
                             [256],         # 直方图尺寸
                             [0, 256])       # 范围
    features.extend(histogram)

observation = np.array(features).flatten()  # 将样本特征值展开成一维数组

observation[0:5]

image_rgb[0,0]  # 显示通过RGB通道的值

import pandas as pd
data = pd.Series([1,1,2,2,3,3,3,4,5])   # 创造一些数据
data.hist(grid=False)
plt.show()      # 显示直方图

colors = ("r", "g", "b")
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image_rgb],       # 图像
                             [i],           # 颜色通道的序号
                             None,          # 不使用掩模
                             [256],        # 直方图尺寸
                             [0,256])     # 范围
    plt.plot(histogram, color=channel)
    plt.xlim([0,256])
plt.show()

