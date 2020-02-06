<!--
 * @描述: 
 * @版本: V1_0
 * @作者: LiWanglin
 * @创建时间: 2020.02.05
 * @最后编辑人: LiWanglin
 * @最后编辑时间: 2020.02.05
 -->

# OpenCV 开发手册（ Python 版）

## 一. OpenCV 介绍

- 这节主要介绍 OpenCV 相关的内容，包括 OpenCV 的简介以及 OpenCV-Python 的简介。

### 1.1 OpenCV 简介

- OpenCV（开源计算机视觉库：<http://opencv.org> ）是BSD许可的开源库，由加里·布拉德斯基 (Gray Bradsky) 于 1999 年创立，第一版于2000年问世。它轻量级而且高效——由一系列 C 函数和少量 C++ 类构成，可以运行在 Linux、Windows 和 Mac OS 操作系统上，同时又提供了 Python、Ruby、MATLAB 等语言的接口，实现了图像处理和计算机视觉方面的很多通用算法。
最新版本是 4.1.2 ，2019 年 10 月发布。
- OpenCV目前的应用领域  
![OpenCV Application Area](./doc_image/opencv_application.png)

### 1.2 OpenCV-Python 简介

- OpenCV-Python 是 OpenCV 的 Python API，旨在解决计算机视觉问题的 Python 绑定库。
- OpenCV-Python 使用了 Numpy ，这是一个高度优化的库，用于使用 MATLAB 风格的语法进行数值运算。所有 OpenCV 数组结构都与 Numpy 数组相互转换。这也使与使用 Numpy 的其他库（例如 SciPy 和 Matplotlib ）的集成变得更加容易。

## 二. OpenCV 入门篇

- 本节主要介绍 OpenCV-Python 开发环境的搭建以及一系列基础的 OpenCV API，让读者对 OpenCV 的应用有一个初步的了解。

### 2.1 搭建 OpenCV 开发环境

- OpenCV 开发环境的搭建主要分为两个步骤，第一步是安装 Anaconda，第二步是安装 OpenCV-Python.

(1) 安装 Anaconda

Anaconda 是一个开源的 Python 发行版本，其包含了 conda、Python 等180多个科学包及其依赖项。通过安装 Anaconda ，能够大量减少配置 Python 环境的时间，减少许多不必要的麻烦。

- 下载 Anaconda
进入Anaconda官方网站 <https://www.anaconda.com/distribution> 下载相对的版本。    
![anncoda web](./doc_image/anaconda.png)  
选择 Python3.7 , 64 位版下载。  
- 安装 Anaconda  
在 Anaconda 的安装过程中，一般都是点击下一步就可以了。但有个地方要注意：
![install Anaconda](./doc_image/install_anaconda.png)  
画红圈的地方要勾选，将 Anaconda 添加到环境变量。
- 为Anaconda配置清华镜像源  
Anaconda 默认的镜像源都在国外，访问不但速度慢，而且经常不稳定。在国内使用的话，把 Anaconda 的镜像源配置为清华镜像源，不仅访问稳定，而且下载速度快，非常适合下载安装 Python 的各种函数库。  
在cmd下运行命令：conda config --set show_channel_urls yes，在用户目录下生成 .condarc 文件。  
修改.condarc文件里面的内容：

    ```python
    channels:
    - defaults
    show_channel_urls: true
    default_channels:
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
    custom_channels:
    conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
    ssl_verify: true
    ```

- 修改 Aoaconda 的 Python 版本  
由于 Python-3.7 版本还没有经过系统的测试，可能存在不稳定的情况，为了避免这种情况。所以我们须更换稳定的版本，在这里，我们选用经过系统测试的 Python-3.6 版本。  
在cmd里面输入：conda install python=3.6 将 Aoaconda 的 Python 版本由 3.7 版本变更为 3.6 版本。  
![Python-version](./doc_image/python-version.png)  
下载完成后，可以在cmd输入ipython查看python版本.
![Python-version2](./doc_image/python-version2.png)  
从图中可以看到，当前python版本为3.6.9。

(2) 安装OpenCV-Python

- 下载 OpenCV-Python  
进入网站：<https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv> ，选择 opencv_python-4.1.2+contrib-cp36-cp36m-win_amd64.whl 文件开始下载。
![OpenCV-python](./doc_image/OpenCV-Python.png)  
- 安装OpenCV-Python  
下载完成后，在cmd输入：pip install + opencv_python-4.1.2+contrib-cp36-cp36m-win_amd64.whl 文件的绝对路径
![OpenCV-python2](./doc_image/OpenCV-Python2.png)  
如果出现：Successfully installed opencv-python-4.1.2+contrib，则表示安装成功。
- 版本验证  
进入 ipython ，输入：

    ```python
    >>> import cv2 
    >>> cv2.__version__
    ```

可以查看 OpenCV-Python 版本  
![OpenCV-python3](./doc_image/OpenCV-Python3.png)  
从图中可以看出，当前 OpenCV-Python 版本为4.1.2.  

(3) 执行一个简单的Opencv程序
代码如下：( test_1_1_test_opencv.py)

```python
import cv2 as cv

image_data = cv.imread("./opencv_manual/test_image/lenacolor.png", cv.IMREAD_COLOR) # 读取图片数据
cv.imshow("Demo1", image_data)    # 显示图片
cv.waitKey(0)   # 等待用户按下按键
cv.destroyAllWindows()  # 摧毁所有显示图片的窗口
```

运行结果：  
![test_1_demo.py运行结果](./doc_image/test_1_1_test_opencv.png)

### 2.2 OpenCV 入门函数讲解

- 本小节主要讲述基本的 OpenCV 函数使用，包括简单的图片操作，简单的视频操操作 ，画图功能以及处理鼠标事件。

#### 2.2.1 简单的图片操作

(1) 目标

- 学习如何读取图片，显示图片，保存图片
- 学习函数：cv.imread(), cv.imshow() , cv.imwrite()
- 在Matplotlib中显示图片

(2) 读取图片

- 使用函数 cv.imread() 读取图像。该图像应位于工作目录中，或者应提供完整的图像路径。
- 第二个参数是一个标志，用于指定应读取图像的方式.  

| 读取标志 |含义 | 数值 |  
|:----:|:----:|:----:|
|cv.IMREAD_COLOR|加载彩色图像。图像的透明度将被忽略。这是默认标志.|1|
|cv.IMREAD_GRAYSCALE :|以灰度模式加载图片|0|
|cv.IMREAD_UNCHANGED |保持原格式不变|-1|

- 代码演示：  

代码：(test_2_1_imread.py)

```python
import cv2 as cv

image = cv.imread("./opencv_manual/test_image/lena512.bmp", cv.IMREAD_UNCHANGED)
print("image的类型为：", type(image))# 打印图片类型
print("image = \n", image)# 打印图片数据
```

运行结果：  
![test_2_imread.py运行结果](./doc_image/test_2_1_imread.png)  
从图中我们可以看出 cv2.imread() 的返回值为 numpy.ndarray 类型。

- 注意：  
如果图片的路径错误，不会有任何的错误输出，只是返回值为None。

(3) 显示图片

- 使用函数 cv.imshow() 在窗口中显示图像。窗口会自动适合图像尺寸。
- 第一个参数是窗口名称，它是一个字符串。第二个参数是我们的图片。您可以根据需要创建任意多个窗口，但窗口的名称要不相同。
- 代码演示：  
代码：(test_2_2_imshow.py)

```python
import cv2 as cv

image = cv.imread("./opencv_manual/test_image/lena512.bmp", cv.IMREAD_UNCHANGED)  # 读取图片
cv.imshow("Test_3", image)     # 开辟一个窗口显示图片
cv.waitKey(0)                  # 等待用户按下按键
cv.destroyAllWindows()         # 释放所有窗口
```  

运行结果：  
![test_3_imshow.py运行结果](./doc_image/test_2_2_imshow.png)  

(4) 写入图片

- 使用函数 cv.imwrite() 来保存图片
- 第一个参数是文件名，第二个参数是你想要保存的图片
- 代码演示：  
代码：(test_2_3_imwrite.py)  

```python
import numpy as np
import cv2 as cv

img = cv.imread("./opencv_manual/test_image/lena512.bmp", cv.IMREAD_UNCHANGED)    # 读取图片
cv.imshow("image", img)     # 显示图片
k = cv.waitKey(0)       # 等待用户按键
if(k == 27):            # 按下 ESC 键摧毁窗口
    cv.destroyAllWindows()
elif(k == ord('s')):     # 按下 S 键保存图片并退出
    cv.imwrite("messigray.png", img)
    cv.destroyAllWindows()
```

(5) 配合 Matplotlib 使用

- Matplotlib 是 Python 的绘图库，可为您提供多种绘图方法。在这里，您将学习如何使用 Matplotlib显示图像。您可以使用 Matplotlib 缩放图像，保存图像等。
- 代码演示：
代码：(test_2_4_use_matplotlib.py)

```python
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("./opencv_manual/test_image/lena512.bmp", cv.IMREAD_UNCHANGED) # 读取图片
# 设置图片在 matplotlib 的显示方式
plt.imshow(img, cmap="gray", interpolation='bicubic')
plt.show()          # 显示图像
```  

运行结果：  
![test_5_use_matplotlib.py运行结果](./doc_image/test_2_4_use_matplotlib.png)

- 注意
OpenCV 加载的彩色图像处于 BGR 模式。但是Matplotlib 以 RGB 模式显示。因此，如果使用 OpenCV 读取彩色图像，则 Matplotlib 中将无法正确显示彩色图像。

#### 2.2.2 简单的视频操作

(1) 目标

- 学会读取视频，播放视频和保存视频
- 学习从相机捕捉和显示
- 学习函数：cv.VideoCapture(), cv.VideoWriter()

(2) 从相机捕获视频

- 想要捕获一个视频，你需要创建一个 VideoCapture 对象，它的参数可以是设备索引或视频文件名称。设备索引只是指定哪个摄像机的编号。相机捕获完成之后，需要使用 cap.release() 结束捕获。
- 代码演示：  
代码：(test_2_5_capture_video.py)

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)    # 创建 VideoCapture 对象
if not cap.isOpened():       # 相机打开失败
    print("相机打开失败")
    exit()                  # 退出程序

while True:
    # 一帧一帧的捕获，如果正确读取帧，ret 返回 True
    ret, frame = cap.read()

    if not ret:       # 读取帧出错
        print("不能接收帧，退出中 ...")
        break
    # 将 BGR 图像转化为灰度图
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 显示图像，循环显示，相当于播放视频
    cv.imshow("frame", gray)
    if cv.waitKey(1) == ord('q'):       # 按下 Q 键退出
        break
cap.release()       # 在结尾的时候，一定要释放捕获
cv.destroyAllWindows()      # 摧毁所有创建的窗口
```

- cap.read() 返回一个布尔值 ( True/False ) 。如果读取正确，他将会返回 True 。
- cap.isOpened() 判断相机是否初始化成功。如果相机初始化成功，返回 True.
  
(3) 从文件播放视频

- 它与从摄像机捕获相同，只是摄像机索引更换为视频名称。在播放帧的时候，应选取适当的 cv.waitKey() 参数。如果参数过小，视频播放速度将会变得很快；参数过大，视频播放速度会变慢（这就是您可以慢动作显示视频的方式）。正常情况下25就可以了。
- 代码演示：
代码：(test_2_6_play_video.py)  

    ```python
    import numpy as np
    import cv2 as cv

    # 创建 VideoCapture 类
    cap = cv.VideoCapture("./opencv_manual/test_video/vtest.avi")  

    while cap.isOpened():       # 视频播放完毕，退出循环
        ret, frame = cap.read()     # 读取视频数据

        if not ret:                 # 读取数据出错
            print("视频解析失败，退出中 ...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cv.imshow("frame", gray)    # 显示图片
        if cv.waitKey(25) == ord('q'):   # 控制播放速度，按 Q 键退出
            break
    cap.release()       # 关闭视频
    cv.destroyAllWindows()  # 摧毁所有创建的窗口
    ```

- 注意
确保安装了正确版本的 ffmpeg 或 gstreamer 。有时，使用 Video Capture 不能成功，主要是由于 ffmpeg / gstreamer 安装错误。

(4) 保存视频

- 对于保存图片，非常简单，仅仅使用 cv.imwrite() 就可以了。但对于保存视频来说，则需要做更多的工作。
- 在保存视频的时候，我们应该创建一个 VideoWriter 对象。在创建 VideoWriter 对象时，传入的参数有：输出文件名，指定 FourCC 编码，fps( frames per second 每秒的帧数) ，帧大小以及 isColor flag 标志。如果 isColor 为 True ，编码器使用彩色框，否则将与灰度框一起使用。  
- FourCC 是一个 4 字节的代码，用于指定视频编解码器。在<http://fourcc.org> 网站上，你可以找到可用的代码列表。它取决于平台。遵循编解码器对保存视频来说效果很好。
  - 在 Fedora 上：DIVX, XVID, MJPG, X264, WMV1, WMV2
  - 在 Windows 上：DIVX
  - 在 OSX 上：MJPG (.mp4), DIVX (.avi), X264 (.mkv)
- 对于 MJPG ，FourCC 代码以 cv.VideoWriter_fourcc('M','J','P','G') 或 cv.VideoWriter_fourcc(*'MJPG') 的形式传递。
- 代码演示：
代码：(test_2_7_save_video.py)

    ```python
    import numpy as np
    import cv2 as cv

    cap = cv.VideoCapture(0)        # 打开相机

    fourcc = cv.VideoWriter_fourcc(*'XVID')     # 定义编码对象
    # 创建 VideoWriter 对象
    out = cv.VideoWriter("output.avi", fourcc, 20.0 ,(640, 480))

    while cap.isOpened():
        ret, frame = cap.read() # 读取帧

        if not ret:         # 帧的读取结果有误
            print("不能接受数据帧。退出中 ...")
            exit

        out.write(frame)        # 写入帧

        cv.imshow("frame", frame)   # 显示图像

        if cv.waitKey(1) == ord('q'):   # 按 Q 退出
            break

    # 工作完成后，释放所有内容
    cap.release()
    out.release()

    cv.destroyAllWindows()  # 摧毁窗口
    ```

#### 2.2.3 OpenCV 中的绘图功能

(1) 目标

- 学习使用 OpenCV 绘制不同的几何形状
- 学习函数：cv.line(), cv.circle(), cv.rectangle(), cv.ellipse(), cv.putText()
  
(2) 参数
在上述的函数中，你将看到一些常见的参数，如下所示：

- img : 要添加图像的图片
- color : 形状的颜色。对于 BGR 图像而言，以元组的方式传递，如蓝色 (255, 0, 0) 。对于灰度图而言，仅仅传递灰度值就可以了。
- thickness : 线或圆等图形的粗细。如果对于封闭的图像（如圆）其thickness值为 -1 ，它将填充形状。默认值为 1 。
- lineType ：线的类型， 是否为 8-connected, anti-aliased 线等等。默认为 8-connected。
  
(3) 画线

- 要绘制一条线，你需要确定线的开始和结束坐标。
- 我们先创建一幅黑色背景图，然后在其左上角到右下角绘制一条蓝线。
- 代码演示：
代码：(test_2_8_draw_line.py)

    ```python
    import cv2 as cv
    import numpy as np

    img = np.zeros((512, 512, 3), dtype=np.uint8)  # 创建黑色背景图
    cv.imshow("img", img)   # 显示原图

    result = cv.line(img, (0 ,0), (511, 511), (0, 255, 0), 1)   # 在背景图上绘制线条
    cv.imshow("result", result)         # 显示处理结果图

    cv.waitKey(0)       # 等待按键
    cv.destroyAllWindows()  # 摧毁窗口
    ```

运行结果：  
![test_2_8_draw_line.py](./doc_image/test_2_8_draw_line.png)

(4) 画矩形框

- 要绘制矩形，你需要矩形确定左上角和右下角的坐标  
- 代码演示：(test_2_9_drawing_rectangle.py)

```python
import numpy as np
import cv2 as cv

# 创建一个黑色图框
img = np.zeros((512,512,3), np.uint8)

# 绘制一个蓝色的矩形框
cv.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)

cv.imshow("rectangle", img)         # 显示图片

cv.waitKey(0)

cv.destroyAllWindows()
```

运行结果：
![test_2_9_draw_rectangle](./doc_image/test_2_9_draw_rectangle.png)

(5) 绘制圆形

- 绘制圆形需要确定圆心点的位置和半径
- 代码演示：(test_2_10_draw_circle.py)
代码：

    ```python
    import cv2 as cv
    import numpy as np

    # 创建一个黑色图框
    img = np.zeros((512,512,3), np.uint8)

    # 绘制一个红色的圆
    cv.circle(img, (447,63), 63, (0,0,255), -1)

    cv.imshow("circle", img)         # 显示图片

    cv.waitKey(0)

    cv.destroyAllWindows()
    ```

运行结果：
![test_2_10_draw_circle](./doc_image/test_2_10_draw_circle.png)

(5) 绘制椭圆

- 为了绘制椭圆，我们需要通过一系类的参数。首先我们需要确定椭圆中心坐标 (x, y) ，然后确定椭圆的轴长度（长轴长，短轴长），再确定椭圆在逆时针方向旋转的角度，最后确定起始角和结束角（起始角和结束角表示从长轴沿顺时针方向测量的椭圆弧的开始和结束。例如，给定0和360表示整个椭圆）
- 代码演示
代码：（test_2_11_draw_ellipse.py）

    ```python
    import cv2 as cv
    import numpy as np

    # 确定两个背景板，形成对比
    ellipse_1 = np.zeros((512,512,3), np.uint8)
    ellipse_2 = np.zeros((512,512,3), np.uint8)

    # 两者的起始角不同，一个为 0° ，一个为 90°
    cv.ellipse(ellipse_1, (256,256), (100,50), 0, 0, 360, 255, -1)
    cv.ellipse(ellipse_2, (256,256), (100,50), 0, 90, 360, 255, -1)

    # 显示图像
    cv.imshow("ellipse_1", ellipse_1)
    cv.imshow("ellipse_2", ellipse_2)

    cv.waitKey()

    cv.destroyAllWindows()
    ```

运行结果：
![test_2_11_draw_ellipse](./doc_image/test_2_11_draw_ellipse.png)

(6) 绘制多边形

- 要绘制多边形，我们需要确定各个顶点的位置，然后将各个顶点的位置变成 Rows * 1 * 2 的形状, Rows 是顶点的数目，其中数据类型要为 int32
- 代码演示:（test_2_12_draw_polygon）
代码：

    ```python
    import cv2 as cv
    import numpy as np

    # 确定两个背景板，形成对比
    img = np.zeros((512,512,3), np.uint8)

    # 确定 4 个点，四个点按照顺序连接成一个封闭图形
    pts = np.array([[50,15],[120,30],[170,50],[30,100]], np.int32)
    pts = pts.reshape((-1,1,2))         # 改变顶点 shape
    cv.polylines(img, [pts], True, (0,0,255))

    cv.imshow("polygon", img)

    cv.waitKey(0)
    cv.destroyAllWindows()
    ```

运行结果
![test_2_12_draw_polygon](./doc_image/test_2_12_draw_polygon.png)

(7) 添加文本

- 在添加文本的时候，需要指定下列参数：
  - 要添加的文本（不支持中文）
  - 文本的位置
  - 字体类型
  - 字体大小
- 字体类型请查看：[字体类型](https://docs.opencv.org/4.2.0/d6/d6e/group__imgproc__draw.html#ga0f9314ea6e35f99bb23f29567fc16e11)
- 代码演示(test_2_13_puttext.py)
代码：

    ```python
    import cv2 as cv
    import numpy as np

    img = np.zeros((512,512,3), np.uint8)

    font = cv.FONT_HERSHEY_SIMPLEX  # 确定字体类型
    cv.putText(img, 'OpenCV', (10,400), font, 4, (255,255,255), 2 , cv.LINE_AA)

    cv.imshow("text", img)      # 图片显示

    cv.waitKey(0)
    cv.destroyAllWindows()
    ```
运行结果：
![test_2_13_puttext](./doc_image/test_2_13_puttext.png)

## 三. OpenCV 进阶篇

### 3.1 OpenCV 核心操作

### 3.2 OpenCV 图像处理

## 四. OpenCV 高级篇

### 4.1 特征检测与描述

### 4.2 视频分析

### 4.3 摄像机定标与三维重建

### 4.4 机器学习

### 4.5 计算摄影

### 4.6 目标检测

## 五. 总结