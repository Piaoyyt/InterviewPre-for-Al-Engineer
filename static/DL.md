
# Deep Learning Part
> This is the DL part which aims to summarize the knowledge of the 
> deep learning for Algorithm Interview.

## Main Content
- [1.基本概念](#BasicC)   
   - [优化器](#Optim)
   - [卷积层](#Convolution)
   - [池化层](#Pooling)
   - [归一化层](#NormalizeLaryer)
   - [注意力机制](#Attention)
- [2.深度网络模型](#NetModel)

## <a id="BasicC"></a>1.基本的概念

### <a id="Optim"></a>1.1.优化器
> **概念**：深度网络模型训练里面负责来控制网络参数更新方向和更新速度的结构。
#### 1.SGD
SGD的全称为Stochastic Gradient Descent,即随机梯度下降，因为SGD里面参数更新的
方向只和当前计算得到的梯度有关，并且参数的更新速率也是一个固定值，属于最早的一种
优化器。
#### 2.SGDM
相比于SGD增加了动量的概念，所谓的动量即考虑了前面的下降方向和下降幅度，让当前时刻的更新
不单单由当前的梯度决定，还受前面的更新方向影响，即动量的体现。
#### 3.Adagrad
自适应的梯度，这里所谓的自适应其实指的是梯度更新里面的学习率自适应，即梯度更新的快慢
程度，加入了不同时刻的梯度平方信息作为调整学习率的依据，使得学习率能够不断变化，这里
是不断的变小。
#### 4.RMSprop
不同于Adagrad里面将梯度平方和的信息作为调整学习率的依据，这样会导致二阶动量会不断的积累，
导致学习率下降的速度过快；这里是采用指数平均公式来计算，避免积累的问题。
#### 5.Adam
在梯度下降方向和梯度下降幅度两方面都加入了指数平均的思想，使得当前参数更新的方向更加的稳定，
并且为了防止在第一次更新的时候梯度更新会偏向于0，在里面还加入了偏置校正。
### <a id="Convolution"></a>1.2 卷积层
#### 1. 普通卷积
> 常规的卷积操作，滑动卷积核的位置，卷积核的维度和输入通道一样，多少种卷积核对应不同的模式，
> 理解为提取不同的特征。
#### 2. 分组卷积（降低普通卷积的复杂度）
> 将输入通道进行分组，每一个卷积核只需要与一组内的特征图进行卷积即可，使得总体的参数量变为原来的
组数分之一。
#### 3. 深度可分离卷积（分组卷积的特例）
>将输入通道的每一个通道作为一组，即分组卷积的特例。
#### 4. 空间可分离卷积
> 将卷积核分成两次分布卷积，即宽和高度上的卷积。
#### 5. 空洞卷积（实例分割里面感知细粒度的问题）
> 该卷积是在实例分割里面被提出来的，以用于解决一般的卷积感受野不够大，但是提高卷积核大小又会降低
输出的尺寸，并且增大参数量的缺点；因为实例分割要求最后的输出能够对原始图像里面的每个像素点都能够
比较准确的感知。空洞卷积就是在原始的卷积核元素之间插0，变化后的卷积核大小为k + (d - 1) x (k - 1)。
#### 6. 可变卷积（检测任务里面目标不规则的问题）
> 在目标检测、识别工作里面，目标角度的位置和角度的变换是一大挑战，即我们要求的目标检测和识别模型应该是空间不变性，
而CNN就具有这样的特性，因为卷积核的共享参数特性和局部性导致可以实现这一效果，但是实际的目标不光存在空间上的改动，
目标本身也可能存在角度的变换或者本身是一个不规则的形状，而一般的卷积核是一个正方形的，那直接用一个正方形的卷积核去
对不规则的目标作卷积势必会导致特征提取不够准确，这时候有几种思路去解决：1.通过数据增强扩充足够多的样本去增强模型适应
尺度变换的能力；2.设计更好的针对几何变换不变的特征或者算法；这两种思路都不是很好，泛化性不够，而且设计特征比较困难，
所以可变卷积就被提出来了。 
- 概念
    > 可变卷积中可变的是卷积核里面每一个像素点的位置，我们知道普通卷积核每一个像素点可以看作是离核中心有一个偏移量，并且
排成了一个正方形的形状，而可变卷积里面这个偏移量是一个可以学习的东西，即核里面每一个位置上的值去找原图像里面待匹配的位置时，
这个位置是不固定的，这样就对应了一个不规则的形状。
- 针对不规则问题的改进
    1. 原始的只是对位置上加了一个偏移量，但是这样容易引入无关的背景信息，导致干扰，所以在位置偏移量的基础上再加上一个权值，
即对每一个采样点的关注程度。
    2. 结合R-CNN（实例patch区域的特征作为teacher来指导主干模型，类似于蒸馏学习）
#### 7.卷积的代码实现
- 不用numpy库的实现方式
```python
def convolution(img, kernel_size, padding, stride):
    '''
    对输入的图像计算卷积
    :param img: 输入图像
    :param kernel_size: 核大小
    :param padding: 边界填充的大小
    :param stride: 核移动的步长
    :return:
    当核所有的值 =  1 / (kernel_size)**2 的时候，且满足 2 * padding - kernel_size + 1 = 0 时就等效于中值滤波了，并且stride
    只能为1，因为此时输出和输入等大；
    其余情况则是普通的卷积运算
    '''
    #计算输出图像的大小，给核赋值
    h, w, c = img.shape
    kernel = [[[1/kernel_size**2 for i in range(kernel_size)] for j in range(kernel_size)] for k in range(c)]
    outputh, outputw = (h + 2 * padding - kernel_size)//stride + 1, (w + 2 * padding - kernel_size)//stride + 1
    outputC = c
    #原始图像padding的范围，即置0的范围
    padding_Hrange = [i for i in range(padding)] + [h + padding + i for i in range(padding)]
    padding_Wrange = [i for i in range(padding)] + [w + padding + i for i in range(padding)]
    output = [[[0 for i in range(outputw)] for j in range(outputh)] for k in range(outputC)]
    for i in range(outputh):
        for j in range(outputw):
            #计算输出每一个位置的像素点对应的输入的起始行和列
            h_s, w_s = i * stride, j * stride
            #各个通道分别累乘
            for c in range(outputC):
                temp = 0
                #输入图像的行和列，位于两端的为padding部分
                for originR in range(h_s, h_s + kernel_size):
                    for originC in range(w_s, w_s + kernel_size):
                        #位于padding直接乘0即可
                        if originR in padding_Hrange or originC in padding_Wrange:
                            temp += 0
                        #对应位置相乘
                        else:
                            #输入图像和核对应位置相乘
                            temp += img[originR-padding, originC-padding, c] * kernel[c][originR - h_s][originC-w_s]
                output[c][i][j] = temp

    return output
import numpy as np
import cv2
img = cv2.imread('./girl.jpg')
print(img.shape)
ans = convolution(img, kernel_size = 5, padding = 2, stride = 1)
ans = np.array(ans).transpose(1, 2, 0)
cv2.imwrite('medianFilter.jpg', ans)

```
用一张图像作为示例，处理前：  
![](pics/girl.jpg)  
经过中值之后：  
![](pics/medianFilter.jpg)  
- 利用numpy库的实现方式
```python
def convolution(img, kernel_size, padding, stride):
    '''
    对输入的图像计算卷积
    :param img: 输入图像
    :param kernel_size: 核大小
    :param padding: 边界填充的大小
    :param stride: 核移动的步长
    :return:
    当核所有的值 =  1 / (kernel_size)**2 的时候，且满足 2 * padding - kernel_size + 1 = 0 时就等效于中值滤波了，并且stride
    只能为1，因为此时输出和输入等大；
    其余情况则是普通的卷积运算
    '''
    #计算输出图像的大小，给核赋值
    h, w, c = img.shape
    kernel = np.ones((kernel_size, kernel_size, c))/(kernel_size ** 2)
    outputh, outputw = (h + 2 * padding - kernel_size)//stride + 1, (w + 2 * padding - kernel_size)//stride + 1
    outputC = c
    #padding操作
    img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)))
    #原始图像padding的范围，即置0的范围
    output = np.random.randn(outputh, outputw, outputC)
    for i in range(outputh):
        for j in range(outputw):
            #计算输出每一个位置的像素点对应的输入的起始行和列
            h_s, w_s = i * stride, j * stride
            #各个通道分别累乘
            for c in range(outputC):
                temp = 0
                #输入图像的行和列，位于两端的为padding部分
                for originR in range(h_s, h_s + kernel_size):
                    for originC in range(w_s, w_s + kernel_size):
                        temp += img[originR, originC, c] * kernel[originR - h_s, originC-w_s, c]
                output[i, j, c] = temp
    return output
```
### <a id="Pooling"></a>1.3 池化层
为了选取特征图区域内的显著特征，并降低特征的维度，通过池化整合特征。
#### 1. 平均池化
>取平均值，反向传播的时候直接将值填充到所有的位置进行传播。
#### 2. 最大值池化
>取最大值，反向传播时通过记录的最大值位置去填充最大值
#### 3. 随机池化
>即随机取区域里面的值作为最后的结果。
#### 5. 空间金字塔池化
>即对特征图做不同尺寸的池化操作，因为池化永远输出一个值，所以最后的输出尺寸能够保持一致，和输入的
> 大小无关，在目标检测里面为了减少因为resize导致的目标出现形变问题，引入了SPP操作。
#### 6.全局池化
>即输出只有一个值，即全局上的池化操作。
### <a id="NormalizeLaryer"></a>1.4 归一化层
#### 1.BN
> 所谓的BN操作，其实就是对于不同样本所对应的特征图，对于同一个通道下的feature map进行归一化，所以一共
会有C也就是通道数这么多个用于归一化的均值和方差，即对B、H、W进行归一化操作，保留C。
#### 2.LN
> LN操作，就是对于同一个样本下不同的通道进行归一化，也就是多少个样本就会有多少个不同的均值和方差，所以这里
归一化的结果不受样本数目的影响，即对C、H、W进行归一化，保留B。
#### 3.IN
> IN操作，即实例归一化，将每一个通道里面宽高对应位置的像素值当成个体，去计算H x W这么多像素值的均值和方差，
所以一共会有C x B 这么多个均值和方差，即对H、W进行归一化操作，保留C和B。
#### 4.GN
> 即将通道进行分组，每一次对组内进行归一化操作，所以LN和IN可以看作是GN的特例，相当于是抽取部分通道进行归一化操作。
### <a id="Attention"></a>1.5 注意力机制
> 个人理解是一种让模型能够有区分度的关注输入不同局部信息的一种手段，1.按照区分度计算的方式可以分为self和cross，区分度的计算如果是根据输入（一般是序列信息）
> 内部块之间的联系来计算得到的，那么就称之为self-attetion；区分度的计算如果是将输入和外面其它（常称之为Key、
> Value）的联系得到的，那么就称之为cross-attetion。2.按照关注的局部信息不同也可以分为空间注意力（在宽度和高度上）、通道注意力（通道上）、两者都融入注意力
> （卷积块注意力）等。
#### 1.常见的注意力模块（按照关注的局部信息分类）
- SE Module
> Squeeze and Excitation，通过在高宽维度上进行GAP压缩维度然后对于每一个通道进行激活得到通道的注意力权重。
- CA Module
> Channel Attetion Module， 相比于SE Module加了一个GMP操作，即融入了更多的信息来丰富注意力的权重表示。
- SA Module
> Spatial Attention Module，在通道上进行GAP和GMP，得到通道数为2的attention map，然后通过一层1*1的卷积即可。
- CBA Module
> Convolution Block Attention Module，包含CA Module和SA Module两部分的注意力机制模块。
#### 2.常见的注意力（按照区分度的计算方式分类）
> 注意力的计算最典型的是通过内积来表示，通过Query、Key和Value三种类型数据之间的联系转化得到，其中Key和Value可以
> 理解为待度量的模板向量（在NLP里面Word2vec的方法里，key、value可以理解为就是语料库的向量），而Query是我们待查询
> 也就是待去转化的数据，即如何让最终的输出里面能够包含Query和Key、Value关系的信息，以NLP里面为例，假如Key里面有
> dog、house、bicycle等，而Query有一个cat，那么很显然cat和dog之间的关系更大，那么如何得到这个更大的关系，就需要
> 将cat和这些词逐个内积然后得到相似度，相似度越高说明attention更大，即最终的输出里面dog对cat的影响会更大，这就是
> attention的本质。
- Self-attention
>  本质上是让输入能够对内部的联系有所关注，所以这里面的Key、Value和Query是同一个输入序列的表示。
- Cross-attention
> 让输入序列能够融入和别的其它序列之间的关联度，而不是自己内部的联系，所以Key、Value和Query不是同一个序列的表示，
> 而是相当于一个外部的一个向量表示，典型的比如transformer里面encoder和decoder部分，encoder负责将输入的序列
> 进行向量的映射，得到序列的向量表示（这个表示会作为后面每一个decoder的Key、Value）而decoder则负责通过前t-1时刻
> 的序列来预测t时刻的词，即每一次的输入是不定长的序列，为了并行处理可以采用mask处理的方式。
- Multi-attention
> 其实多头attention只是将原来的向量表示拆分成多个子空间的表示形式然后最后再拼接起来，使得向量表示的信息更加的丰富，
对模型的表现性能有提升的作用。
## <a id="NetModel"></a>2.网络模型
#### 1.LeNet
> LeNet是最早被用来设计进行手写数字识别的网络，由几层卷积层和全连接层组成，结构很简单。
#### 2.AlexNet
> AlexNet是当年用于ImageNet目标检测竞赛的冠军网络，核心点是用了大的卷积核(5、7)，采用了局部响
应归一化（LRN)， 并加入了dropout。
#### 3.VggNet
> 相比于之前的网络结构，核心点在于探索了较深的网络可以提高网络的性能，用小的卷积核代替之前大的卷积核，
并且增加了网络的深度，增加了网络的非线性表达能力。因为两个大小为3的卷积核相当于一个大小为5的卷积核。
#### 4.GoogleNet
> 核心在于Inception模块的设计，该模块通过多个小的卷积核对feature map进行不同的处理，最
后将不同通道数的输出feature map拼接起来，增加了网络的宽度，实际上增加了网络的学习能力， 即每一层
能够学习到的东西更加丰富【1*1的卷积核实现维度的升降维】。 
#### 5.Incepv2-Incepv4
> Inception的不同版本都可以看作是基于googleNet做的改进。
  > - v2：主要加入了BN到网络里面，即批量归一化，对于同一通道的不同样本特征图进行归一化操作，原始论文里面
说降低了训练时的协方差偏移，使得训练能够更快的收敛。
  > - v3: 主要是将普通的卷积进行空间上分解，分解成宽和高度的两步卷积，但是尽管这种可以降低参数量，但是
这两种卷积方式并不能等价，所以如果对于网络性能要求比较好的时候这种方式通常不会采用。
  > - v4: 主要是结合了ResNet里面的残差结构来设计模型。
#### 6.ResNet
  >核心点就是残差块的设计，所谓的残差块其实就相当于一个恒等映射，一般的网络层是将输入直接映射到输出，当网络
  > 较深的时候，如果网络能力下降会使得输出丢失掉很多有用的信息，而残差块的设计直接让输入直接和输出相连，网络
  > 层相当于只学习到了输出和输入之间的差，即使是最差的情况，网络层的输出不会丢失输入的有效信息，这样的设计能
  > 够使得当网络层很深的时候仍然能够学习到有用的东西。
#### 7.SeNet
> SeNet全称为Squeeze and Excitation Net，即先将宽和高压缩成一维的形状，那么每一个通道就得到了各自的值，
> 这个值作为激活前的值经过sigmoid能够得到一个权重，即每个通道的权重，可以看作通道上的注意力，然后和原始的特征图
> 进行相乘。
#### 8.MobileNet
> MobileNet是轻量化网络里面的一个典型代表，轻量化的方式是对于卷积操作进行设计的，即不同于常规的卷积，而是采用
> 分组卷积的形似，将原始特征图的通道进行分组，使得卷积的参数量能够大大降低；同时为了避免分组卷积使得不同通道之间
> 的联系被忽视，在后面又加入了逐像素的卷积操作，即大小为1的卷积核，这种可以融合不同通道之间的信息。
#### 9.ShuffleNet
> 核心点在于shuffle the channel，我们知道分组卷积会造成一个后果，就是不同组的channel之间的信息被忽略了，因
> 为组分好之后，不同组就无法交互了，而shuffle就是提前把通道进行打乱，然后对打乱后的分组，这样就可以解决这个问题。
> 
