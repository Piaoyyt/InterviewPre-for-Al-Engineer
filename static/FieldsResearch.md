
- [1.人脸质量评价](#FIQA)
- [2.美学质量评价]()
- [3.目标检测](#ObjectDetection)
   - [3.1经典方法](#ClassicalMethod)
   - [3.2常见问题](#CommonProblems)
   - [3.3前沿方法](#FrontierPaper)
- [4.人脸识别](#FaceRecognition)
   - [4.1经典方法](#FRClassicalMethod)
   - [4.2常见问题](#ClassicalQuestion)
   - [4.3]()
## <a id="FIQA"></a>1.人脸图像质量评价
> 
### <a id ='FIQAclassificalM'></a>1.1经典的方法
## <a id="ObjectDetection"></a>3.目标检测篇
### <a id="ClassicalMethod"></a>3.1经典方法
- Two-stage
  #### 1.滑动窗口
  >滑动窗口产生候选框→候选框的特征送入到SVM分类器进行分类、线性回归进行回归
  #### 2.R-CNN
  >不再采用滑动窗口，而是用选择性搜索的方式得到候选区域→不同的区域通过resize传入到网络模型里面得到不同区域的
  > 特征→传入到SVM分类器里面进行分类+线性回归回归。
  #### 3.SPP-Net
  >为了解决resize导致的目标出现形变的问题，引入了SPP即调整金字塔池化，并且先提取整张图的特征，然后在特征层面
  > 去选取候选区域→传入到SVM分类器进行分类+线性回归回归。
  #### 4.Fast-R-CNN
  >借鉴了SPP的思想，采用了ROI池化（即理解为单一尺寸的池化），并且在特征层面去提取候选区域的特征→采用全连接层
  > 进行最后的分类和回归，分类采用softmax、回归采用smooth L1 loss。
  #### 5.Faster-R-CNN
  > 不再是提前生成候选区域，而是结合anchor的设计，让网络一部分RPN检验anchor是否是背景还是非背景，然后将
  检验为非背景的部分传入后面的分类回归网络进行最后的目标检测和回归，整体上看已经很接近one-stage的设计思路
  > 了，即尽可能将候选框提出和检测流程融到一块。
- One-stage
  #### 1.Yolo
  > 将待检测的图片划分成等大的网格，每个网格经过深度网络(后面是两层全连接层+reshape构成)映射到输出feature map的一个点上，该点的不同通道
  > 存储了网格里面是否存在目标的信息，在原论文的设计里面，每一个网格负责预测两个目标（其实实际里面更加简化了，
  > 即这两个目标默认是同一个），所以yolo里面实际的检测框数目很少，即S x S x 2，并且实际的目标只能有S x S个，所以
  > yolo对于那些在同一个网格里面出现多个目标的情况检测很差。
  #### 2.SSD
  > 采用深度网络模型VGG19作为特殊提取backbone部分，不过将全连接层换成了全卷积（全连接会损失空间上的信息），同时对于不同层上的
  > 特征图抽取出来分别来预测各个点上的框（每个点上会设默认的box，预测默认的框之间的偏差即可)。
  #### 3.Yolov2
  > yolo系列里面开始引入anchor的方法，使得能够检测的框的数目大大增加，；网络上采用了darknet19，并且将低层的特征和高层的特征进行
  > 融合，使得最后对小目标的检测效果更好。
  #### 4.Yolov3
  > 网络上采用了darknet53，并且加入了FPN的操作，将高层的特征通过上采样和低层的特征进行融合，使得高层的特征包含的信息更加丰富。
  > 
  #### 5.Yolov4
  > 网络结构采用了CSPDarkNet53+Neck（SPPNet+PAN）+yolov3head；框的回归采用了CIOU； 加入了很多的训练技巧：
     数据增强：Mosaic数据增强，即对四张图像进行随机的缩放、裁剪和排布，多张图像的融合；
     数据不平衡问题：hard负实例挖掘、在线困难实例挖掘，焦点丢失、标签平滑、知识蒸馏；
     正则化：DropPath、DropBlock；
     模拟物体遮挡：随机的选择部分矩形区域、在特征图上dropblock。
  #### 6.Yolov5
  > 网络结构增加了Focus操作，即切片操作； 自适应的锚框的计算，非预先选定，而是在训练过程中同步迭代； 自适应图片的缩放：即测试的时候采用缩减黑
  > 边的方式，而不是传统直接填充，分别求出长宽的缩放比例，并找到最小值；按照最小缩放比例对图像做同性缩放；padding到想要的尺寸。 
  > 其实就是先resize再padding，这样填充黑色比较少。
### <a id="CommonProblems"></a>3.2常见问题
#### 1.正负样本的划分方式以及为什么让一个gt对应多个正anchor？
> 1.正样本就是用来学习anchor(先验框)怎么回归的；2.关于正样本的选取，不同的算法设定的策略有所不同，对于yolov3,先判断物体中心点落在哪个网格内，然后计算目标框(gt)与该网格内所有anchor的iou,
取iou最大的那个为正样本；对于faster_rcnn，首先计算每一个anchor与目标框的iou,如果超过正样本阈值则设定为正样本，然后对每一个目标框将最大iou的anchor设置为正样本（此时iou可能小于正样本阈值），
保证每一个目标框最少有一个正样本与其匹配；3.每一个gt框匹配多个正样本，能够使模型学习的更充分，二阶段检测算法比一阶段算法精度更高，在一定程度上就是因为二阶段检测算法能够匹配更多的正样本以及设置更
好的正负样本比例，每个gt框只取对应一个IOU最大的正样本也可以，只是可能会影响模型的精度．
#### 2.对anchor-free和anchor-based的理解？
> 1.anchor-free：就是在检测的时候不依赖手动设计anchor，即让网络自己去学习实际的框大小和形状，不给一个基准，比如yolov1就是典型的anchor-free，其实anchor-free和实例分割里面的思想比较接近了，
> 即预测目标的中心位置以及框的大小（不同于anchor-based是对所有anchor对应的预测框进行回归和分类），因此anchor-free更像是一种像素点单anchor的情况。  
> 2.anchor-based：预先设计出不同的anchor，之后的框的检测和回归都是在这些anchor之上进行预测的，需要有比较多的先验知识在里面。  
- anchor-based的优缺点：
>   - 优点：  
      1.  手工设计了不同的anchor，后面直接网络在这些anchor上面预测，能够保证网络的性能不至于很差。  
      2.  由于设计的anchor大小和形状多变，对于最终的召回效果有提升作用，特别是小目标的检测。
>   - 缺点：  
      1.  生成的anchor需要较强的先验知识，即不同的任务下anchor就可能需要改动。  
      2.  生成的anchor可能存在很多冗余的，即存在大量的负样本，正负样本不平衡的问题，这也是导致one-stage性能不如two-stage的原因之一。  
      3.  正负样本的阈值问题也是一个需要设计的点。
- anchor-free的优缺点：
>    - 优点：  
      1. 不需要很强的先验知识，但是对网络模型的要求更高一点，冗余的框少了很多。
      2. 解空间更加的灵活，并且计算量会更小，使得检测模型更加高效。  
>    - 缺点：  
      1. 语义的模糊性，典型的比如有两个目标落在同一个网格区域里面，那么这样就会导致只能检测出一个出来，而anchor-based可以都检测出来。
      2. 因为缺失一个先验知识作为基准，使得检测结果不稳定，所以需要加入一些trick。
### <a id="FrontierPaper"></a>3.3前沿思想
- [yolox](https://zhuanlan.zhihu.com/p/392221567)
  > 1、Decoupled head(预测分支解耦):相比于yolov3到v5，其中的分类和回归都是共享一个特征，即非解耦的方式得到，yolox里面采用解耦的方式分别来预测
  > 框的类别和回归坐标。  
  > 2、强大的数据增强:添加Mosaic（将图片进行随机的旋转、裁剪、拼接）和MixUp（两张图以一定的比例对rgb值进行混合，同时需要模型预测出原本两张图中所有的目标）。  
  > 3、Anchor-free：使用anchor时，为了调优模型，需要对数据聚类分析，确定最优锚点，缺乏泛化性；增加了检测头复杂度，增加了每幅图像预测数量。
  > 使用ancho-freer可以减少调整参数数量，减少涉及的使用技巧。  
  > 4、[SiamOTA:](https://zhuanlan.zhihu.com/p/392221567) 其实就是正样本的选择方式，有的是直接将gt所在的那个格子中所有的预测框与gt计算
  > 取最高的作为正样本（yolov3）;有的是取所在网格一定半径的格子来计算得到多个正样本；这里anchor-free下每一个特征图只预测一组anchor，而SiamOta的
  > 核心就在于对于不同的gt，我们去计算出前10个损失最小的框，然后将这些框的IOU求和，求和的结果就是最后要选择的正样本数量，即对于不同gt选不同的正样本数量。
- [yolov6](https://tech.meituan.com/2022/06/23/yolov6-a-fast-and-accurate-target-detection-framework-is-opening-source.html)
  >统一设计了更高效的 Backbone 和 Neck ：受到硬件感知神经网络设计思想的启发，基于 RepVGG style[4] 设计了可重参数化、更高效的骨干网络 EfficientRep Backbone 和 Rep-PAN Neck。
  优化设计了更简洁有效的 Efficient Decoupled Head，在维持精度的同时，进一步降低了一般解耦头带来的额外延时开销。
在训练策略上，我们采用Anchor-free 无锚范式，同时辅以 SimOTA 标签分配策略以及 SIoU 边界框回归损失来进一步提高检测精度。
- [yolov7](https://arxiv.org/abs/2207.02696)
  > 主要在几个方面做了改进：1.模型的结构重参数化，即将原来训练中的参数在推理的时候转化为等效的另一组基本可以等效的参数，减少网络结构的复杂度，本文研究了
  > 如何高效的去替换；2.针对网络的参数提出了扩展和复合缩放，高效利用参数。
## <a id="FaceRecognition"></a>4.人脸识别篇
> 人脸识别即身份识别，高级语义信息提取的过程，整个人脸识别的流程大致可以分成人脸校正、人脸裁剪、人脸特征提取、人脸比对或者直接人脸识别（本质上也是人脸比对）。
### <a id="FRClassicalMethod"></a>4.1经典方法
#### 1.FaceNet
> 采用三元组的形式，即传入p、n、t三张人脸图像，让网络学习的特征表示使得p和t尽可能的近，使得n和t尽可能的远，所以这里采用的是三元损失。
#### 2.VGGFace
> 利用VGGNet来做的人脸识别工作，首先训练人脸分类器，然后对分类也就是最后的映射层进行三元组学习人脸表示。
#### 3.Sphereface（乘性角度）
> 引入了角度度量的思想，即在原始softmax分类损失的基础上加入了角度间隔的约束，使得原来的分类问题变得更难，即不光是要分类正确，还要使得不同的类别的决策边界之间
> 存在一定的角度，这样能够使得人脸特征更具判别性。Sphereface是对原来的角度乘以了一个因子，使得对人脸图像距离人脸中心的角度要求更小，即同一类的人脸特征表示将会
> 更加的紧凑。
#### 3.Cosface（加性余弦）
> 和Sphereface的思想是一致的，只不过是在余弦值的层面去约束，也就是相似度的层面，通过对原始的余弦值进行一个减值操作来使得原来相同角度余弦下的损失更大，即约束
相当于更强了。
#### 4.Arcface（加性角度）
> 在角度上做文章，同样也是使得在原来的角度下相似度会变低，即要么角度变大，要么是直接相似度变低(Cosface的做法)；这里是
> 让角度变大，即加上一个角度的偏移。
###<a id = 'ClassicalQuestion'><a/> 4.2常见问题的思考
#### 1.人脸识别方法的总结
人脸识别问题本质是一个分类问题，即每一个人作为一类进行分类检测，但实际应用过程中会出现很多问题。第一，人脸类别很多，如果要识别一个城镇的所有人，那么分类类别就将近十万以上的类别，另外每一个人之间可获得的标注样本很少，会出现很多长尾数据。根据上述问题，要对传统的CNN分类网络进行修改。

我们知道深度卷积网络虽然作为一种黑盒模型，但是能够通过数据训练的方式去表征图片或者物体的特征。因此人脸识别算法可以通过卷积网络提取出大量的人脸特征向量，然后根据相似度判断与底库比较完成人脸的识别过程，因此算法网络能不能对不同的人脸生成不同的特征，对同一人脸生成相似的特征，将是这类embedding任务的重点，也就是怎么样能够最大化类间距离以及最小化类内距离。

在人脸识别中，主干网络可以利用各种卷积神经网络完成特征提取的工作，例如resnet，inception等等经典的卷积神经网络作为backbone，关键在于最后一层loss function的设计和实现。现在从两个思路分析一下基于深度学习的人脸识别算法中各种损失函数。

思路1：metric learning，包括contrastive loss, triplet loss以及sampling method

思路2：margin based classification，包括softmax with center loss, sphereface, normface, AM-sofrmax(cosface) 和arcface。

- Metric Learning
  - Contrastive loss
    >深度学习中最先应用metric learning思想之一的便是DeepID2了。其中DeepID2最主要的改进是同一个网络同时训练verification和classification（有两个监督信号）。其中在verification loss的特征层中引入了contrastive loss。Contrastive loss不仅考虑了相同类别的距离最小化，也同时考虑了不同类别的距离最大化，通过充分运用训练样本的label
      信息提升人脸识别的准确性。因此，该loss函数本质上使得同一个人的照片在特征空间距离足够近，不同人在特征空间里相距足够远直到超过某个阈值。(听起来和triplet loss有点像)。
      ![](pics/ContrastiveLoss.jpg)
  - Triplet loss
    >Google在DeepID2的基础上，抛弃了分类层即Classification Loss，将Contrastive Loss改进为Triplet loss，只为了一个目的：学习到更好的feature。
直接贴出Triplet loss的损失函数，其输入的不再是Image Pair，而是三张图片(Triplet)，分别为Anchor Face, Negative Face和Positive Face。Anchor与Positive Face为同一人，与Negative Face为不同的人。那么Triplet loss的损失函数即可表示为：
      ![](pics/TripletLoss.jpg)
     本质上是使得anchor和postive的图像对映射得到的特征距离尽可能的小，anchor与negative的图像对映射得到的特征距离尽可能的大。
- 存在的不足：
  - 模型训练依赖大量的数据，拟合过程很慢。由于contrastive loss和triplet loss都是基于pair或者triplet的，需要准备大量的正负样本，，训练很长时间都不可能完全遍历所有可能的样本间组合。网上有博客说10000人、500000张左右的亚洲数据集上花一个月才能完成拟合。
  - Sample方式影响模型的训练。比如对于triplet loss来说，在训练过程中要随机的采样anchor face, negative face以及positive face，好的样本采样能够加快训练速度和模型收敛，但是在随机抽取的过程中很难做到非常好。
  - 缺少对hard triplets的挖掘，这也是大多数模型训练的问题。比如说在人脸识别领域中，hard negatives表示相似但不同的人，而hard positive表示同一个人但完全不同的姿态、表情等等。而对hard example进行学习和特殊处理对于提高识别模型的精度至关重要。
- 对此做的改进
  - finetune
    > 不再是直接用三元组来训练人脸识别网络，而是先用softmax训练人脸识别模型，然后再将顶层的分类层去掉，换成triplet层，相当于对特征进行校正，加快了速度。
  - 损失函数做的改进（引进对hard sample的挖掘，使得模型得到更具判别性的特征）
    > 对于batch中的每一张图片a，我们可以挑选一个最难的正样本和一个最难的负样本和a组成一个三元组。首先我们定义和a为相同ID的图片集为A，剩下不同ID的图片图片集为B，则TriHard损失表示为：  
     ![](pics/LosswithHardSample.jpg)  
    损失函数的另一形式也可表示为：
     ![](pics/LosswithHardSample2.jpg)  
     我们知道，contrastive loss和triplet分别是利用二元组和三元组的信息，并且没有考虑难样本的挖掘，针对此，有一种lifted structure feature embedding的方式被提出：
     ![](pics/ContrastTripletLiftedloss.jpg)  
    作者在此基础上给出了一个结构化的损失函数。如下图所示:
     ![](pics/StructedLoss.jpg)  
    通过这种方式，让类间最大距离（难样本）尽可能的小，同时使得类间最小的距离尽可能的大。
     ![](pics/StructedSimilarity.jpg)  
    上面的损失函数解决了难样本挖掘的问题，但是对于非常难的负样本，通常损失会非常的平滑，也就导致模型不能很好的学习到有效的信息，这时候可以加入自适应的损失，即提高那些比较难的
    负样本对的损失，让模型能够充分挖掘这样难负样本对的有效信息。  
     ![](pics/HardSampleLoss.jpg)    
     上面的beta即是来判断样本难易程度的阈值，对于同类的样本对，当然是在距离超过某一个值时判定为难，此时损失会增加；同理负样本对也一样。
  - 对sample方式的改进
    > 作者的分析认为，sample应该在样本中进行均匀的采样，因此最佳的采样状态应该是在分散均匀的负样本中，既有hard，又有semi-hard，又有easy的样本,通过计算样本对的距离的分布的概率，
     最后推出每一个距离下采样的比例，从而实现整个数据样本上的均匀采样。
     ![](pics/SampleMethods.jpg)
- Margin based
  > 所谓的margin based就是在margin也就是特征的角度层面进行改进，迫使类内特征更加紧凑，类间特征更加分散。
  - Center Loss
    > 在softmax的基础上加上样本特征距离类中心的距离:
     ![](pics/CenterLoss.jpg)
  - L-Softmax
    > 对于类内的样本特征角度乘上一个常数，从而使得类内的样本特征间距更加紧凑，本来夹角是10度的人脸特征对可能
    就已经够了，加入了这样一个常数后，假如是2，那么相同的条件下就会限制样本特征角度为5度。
     ![](pics/L-softmaxLoss.jpg)
     ![](pics/L-softmaxLoss2.jpg)
    但是这里没有对w进行归一化，即没有考虑到特征的模值带来的影响。
  - Normface
    > 在传统的softmax基础上对样本的特征和参数进行了归一化。
     ![](pics/Normface.jpg)
  - AM-softmax(CosFace、Sphereface、Arcface)
    > 综合了前面的特征归一化以及大间隔的损失函数特点进行的改进，改进的方向分别为在余弦值上、乘性角度、加性角度上。

#### 2.难样本问题
> 按照学习的难以来区分，我们的训练集可以分为Hard Sample和Easy Sample. 顾名思义，Hard Sample指的就是难学的样本 
（loss大），Easy Sample就是好学的样本（loss小），在人脸图像里面难样本可以理解为那些相似但非同一个人脸id的图像或者
不相似但是是同一个id的图像，通常是由于拍摄的环境影响的，人脸的角度、图像的噪声都会影响人脸特征的判别性，这样会让模型在
训练的时候捕捉到了对判别无关紧要甚至是干扰项的噪声。
