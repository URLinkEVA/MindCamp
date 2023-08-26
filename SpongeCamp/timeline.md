# 821

## 分子模拟+dl

分尺度

### 增强抽样算法

### 基于物理和数据的混合模型

- 小蛋白/小分子

统一在框架

## 前景

物理数据驱动

大模型

场景化2B应用（覆盖各行业）

mindx SDK

原生加速网络库

```
大算子融合
整图下沉
自适应梯度切分
```

## mindspore2.0新特性

### 回顾

物理驱动PINNs，PDE正向求解，通过物理约束实现无监督

数据驱动alphafold2

物理加数据DeePMD，google流体力学，小数据集训练AI力场再推理

### 编程新范式

函数式与面向对象融合编程

DualCore框架，AI和函数式编程原生融合，共用同一套微分逻辑

### 即时编译

ms.jit修饰器

- 一行切换动静态图
- 即时编译，被修饰函数转为整图

### functorch

functorch需要手动将Module转换为function，mindspore直接支持Cell进行函数内调用和函数变换

### numpy

mnp.

### scipy

[ref](https://zhuanlan.zhihu.com/p/462806946)

### Vmap

自动向量化特性，批处理逻辑从函数中脱离

## ModelArts

AI落地难（开发算力人才）

## 科学计算基础操作

2023Summer

密钥

SHA256:senqZo6hCXlkeQtOmVxSPM/CGZZOz00b4+HFE/CyK98



```
git clone 地址
cd application
cd mindsponge
mkdir community
git status
cd applications
git add .
git commit -m "homework"

git push

bash build.sh -e ascend -j 128
```



# 822

## 业界趋势及实践

deepmind加速排序

[论文地址](https://www.nature.com/articles/s41586-023-06004-9)

二度回顾PINNs代表的物理及数据驱动

布局加套件落地

### 流体Flow

大模型

- 数据预处理
- vit
- 小波变换

### SOTA模型套件SciAI

待发布

## 分子动力学

模拟原理拓展和应用

### 模拟原理

回顾发展

<img src="D:\term4\活动\SpongeCamp\image-20230822100127642.png" alt="image-20230822100127642" style="zoom:80%;" />

- 量子力学方法

薛定谔方程采用近似，精度越高计算越难

- 分子力学方法

经典力学描述等价量子效应

各原子笛卡尔坐标



![image-20230822101328510](D:\term4\活动\SpongeCamp\image-20230822101328510.png)

传统分子力场的数学形式

![image-20230822101610549](D:\term4\活动\SpongeCamp\image-20230822101610549.png)

分子力场开发流程

<img src="D:\term4\活动\SpongeCamp\image-20230822101731090.png" alt="image-20230822101731090" style="zoom:50%;" />

过场繁琐，尝试ml替代

### 突破分子模拟局限性

模拟尺度小

![image-20230822103445699](D:\term4\活动\SpongeCamp\image-20230822103445699.png)

### 未来展望

分子动力学模拟-时间平均

实验观测-系综平均

各态历经Ergodicity

- 推荐书单

<img src="D:\term4\活动\SpongeCamp\image-20230822104818315.png" alt="image-20230822104818315" style="zoom: 50%;" />



## 智能分子模拟套件



体系模板Template



速度Verlet积分器

约束控制器（linear constraint solver）约束体系键长

温度的生成与计算，温度控制器Thermostat Controller

积分器与控温器的协同工作

mindsponge主程序是sponge，将体系、势能和优化器三模块组装起来

近邻表Neighbour List

基于偏置势

埋拓动力学Metadynamics

基于能量包装器的增强采样方法

运行模拟与回调函数Callback

集成变量Colvar与度量函数Metrics

H5MD模拟轨迹文件



## QM/MM

半经验方法为主

hartreefock与DFT

AM1半经验方法

- 单中心积分近似
- 双中心积分近似

精确刻画势能面

启发其他



## 上机|建模与模拟

tutorial_c01.ipynb

tutorial_c02.ipynb



## 作业

![image-20230823014843313](D:\term4\活动\SpongeCamp\image-20230823014843313.png)

1.为 case1.pdb 构建一个 PBC 系统，运行10ps 的正常产品模拟，绘制 phi-psi 分布图。

提示: 主要基于 p03，并添加 phi 和 psi 的CVs输出。

2.使用相同的系统，运行10ps 的 MetaD 模拟，绘制 phi-psi 分布图。

提示: 添加 MetaD 模块，并将 WithEnergyCell 作为 p04 的done。

3.比较结果，并分析 MetaD 作用。

### 1

**tutorial_p02.py**：读取1个pdb蛋白质分子，构建周期性水盒子，做能量极小化。

**tutorial_p03.py**：读取p02保存的水盒子，进行蛋白模拟流程示例：NVT模拟—NPT模拟—成品模拟。

p02保存的水盒子是p02.pdb

phi-psi



Barostat cannot be used for the system without periodic boundary condition.



### 3

MetaD增强采样



# 823

## 药物设计

### 药物发现途径

- 基于现象发现
- 基于靶标发现

高通量筛选

### 药物发挥作用

锁钥模型，药物与作用靶标结合

非共价结合



化合物对蛋白质活性的控制

直接占据活性部位（抑制）

别构调控（激活或抑制on/off state）Allosteric regulation 

在临床实验上ml筛部分

### 药物设计分类

- 基于靶标三维结构

- 不依赖靶标三维结构

### 计算机辅助蛋白质结合化合物发现

- 搜索已知化合物数据库
- 片段连接方法
- 从头生长de novo

分子对接具体筛选

蛋白质结合口袋探测算法

### 存在问题

可靠有效算法

准确打分函数（精确结合自由能计算）

靶标结构柔性的处理

## 基于ms图神经网络的应用

图学习任务与方法

- 边任务、节点任务、整图任务

- 深度学习、矩阵分解法、统计方法、随机游走

### 生物组学的网络推理

降本增效

### 分子性质预测

#### 矩阵分解法

![image-20230823104412007](D:\term4\活动\SpongeCamp\image-20230823104412007.png)

#### 随机游走（偏dfs）

走不同结点概率衡量

![image-20230823104544279](D:\term4\活动\SpongeCamp\image-20230823104544279.png)

红色箭头走领域，类卷积想法，ReLU

#### 消息传递架构

![image-20230823105126951](D:\term4\活动\SpongeCamp\image-20230823105126951.png)

#### 池化前后传播

![image-20230823105431642](D:\term4\活动\SpongeCamp\image-20230823105431642.png)

#### 图的池化

![image-20230823105652238](D:\term4\活动\SpongeCamp\image-20230823105652238.png)

#### 图卷积的表征学习

大模型的前身

![image-20230823105836336](D:\term4\活动\SpongeCamp\image-20230823105836336.png)

word2vec邻域



#### 基于随机游走的表征学习

![image-20230823110139890](D:\term4\活动\SpongeCamp\image-20230823110139890.png)

借鉴思想，算力有限追求速度



#### 变分自编码（VAE）

正态分布空间中采样

![image-20230823110433412](D:\term4\活动\SpongeCamp\image-20230823110433412.png)

#### 图变分自编码（Graph VAE）

![image-20230823110627962](D:\term4\活动\SpongeCamp\image-20230823110627962.png)

归一化再重构，向量到矩阵做外积



#### Transformer

![image-20230823112337368](D:\term4\活动\SpongeCamp\image-20230823112337368.png)

#### 节点的位置编码

![image-20230823112823143](D:\term4\活动\SpongeCamp\image-20230823112823143.png)

#### Graphormer

![image-20230823112908283](D:\term4\活动\SpongeCamp\image-20230823112908283.png)

#### 可解释性

![image-20230823113246597](D:\term4\活动\SpongeCamp\image-20230823113246597.png)

黑盒模型

![image-20230823113621503](D:\term4\活动\SpongeCamp\image-20230823113621503.png)

#### 群论和等变性

![image-20230823113849671](D:\term4\活动\SpongeCamp\image-20230823113849671.png)

## aichemist算法包

不同场景训练不同模型，图神经网络动态构建

构建Graph类

### GNN实现

message\aggregate\combine

![image-20230823115346887](D:\term4\活动\SpongeCamp\image-20230823115346887.png)



## 基于传统和机器学习方法的分子对接程序DSDP的介绍与应用

每年住院人次1亿/y > 一款药物研发十几年

CADD->AIDD计算机到AI辅助药物设计

### 分子对接流程概述

![image-20230823140945097](D:\term4\活动\SpongeCamp\image-20230823140945097.png)

![image-20230823141143171](D:\term4\活动\SpongeCamp\image-20230823141143171.png)

去重排序/成药性分析 improve sr

```
from rdkit import Chem
...
AllChem.GetMorganFingerprint()
```

### 蛋白质结合口袋预测

![image-20230823144312658](D:\term4\活动\SpongeCamp\image-20230823144312658.png)

机器学习引入位点预测的两种策略

- 与传统方法相结合提高传统方法的精度
- 端到端机器学习方法

### 构象采样

刚性对接，半柔性对接，柔性对接

传统随机采样算法：蒙特卡洛，遗传算法

#### EquiBind

![image-20230823150419548](D:\term4\活动\SpongeCamp\image-20230823150419548.png)

#### TankBind

![image-20230823150437452](D:\term4\活动\SpongeCamp\image-20230823150437452.png)

#### DiffDock

![image-20230823150451634](D:\term4\活动\SpongeCamp\image-20230823150451634.png)

![image-20230823150621737](D:\term4\活动\SpongeCamp\image-20230823150621737.png)

### 蛋白质小分子相互作用评估

打分函数

![image-20230823150709443](D:\term4\活动\SpongeCamp\image-20230823150709443.png)

多个数据进行打分

### DSDP传统方法与ml优势结合

![image-20230823151908948](D:\term4\活动\SpongeCamp\image-20230823151908948.png)

![image-20230823152827161](D:\term4\活动\SpongeCamp\image-20230823152827161.png)



#### 三线性插值

![image-20230823152159062](D:\term4\活动\SpongeCamp\image-20230823152159062.png)

![image-20230823152727046](D:\term4\活动\SpongeCamp\image-20230823152727046.png)

### conclusion

![image-20230823154325455](D:\term4\活动\SpongeCamp\image-20230823154325455.png)



## MolEdit基于生成式学习的分子编辑

### 功能分子设计

基团锚定再link

### AI时代

生成式学习

比如给蛋白口袋/分子片段/结构/分子性质

- 分子式化学空间
  - 离散，表示简洁，优化难，要近似
  - 数据多
  - 分子图有稀疏性

- 坐标空间
  - 物理上完备表示，良定义梯度优化
  - 数据少
  - 构象采样额外复杂度

选取合适模型

### MolGAN：分子图+GAN

分子图是稀疏的

### EDM：3D结构+扩散模型







## 药物分子的性质预测及分子对接





## 作业

![image-20230823115647095](D:\term4\活动\SpongeCamp\image-20230823115647095.png)





### 进阶作业

![image-20230823115903734](D:\term4\活动\SpongeCamp\image-20230823115903734.png)



# 824

## 自由能微扰计算的理论与实践：SPONGE的应用

### 自由能微扰（FEP）计算

nn，dock初筛

![image-20230824090558684](D:\term4\活动\SpongeCamp\image-20230824090558684.png)

### 分子动力学

![image-20230824091210549](D:\term4\活动\SpongeCamp\image-20230824091210549.png)

熵



### 增强采样

![image-20230824092432034](D:\term4\活动\SpongeCamp\image-20230824092432034.png)

### 炼金术自由能计算

![image-20230824092905578](D:\term4\活动\SpongeCamp\image-20230824092905578.png)

不断加中间态，线性插值

![image-20230824092935257](D:\term4\活动\SpongeCamp\image-20230824092935257.png)

### FEP计算

![image-20230824093422287](D:\term4\活动\SpongeCamp\image-20230824093422287.png)

模拟退火算法[退火算法(Annealing)简介与详解](https://blog.csdn.net/bingokunkun/article/details/118583729)



### 温度积分增强抽样（ITS）

![image-20230824094803782](D:\term4\活动\SpongeCamp\image-20230824094803782.png)

### 实践部分

![image-20230824103012115](D:\term4\活动\SpongeCamp\image-20230824103012115.png)

## 华为云AI在生物医药领域的探索和应用

缩短70%时间，提升10%sr

- 靶点发现/个性化药物
- 靶点模拟，精准对接



残差全连接网络（RFCN）

resnet，densenet



![image-20230824113006731](D:\term4\活动\SpongeCamp\image-20230824113006731.png)

pangu先导药研发周期压缩到1个月

![image-20230824113455298](D:\term4\活动\SpongeCamp\image-20230824113455298.png)





## 蛋白质结构预测：从生物学观察到算法

### 蛋白质及蛋白质折叠

anfinsen实验：去折叠的蛋白质在体外可自发再折叠，结构信息蕴含在序列之中

![image-20230824140355990](D:\term4\活动\SpongeCamp\image-20230824140355990.png)

### 比较建模法

![image-20230824141041912](D:\term4\活动\SpongeCamp\image-20230824141041912.png)

### 规范法

![image-20230824141219799](D:\term4\活动\SpongeCamp\image-20230824141219799.png)

![image-20230824141408870](D:\term4\活动\SpongeCamp\image-20230824141408870.png)



## 深度学习蛋白结构预测方法介绍

PSP protocols



accessing evolutionary information-Transformers





## 深度学习和实验信息在蛋白结构预测问题中的应用



## 深度学习和实验信息辅助的蛋白质结构预测上机演示



# 825

## 蛋白质设计：从能量函数优化到人工智能生成



![image-20230825090249609](D:\term4\活动\SpongeCamp\image-20230825090249609.png)

![image-20230825090348890](D:\term4\活动\SpongeCamp\image-20230825090348890.png)

能量依赖于结构

![image-20230825090439669](D:\term4\活动\SpongeCamp\image-20230825090439669.png)

![image-20230825090546409](D:\term4\活动\SpongeCamp\image-20230825090546409.png)

![image-20230825090816883](D:\term4\活动\SpongeCamp\image-20230825090816883.png)

### 漏斗模型

![image-20230825091101269](D:\term4\活动\SpongeCamp\image-20230825091101269.png)

### 决定蛋白质自由能地貌的相互作用

#### 溶剂效应

![image-20230825091140875](D:\term4\活动\SpongeCamp\image-20230825091140875.png)

#### 协同效应

自由能净变化（大/小）

![image-20230825091522308](D:\term4\活动\SpongeCamp\image-20230825091522308.png)

![image-20230825091728144](D:\term4\活动\SpongeCamp\image-20230825091728144.png)

### 蛋白质结构计算建模

#### 基本范式

![image-20230825091852879](D:\term4\活动\SpongeCamp\image-20230825091852879.png)

#### 优化

![image-20230825092050810](D:\term4\活动\SpongeCamp\image-20230825092050810.png)

#### 抽样

![image-20230825092131722](D:\term4\活动\SpongeCamp\image-20230825092131722.png)

### 基本范式的困境

- 能量函数或分布函数困境
- 算法困境

![image-20230825092257341](D:\term4\活动\SpongeCamp\image-20230825092257341.png)

### 克服模型精度不足

- 数据驱动
- AI

![image-20230825092601292](D:\term4\活动\SpongeCamp\image-20230825092601292.png)

### 神经网络

![image-20230825092724953](D:\term4\活动\SpongeCamp\image-20230825092724953.png)

![image-20230825092815534](D:\term4\活动\SpongeCamp\image-20230825092815534.png)

### 基于能量优化的蛋白质设计

![image-20230825092843789](D:\term4\活动\SpongeCamp\image-20230825092843789.png)

大分子能量模型类型

![image-20230825092920990](D:\term4\活动\SpongeCamp\image-20230825092920990.png)

![image-20230825092939107](D:\term4\活动\SpongeCamp\image-20230825092939107.png)

### 统计能量函数

![image-20230825093004175](D:\term4\活动\SpongeCamp\image-20230825093004175.png)

#### 传统解决方法

![image-20230825093108133](D:\term4\活动\SpongeCamp\image-20230825093108133.png)

#### 改进方法

- 引入更高维的边缘分布

克服技术困难（维度灾难问题）

避免过拟合



SCUBA+ABACUS2 de novo

扩散生成采样好



## 检索增强的大语言模型



## 蛋白质属性建模以及突变预测

### dl应用于蛋白质

Bert encoder

GPT decoder

注意力机制

#### 蛋白质数据库

PDB，Uniprot



## 细胞调控图谱的计算解析

