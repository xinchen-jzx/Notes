---
title: The Deep Learning Compiler - A Comprehensive Survey 
date: 2024-12-12
tags: AICompiler Survey
---

## 1. High-level IR

Graph-IR的表示方法: 
- **DAG-based IR** (有向无环图): 结点表示算子, 边表示tensor
    - 优点: 可以快速分析数据依赖, 方便优化
    - 缺点: 图上的结点、边语义不明确
- **Let-binding-based IR**: 对每一个变量使用`let`建立节点, 语义明确
    - 它对每一个`let`计算结果, 建立一个map, 每个表示都通过查表来得到结果

> TVM的Relay IR借鉴了上述两种方法

tensor计算的表示方法: 
- **Function-based**: Glow, nGraph以及XLA采用该表示, 其提供封装好的算子. tensor的计算视为一种函数
- **Lambda expression**: TVM采用该表示, 其使用lambda函数表达式, 不需要声明新函数
- **Einstein notation**: 比lambda表达式更简单, 但算子需要满足结合和交换性质, 其方便并行化

Graph-IR的实现: 
- **Placeholder**: 具有关于tensor shape信息的变量
- **Unknown (Dynamic) shape representaton**: 允许某一维的大小未知
    - TVM使用`Any`来表示, 如`Tensor<(Any, 3), fp32>`
    - XLA使用`None`来表示, 如`placeholder("float", [None, 3])`
    - nGraph定义了`PartialShape`类
- **Data layout**: 逻辑地址到内存分片的映射
    - NCHW与NHWC: 
        - 比如卷积神经网络的特征图 (Feature Map)通常用四维数组保存, 即4D: 
            - N: Batch数量, 例如图像的数目
            - H: Height, 特征图高度, 即垂直高度方向的像素个数
            - W: Width, 特征图宽度, 即水平宽度方向的像素个数
            - C: Channels, 特征图通道, 例如彩色RGB图像的Channel为3
        - NCHW: 同一个通道的数据值连续排布, 更适合需要对每个通道单独运算的操作
            - 其计算时需要的存储更多, 适合GPU运算, 利用GPU内存带宽较大并且并行性强的特点, 其访存和计算的控制逻辑相对简单
        - NHWC: 其不同通道中的同一位置元素顺序存储, 更适合那些需要对不同通道的统一数据做某种操作
            - 更适合多核CPU运算, CPU内存带宽相对较小, 每个数据计算的时延较低, 临时空间也很小
    - TVM和Glow将Data layout表示为算子参数, 需要这些信息来进行计算和优化
    - XLA将其表示为后端硬件的约束
    - Relay和MLIR将其添加到tensor的类型系统中
- **Bound inference**: 推断迭代器的上下界
    - TVM中iterator建立一个DAG图, 点表示迭代器, 边表示运算, 根节点的shape of placeholders被确定后就可以递归进行处理

Graph-IR需要支持多个算子: **algebraic operators** (+, exp, topK等等), **neural network operators** (convolution, pooling等等), **broadcast and reduction operators** (min, argmin等等)以及**control flow operators** (conditional, loop等等)

## 2. Low-level IR

- **Halide-based IR**: 其设计哲学是计算和调度分离
    - TVM改进了它
- **Polyhedral-based IR**: loop的大小更加灵活
- **Other unique IR**
    - Glow: 包含`declare`和`program`两种操作, 用`@in, @out, @inout`帮助分析内存优化的时机
    - MLIR: 使用Dialect提供对IR的抽象. 自定义Dialect方便开发者适配新硬件
    - HLO IR of XLA: 同时是high/low-level IR, 足够细粒度

## 3. Frontend Optimizations

### 3.1 Node-level optimizations

- **Nop Elimination**: 消除不必要的节点
    - 例如: 只有一个节点的sum求和算子等
- **Zero-dim-tensor Elimination**: 消除0维向量, 或者消除某个维度为0的向量

### 3.2 Block-level optimizations

- **Algebraic simplification**
    - optimization of computation order
        - 例如: $A^TB^T=(BA)^T$
    - optimization of node combination
        - 例如: 将多个transpose消成一个
    - optimization of ReduceMean nodes
- **Operator fusion**
- **Operator sinking**

### 3.3 Dataflow-level optimizations

- **Common sub-expression elimination (CSE)**
- **Dead code elimination (DCE)**
- **Static memory planning**
    - in-place memory sharing: 复用输入和输出的memory
    - standard memory sharing: 在不覆盖的情况下, 复用前面用到的内存
- **Layout transformation**
    - 计算出tensor最好的存储方式, 然后在计算图中增加layout transformation节点

## 4. Backend Optimizations

目前算子分为如下两类: 
- **访存密集型**: 如RNN训练任务, 由于RNN网络模型结构的计算密度更低, 瓶颈转移到host端的Op Launch上, 因此kernel之间甚至出现了大量空白
- **计算密集型**: 计算密度是指一个程序在单位访存量下所需的计算量, 单位是$\text{FLOPs/Byte}$

NOTE: 对于一个算子, 要实现正确的逻辑计算简单, 但是要结合硬件能力达到高性能就比较难
- 目前业界一个最为常见的方式就是**将预置的算子实现封装成计算库**
    - 如何面对AI领域算子迭代更新快?
    - 如何解决同一算子在多平台移植后一致性问题? 
    - 如何面对算子组合爆炸问题? 如参数多样、融合大算子等
- 另一种方法: **自动化生成**
    - 相关研究: 
        - A Survey on Compiler Autotuning using Machine Learning
        - Machine Learning in Compiler Optimization
        - Machine Learning in Compilers: Past, Present and Future
    - Auto Tuning: 尝试融合机器学习方法
    - Polyhedral: 编译技术

### 4.1 Hardware-specific Optimization

![Hardware-Specific Optimizations](assets/hardware-specific.png)

### 4.2 Auto-tuning