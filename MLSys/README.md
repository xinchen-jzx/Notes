# Machine Learning System

https://zhuanlan.zhihu.com/p/104444471

## 1. 分布式机器学习 (Distributed DNN Training)

这个又可以分为两个方面: from ML/system perspective. 安利一下刘铁岩老师的《分布式机器学习》这本书, 还有UCB CS294 19fall的一节

### 1.1 ML

从ML的角度做, 主要是发明或改进分布式训练算法[ch4] [ch5], 保证在分布式加速的同时, 仍然能达到原来的学习效果 (loss/accuracy). 因此很多工作被投在像ICML、NIPS这种专业ML会议上. 主要用到的方法包括优化 (optimization)和统计学习理论 (statistical learning theory). 还有一类工作涉及到如何把单机算法改造成分布式[ch9], 比如同步/异步SGD等. 这里主要涉及到的问题是如何降低分布式环境下的通信开销, 提高加速比

### 1.2 System

还有一个就是从System的角度做. 从分布式计算的角度来看, 可以把相关工作分为以下几类: 
1. 对于计算量太大的场景 (计算并行), 可以多线程/多节点并行计算, 多节点共享公共的存储空间. 常用的一个算法就是同步随机梯度下降 (synchronous stochastic gradient descent), 含义大致相当于K个 (K是节点数)mini-batch SGD [ch6.2]
2. 对于训练数据太多, 单机放不下的场景 (数据并行, 也是最主要的场景), 需要将数据划分到多个节点上训练. 每个节点先用本地的数据先训练出一个子模型, 同时和其他节点保持通信 (比如更新参数)以保证最终可以有效整合来自各个节点的训练结果, 并得到全局的ML模型 [ch6.3]
3. 对于模型太大的场景, 需要把模型 (例如NN中的不同层)划分到不同节点上进行训练. 此时不同节点之间可能需要频繁的sync。这个叫做模型并行 [ch6.4]
4. Pipeline Parallelism: 这是去年 (SOSP19 PipeDream)才出现的概念. Pipeline Parallelism相当于把数据并行和模型并行结合起来, 把数据划分成多个chunk, 也把训练模型的过程分成了Forward Pass和Backward Pass两个stage. 然后用流水线的思想进行计算

另外, 分布式ML本质上还是分布式系统嘛, 所以像传统分布式系统里的一些topic (比如一致性、fault tolerance、通信、load balance等等)也可以放到这个背景下进行研究

最近挖的比较多的坑大致涉及以下几个点: 

#### 1.2.1 分布式ML系统设计

[ch7.3] 最著名的就是几大分布式DL模型：Parameter Server / AllReduce等。

个人感觉这里面一个可以挖的坑是Decentralized Training