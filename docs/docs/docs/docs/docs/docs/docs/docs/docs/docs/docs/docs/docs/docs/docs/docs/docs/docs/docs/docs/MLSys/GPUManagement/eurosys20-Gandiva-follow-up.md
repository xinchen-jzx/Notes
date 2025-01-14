---
title: eurosys20-Gandiva_follow_up
date: 2024-12-23
tags: GPU Cluster Management
---

> Balancing Efficiency and Fairness in Heterogeneous GPU Clusters for Deep Learning

本文是Gandiva的后续工作，发表于EuroSys20。类似于HiveD，本文认为，在共享集群中，Deep Learning Training job的performance不能比在使用同样资源的私有集群上差；如果用户有不能充分利用的资源，没有被使用的那一部分资源要被分享给其他用户以提升集群的效率。集群中有不同性能的资源，为不同的job分配不同性能的设备是不公平的。然而已有的一部分工作即使在同构集群中也不能保证公平性，这些工作往往注重与减少JCT (Job Completion Time)或者提升集群整体效率。本文提出了一个通过把未充分利用的计算资源分配给其他活跃用户来保证集群中GPU时间公平性的调度器Gandiva_fair。Gandiva_fair用"ticket"表示公平性。Gandiva_fair通过三点来同时保证公平性和效率：（1）通过中心调度器调度大型任务，每个server上的**split scheduler**负责调度该服务器上的小任务；（2）Gandiva_fair中的**load balancer**通过job迁移来把任务均匀地分布在集群中；（3）通过**自动资源交易**的方法解决异构性带来的效率问题。

## 设计

### Split Stride Gang-Scheduling

#### 小任务 (单机任务)

每个job都有一个pass值，每运行一个时间单位就运行当前pass值最小的job，pass值累加这个job的ticket数的倒数。如果某次调度的时候，剩余的GPU数量少于需要被调度的job所需GPU数量，那就先去调度下一个job，等到下一轮的时候，这个job就成为了pass最小的，它一定会被最先分配到GPU。

#### 大任务 (多机任务)

对于多机任务来说，需要一次性分配好所有的GPU；对于小任务来说，每个server上单独调度小任务效率更高。因此，Gandiva_fair采用Split Stride scheduler。其中，每个server上的per-server scheduler用于调度单机任务，central scheduler用于调度多机任务和每个server上所有小任务组成的"集成任务"。

![Big Job](../../../assets/posts/projects/AICompiler/eurosys20-Gandiva_follow_up-fig1.png)

### Load Balance

fair的定义：inter-user fairness，根据每个user的ticket的数量占总的ticket数量的比例来分配GPU资源。local per-server scheduler可以根据ticket的比例调度每个小job的资源。在此基础上，如果能保证center scheduler对每个local per-server scheduler的资源调度是公平的，就能保证整体的公平性了。这只需每个server上的ticket数量尽量平均 (每个node上的ticketLoadPerGPU尽量相等, load balance）。

在有资源空闲，但与此同时也有等待中的任务的时候，Gandiva_fair会（1）将部分任务迁移到其他的server上；（2）考虑把任务"pack"起来 (参考Gandiva，pack不影响前面对ticketLoadPerGPU的计算)。

### 处理异构性问题

仅针对调参的时候，一个用户提交大量相似的任务的情景。通过同一用户的不同job在不同类型的GPU上运行，记录效率，从而决定怎样为这个任务分配GPU是最优的。对于不同的用户，最开始给他们分配的GPU型号是相对公平的。但是因为他们的任务不同，Gandiva_fair可能会根据job的特点，在他们之间进行GPU的交易，从而使得每个用户的运行效率都有所提升。

<img alt="Algorithm 2" align="center" src="assets/posts/projects/AICompiler/eurosys20-Gandiva_follow_up-fig2.png">

## 总结

本文主要解决了两件事：
1. 通过调度算法和负载均衡来保证不同用户之间的公平性；
2. 充分利用异构集群的特点来提高集群的效率。