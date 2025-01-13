# NSDI22 - MLaaS

> MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters

随着机器学习 (ML)技术的持续进步和最近大量数据集的可用性，科技公司正在部署大型ML即服务 (MLaaS)云，通常使用异构GPU来提供大量ML应用程序。然而，在异构GPU集群中运行不同的ML工作负载会带来许多挑战。

![](../../../assets/posts/projects/AICompiler/nsdi22-MLaaS-fig1.png)

在本文中，作者对从阿里巴巴拥有6000多个GPU的生产MLaaS集群中收集的为期两个月的工作负载跟踪进行了表征研究，解释了集群调度面临的挑战，包括Low Utilization Caused by Fractional GPU Uses、Long Queuenig Delays for Short-Running Task Instances、Hard to Schedule High-GPU Tasks、Load Imbalance、Bottleneck on CPUs。

## 1. Workload Characterization

### 1.1 Temporal Pattern

负载时间分析如下图所示：

![Error](../../../assets/posts/projects/AICompiler/nsdi22-MLaaS-fig3.png)

运行时长和调度延迟如下图所示：
- 运行时长从秒级到天级 (Philly)(a)

- 短任务的调度延迟占据生命周期的相当比重 (b)；

- 申请整卡和高端卡的任务等待时间更长 (c, d)。

![Error](../../../assets/posts/projects/AICompiler/nsdi22-MLaaS-fig4.png)

### 1.2 Spatial Pattern

![Error](../../../assets/posts/projects/AICompiler/nsdi22-MLaaS-fig5.png)

- 资源申请：CPU/GPU/Mem均存在长尾分布现象 (实线)；

- 资源使用：不均衡，超过90%的业务GPU算力用量少于半张卡 (虚线)；

- 资源申请和使用量存在Gap -- 提升效率的空间。

## 2. GPU Machine Utilization

### 2.1 Utilization of Compute Resources

![](../../../assets/posts/projects/AICompiler/nsdi22-MLaaS-fig2.png)

- 8卡机器的CPU利用率显著高于2卡机器 (中位和长尾)；

- GPU利用率的中位 (P50)和长尾 (P90)存在较大差距；

- 主存和显存水位不高。

### 2.2 Low Usage of Network and I/O

![](../../../assets/posts/projects/AICompiler/nsdi22-MLaaS-fig6.png)

实验显示，IO等待目前还没有在集群中成为瓶颈。

## 3. Oppotunities for Cluster Management

### 3.1 GPU Sharing

GPU独享导致GPU利用率很低。为了避免这一问题，PAI支持GPU共享。

![](../../../assets/posts/projects/AICompiler/nsdi22-MLaaS-fig7.png)


### 3.2 Predictable Duration for Recurring Tasks

- 重复提交作业是常见的 (65%的作业重复超过5次)。
    - 基于作业元信息做特征工程，形成标识符。

![](../../../assets/posts/projects/AICompiler/nsdi22-MLaaS-fig8.png)

## 4. Challenges of Scheduling

### 4.1 Deployed Scheduling Policies

1. Reserving-and-Packing：
    - 即：将对GPU需求较高的应用放在高级GPU设备上，将需求较低的任务打包放到低级设备上。
2. Load-Balancing：
    - 即：在上述基础上，优先将应用放在分配率低的机器上。

![](../../../assets/posts/projects/AICompiler/nsdi22-MLaaS-fig9.png)

### 4.2 Open Challenges

1. Mismatch between machine specs and instance requests；
2. Overcrowded weak-GPU machines；
3. Imbalanced load with high-end machines；
4. CPU can be the bottleneck。