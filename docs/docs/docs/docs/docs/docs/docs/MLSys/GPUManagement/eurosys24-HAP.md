# EuroSys24 - HAP

> HAP: SPMD DNN Training on Heterogeneous GPU Clusters with Automated Program Synthesis

## 1. 摘要

HAP是一个自动化系统，旨在加速在异构集群上的SPMD (Single-Program-Multiple-Data)类型的模型训练。通过优化张量分片策略、异构设备间的分片比例和张量通信方法，实现分布式训练的优化。HAP将模型分割问题表述为自动化程序合成问题 (Automated Program Synthesis)，通过A*搜索算法在分布式指令集上生成分布式程序，同时解决最优张量分片比例的问题，进而按照SPMD计算范式并行执行。

## 2. 问题挑战

解决在异构集群上训练大模型时如何有效利用不同GPU设备及网络连接等资源。

## 3. 详细解决方案

### 3.1 执行流程

![](../../../assets/posts/projects/AICompiler/eurosys24-HAP-fig1.png)

1. 输入数据包括：
    - 单设备程序；

    - 集群参数明细：包括设备如GPU的FLOPS、通信原语的延迟和带宽等。
2. 将上述数据输入到CPU上的程序合成器和负载均衡器，以确定最优的分布式程序Q和分片比例B；
    - Distributed Program Q：根据单设备上的原始程序自动程序合成的，目的是按照SPMD计算范式在多设备上实现并行处理，同时保持与原始单设备程序相同的语义。这种方法特别适用于那些结构相对简单、允许进行有效语义分析的程序。
        - 在大模型训练中，Tensor程序因为非递归且无副作用刚好满足条件。
    
    - Sharding Ratios B：在分布式训练中如何在不同GPU之间分配Tensor的Shards。大型的张量可能无法在单个设备上完全存储，因此需要将这些张量分割成更小的分片，并分布到多个设备上。B就是用来描述这个分布的比例和大小。
        - 有两种分片比例方法：CP (按照计算能力设置分片比例)和EV (均匀分片)。
    
### 3.2 自动化程序合成

HAP将模型分割问题转化为程序合成问题，利用分布式指令集从头开始生成分布式程序，该程序在语义上类似于为单设备设计的程序，即实现SPMD计算范式。

![](../../../assets/posts/projects/AICompiler/eurosys24-HAP-fig2.png)

![](../../../assets/posts/projects/AICompiler/eurosys24-HAP-fig3.png)

### 3.3 负载均衡

利用负载均衡分配计算通信任务，确保所有设备高效工作。
- 将负载均衡问题转换为线性规划问题：目标是最小化整个训练过程的迭代时间，同时考虑到不同设备的计算速度和内存容量；

- 模型分割：在更复杂的情况下，DNN模型可能会包含许多层，每层的计算和通信比例可能不同。论文中提出将模型分割成多个段 (Segments)，并为每个段独立确定分片比例；

- All-to-All通信：在模型分割的基础上，为了协调不同段之间的张量分片比例，HAP在段的边界插入All-to-All通信操作和同步点，以确保数据在设备之间的正确分配；

- 优化算法：论文中使用了迭代优化方法来交替优化分布式程序和分片比例。通过这种方式，系统可以逐步逼近最优的负载均衡解决方案。

### 3.4 A*搜索算法

HAP使用A*搜索算法来寻找最优的分布式程序，该算法结合了动态规划的思想，通过循环搜索和成本预估来加速搜索过程。其中成本预估指的是HAP提出的一种基于仿真的方法，将分布式程序的执行分成多个段，并计算每个阶段的通信和计算时间。

![](../../../assets/posts/projects/AICompiler/eurosys24-HAP-fig4.png)

### 3.5 通信优化

- 异构集群中的All-Gather实现
    - 在异构集群中，根据分片比例的不同可能需要不同的All-Gather实现方式；
    
    - 论文中提到了两种方法：填充 (Padded All-Gather)和分组广播 (Grouped Broadcast)，并在搜索过程中自动选择性能更好的方法。

![](../../../assets/posts/projects/AICompiler/eurosys24-HAP-fig5.png)

- Sufficient Factor Broadcasting (SFB)：SFB利用梯度张量的低秩结构来减少通信量。
    - 在搜索过程中，通过添加特定的规则来探索SFB的应用以优化通信。

![](../../../assets/posts/projects/AICompiler/eurosys24-HAP-fig6.png)

### 3.6 搜索时间的优化

- 融合Hoare三元组：将具有空前提条件的Hoare三元组与其消费者融合，减少搜索过程中对这些指令位置的枚举；

- 避免重复通信：不允许对同一参考张量进行多次通信，减少不必要的通信指令；

- 移除冗余属性：从部分程序中移除不会用于任何指令前提条件的属性，增加可以剪枝的程序数量；

- 启发式函数优化：使用最小化执行时间作为启发式函数，确保不会高估未来成本。

## 4. 实验效果

![](../../../assets/posts/projects/AICompiler/eurosys24-HAP-fig7.png)
