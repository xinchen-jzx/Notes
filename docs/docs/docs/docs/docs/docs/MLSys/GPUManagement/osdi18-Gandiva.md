# OSDI18 - Gandiva

> Ganddiva: Introspective Cluster Scheduling for Deep Learning

## Overview

Gandiva：一个集群调度框架，使用特定领域知识，优化了GPU集群训练深度学习模型的延迟与效率。

## Motivation

首先，深度学习的训练任务在运行时往往有以下的三个特征：
- **Feedback-Driven Exploration**：任务在训练过程中往往需要指定超参数，例如学习率、优化器、迭代次数等。通过训练过程中模型的表现，我们会提前终止不恰当的超参数组合，从而减少不必要的资源占用。

- **Heterogeneity**：这个特征主要是针对于资源的使用情况而言。训练任务在运行过程中，例如前向传播、反向求导、梯度更新等阶段对GPU资源 (GPU的算力、显存等等)的需求是不同的。batch与batch之间GPU资源往往是空闲的。资源使用情况在任务进行的全程是不一致的，但这没有被充分的挖掘和利用。

- **Intra-Job Predictability**：训练任务往往是重复的迭代，每个迭代总是进行相同的任务。因此每个batch的运行时间在不受外界干扰 (例如资源抢占)时往往相近，资源的使用呈现周期性的变化。这篇论文充分利用了这个特征，从而对资源分配与任务调度提出application aware time-slicing以及profile-driven introspection优化方法。

![Predictability](../../../assets/posts/projects/AICompiler/osdi18-Gandiva-fig3.png)

此外，论文还着重分析了两点外界因素对DLT训练任务的影响。这两点也是后续设计的考虑因素：
- **Sensitivity to Locality**：这一点主要考虑了GPU彼此之间的亲和性对训练任务的影响。两块GPU可能分布在 (1) DiffSocket：不同的CPU Socket；(2) SameSocket：相同的CPU socket，但是不同的PCIe switch；(3) SamePCIeSw：相同的PCIe switch 这三种情况。对于相同的训练任务，GPU彼此之间的亲和性不同对训练效率会产生较大的影响。

- **Sensitivity to Interference**：这一点主要考虑任务之间的相互影响。作者发现，共享于相同GPU节点上的模型之间往往会对彼此的任务产生消极的影响，但是受到影响的程度却往往不同。但是，对于模型受到影响的程度和模型具体有什么样的关系文中没有详细的分析。

![Sensitivity](../../../assets/posts/projects/AICompiler/osdi18-Gandiva-fig2.png)

## Design

![Design](../../../assets/posts/projects/AICompiler/osdi18-Gandiva-fig4.png)

### Implementation

![Implementation](../../../assets/posts/projects/AICompiler/osdi18-Gandiva-fig1.png)

整个框架的实现借助于目前炙手可热的容器编排系统 -- Kubernetes。上图右侧Kubernetes Master是系统的控制平面，而Kubernetes Node则是真正进行计算的节点。论文的改进指出在于图上的绿色方框内：左侧实现了一个资源调度器，右侧实现了一个与调度器进行交互的客户端。调度器从宏观的角度对用户的训练任务进行资源分配，而客户端则是真正负责任务的开始、结束、暂停、重启等操作。

此外，为了和调度器兼容使用，作者还对当前流行的深度学习计算框架进行了代码修改，在阅读论文我总结主要作用包括以下两点：(1) 便于模型迁移 (migration)；(2) 衡量当前模型训练性能 (profiling)。

### Mechanism

论文中，作者介绍了4种不同的设计机制，依次为Suspend-Resume and Packing、Migration、Grow-and-Shrink以及Profiling：
1. **Profiling**：在这里，Profiling的作用是衡量在某种资源配置下，GPU的利用率以及训练任务的表现。传统的资源调度框架在衡量该项指标时，往往从GPU的利用率的角度入手，即考察GPU的内存 (memory)和计算资源 (core)的利用率。Gandiva则更加考虑了深度学习的训练任务的特质 -- intra-predictability。即它通过比对每个迭代过程所花费的时间来判断该中资源分配是否合理。
2. **Grow-and-Shrink**：这里是指对于某一任务分配的GPU的数量。我们知道，对于大部分的任务而言，GPU的数量越多，那么模型训练的时间越短。因此在集群运行的任务较少时，分配给某一任务的GPU数量则会增多 (Grow)；反之则会减少 (Shrink)。
3. **Migration**：已知模型训练的性能会受到GPU的亲和性，以及模型之间的相互作用的影响。Migration则是为了解决这一问题。当发现训练的性能受到了如上因素的影响时，调度器则会选择将某一任务分配到其他的GPU上。Migration的实现方式有两种，一是通过CRIU (作用是冻结用户程序)，二是通过深度学习框架提供的保存与恢复机制，例如`tensorflow.train.saver`的使用。作者在使用第二种方法进行实现时，引入了warm up的机制 (提前预热)从而减小Migration带来的开销。
4. **Suspend-and-Resume and Packing**：这是Gandiva框架中最重要的一个设计原则，解决的是如何使多任务共享GPU。这一机制包含了两种解决方法: (1) Suspend and Resume；(2) Packing.
    - Suspend-and-Resume类似于在CPU上进程的调度问题，当然这里还引入了GPU的额外的特点。Suspend的作用是将某一任务暂时挂起。当调度器试图将任务挂起时，共有以下4步：(1) 等待一段时间，直至该任务的显存使用达到最低；(2) 将显存上的数据转移到内存中；(3) 释放显存资源；(4) 将任务挂起。Resume则是恢复某一任务的执行，主要有3步：(1) 分配显存资源；(2) 将内存中的数据转移到显存中；(3) 启动该任务。Suspend-and-Resume这个方法的一个好处在于当多个任务需要的显存总和高于GPU的总和时，任务依旧能够执行，而不是像传统的方法中，某一任务一旦获取资源，直至任务运行结束才会释放资源。
    
    - Packing是让不同的任务同时在GPU上运行。文中没有对这种方法的实现提供更为详细的介绍，个人猜测应该是采取类似于Nvidia MPS中上下文共享的方法。因为不同的进程共享了GPU上下文，因此与上一种方法对比则减少了上下文切换的开销。当然，这种方法的局限性在于(1) 任务之间不能产生消极的相互影响；(2) 任务需要的显存总量不能超过GPU能够提供的最大值。
    
    - 论文中在 Packing 算法的实现采用了贪心算法的思想。首先，所有的任务都采用Suspend-and-Resume的方法运行一段时间并进行 Profiling。然后将显存使用用最小的进程进行Packing。如果Packing之后的结果比Packing之前的好的话，那么重复上述步骤; 否则，取消Packing, 重新选择任务Packing。

## Summary

总体来说，论文的出发点是Intra-job predictability，即考虑深度学习任务在训练过程中周期性的特点。在这基础上，设计了4条原则来指导GPU资源在深度学习训练任务中的分配问题。