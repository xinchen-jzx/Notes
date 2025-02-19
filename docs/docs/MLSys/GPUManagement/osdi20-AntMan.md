# OSDI20 - AntMan

> AntMan: Dynamic Scaling on GPU Clusters for Deep Learning

这一新工作通过对调度器和训练框架的联合设计，减少了同一个物理GPU上多个任务间的相互干扰，从某种程度上实现了对GPU资源的超卖，提高了GPU资源的利用率。

## 1. 动机

随着深度学习在工业界的落地，在云上进行模型训练和推理的需求越来越旺盛。但是，GPU这样的硬件设备的资源利用率，一直以来都处于相对比较低的水平。一方面因为模型训练往往涉及许多不同的环节，比如数据的预处理等。有些环节不适合在GPU上执行。另一方面，随着数据规模的扩大和模型复杂度的提升，分布式训练逐渐成为工业界训练场景的主流选择。同步SGD的分布式训练中，很多的时间花费在了等待网络IO上，GPU从微观的角度来看，经常空闲。

![](../../../assets/posts/projects/AICompiler/osdi20-AntMan-fig1.png)

注意：在分布式训练中，通常需要任务下所有的实例都被运行时，训练才能开始。因此需要在调度上进行gang scheduling的支持。而gang scheduling会造成GPU的等待。任务所需要的GPU越多，在任务执行前GPU空闲的时间越长。这也很容易理解：调度器需要先把空闲的GPU hold住一段时间，在确定分布式训练中所有的PS和Worker都可以被运行，才会真正创建所有实例。GPU被调度器Hold的这段时间没有办法被其他任务使用，所以会造成一定的浪费。

多个任务共享一个GPU可以提高利用率，但是多个任务会在GPU上 (如Memory hierarchy等)有相互的干扰。因此，在生产环境中，通常不会采取这样的方式。目前也有一些GPU虚拟化的解决方案 (如 Amazon Elastic Inference)，但是在容器环境下，也很少有可以落地生产环境的。

除此之外，GPU在训练过程中，SM和显存也存在一定程度的不均衡。在左图中，DeepFM的模型训练通常需要进行数据的预处理，这个过程只需要CPU参与，因此GPU利用率为0%。右图也是有类似的情况。

![](../../../assets/posts/projects/AICompiler/osdi20-AntMan-fig2.png)

## 2. Key Insight

文章通过实验，论述了Key Insight：大部分模型本身占用的显存并不多，使用的显存多来自mini-batch过程中，在单个mini-batch中会被申请和释放。文章中所有的design基本都是围绕这一Key Insight展开的。

## 3. 系统设计

文章联合设计了调度器和框架，让框架在训练任务的角度来支持显存和算力的动态调整，然后让调度器从集群的角度利用这一新的特性进行更有针对性的调度。

### 3.1 框架层的设计

#### 3.1.1 显存方面

![](../../../assets/posts/projects/AICompiler/osdi20-AntMan-fig7.png)

现有深度学习框架中，缓存的GPU内存大小 (红色虚线)的总大小随着深度框架中创建的张量而增加 (a∼b)。一般来说，有些张量只在深度学习训练的某些阶段使用，在其他情况下这些张量是不被需要的。但是这部分缓存的GPU内存没有释放。

AntMan扩展GPU内存上限。它会主动检测已用内存，以收缩缓存内存，从而内省地调整GPU内存使用量，使之适合。这是通过在处理mini batch时监控应用程序性能和内存需求来实现的。AntMan尽可能在GPU设备上分配张量，但是，如果GPU内存仍然不足，则可以在GPU之外使用主存分配张量 (d~e)。有了这样的通用内存支持，作业甚至可以继续处理低于其实际GPU内存需求的进程。当GPU内存的上限增加时，张量可以自动分配回GPU (f)。

![](../../../assets/posts/projects/AICompiler/osdi20-AntMan-fig3.png)

#### 3.1.2 计算方面

在计算方面，如果多个任务运行在同一个GPU上时，主要会带来GPU Kernel的排队延迟，和PCIe总线带宽的争抢。以下图（a）和（b）来说，B任务影响了A任务原本的执行，为了解决这个问题，Antman在框架层引入了GPU Op Manager。在原本的设计中，一旦Kernel的控制依赖被满足了，就会被执行。GPU Op Manager接管了原本的流程，它会控制GPU Kernel被issue的频率。因为GPU Op Manager只会管控 GPU Kernel，因此CPU的Op可以照常被执行，不会被Block。如下图（c）所示，满足了依赖的CPU Op可以在GPU Op没有被执行的时候照常执行，不会受到GPU Op Manager的影响。

![](../../../assets/posts/projects/AICompiler/osdi20-AntMan-fig4.png)

### 3.2 调度器的设计

在调度器的设计上，Antman并没有采取Monolithic的架构，而是存在两个角色：Global Scheduler和Local Coordinator。其中全局的调度器负责进行任务的调度，而Local Coordinator主要负责根据深度学习的训练任务的执行情况 (任务情况，硬件指标，mini batch的执行时间，显存和内存的使用情况等)管理训练任务的全生命周期。

![](../../../assets/posts/projects/AICompiler/osdi20-AntMan-fig5.png)

Antman根据SLA把任务分为resource-guarantee和opportunistic两种任务，其中前者需要保证与独占GPU卡类似的训练速度。Antman的设计目标是在保证resource-guarantee类型任务的SLA的同时，提高集群的利用率。opportunistic类型的任务主要就是用来提高集群利用率的。

全局调度器的调度算法比较简单，如下图所示。首先调度器会根据拓扑，获得一个最优的节点组合。这一步与业界主流基本一致，尽可能考虑到NVLink等硬件资源的拓扑结构，进行分配。如果是resource-guarantee的任务，有合适的节点就会直接运行。对于opportunistic类型的任务，Antman会找到负载最低的节点去执行。

![](../../../assets/posts/projects/AICompiler/osdi20-AntMan-fig6.png)

Local Coordinator最主要的职责是管理在共享的GPU上训练任务。在Antman中，一个GPU只会被分配给一个resource-guarantee的任务。所以当有opportunistic的任务已经在GPU上运行时，为了满足新来的resource-guarantee任务，Local Coordinator会限制opportunistic任务使用的SM和显存。随后慢慢提高opportunistic的限制，确保在不影响resource-guarantee任务的训练速度 (mini batch的耗时)的同时，提高opportunistic的资源限额。

## 4. 总结

阿里的这一工作通过对框架和调度器的联合设计，在某种程度上支持了任务的SLA级别，以及GPU资源在显存和算力级别的共享。可以看出，在这一方面的工作从调度器单方面的优化，逐渐走向co-design的方向。GPU的虚拟化和复用也逐渐走向落地。