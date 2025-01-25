# OSDI20 - HiveD

> HiveD: Sharing a GPU Cluster for Deep Learning with Guarantees

随着深度学习训练需求的不断扩张，很多组织和机构都会选择自建多租户集群来共享GPU资源。然而现有的GPU集群管理方案因为使用GPU配额 (Quota)机制可能导致严重的共享异常 (Sharing Anomaly)现象，即某些租户的深度学习任务甚至比在私有集群上更差。为了从根本上解决问题，该论文提出一个新的多租户GPU集群管理方案HiveD，通过新的资源抽象和调度框架从而100%保证共享安全 (Sharing Safety)，同时不失一般性地和任何任务调度策略兼容。

HiveD首先提出了一种新的资源抽象，被称为Cell (类比蜂巢Hive中的蜂窝)。HiveD通过多级Cell来描述GPU集群中拓扑的层级组织。在Cell的基础上，HiveD会为每个租户提供一个虚拟私有集群 (Virtual Private Cluster，简称VC)。每个租户的VC显式定义了它拥有的各级Cell的配额。为了在保证共享安全的同时，又能灵活地分配GPU资源，论文为HiveD设计了一个Buddy Cell Allocation算法，来把VC中的逻辑Cell和物理GPU进行动态绑定。该论文在理论上证明了该算法能100%提供共享安全。此外该算法还可以降低物理集群的GPU碎片化、更灵活地处理设备故障、支持动态重配置VC。

## 背景和动机

**多租户集群 (Multi-Tenant Cluster)**：HiveD的目标场景是一个集群里有多个tenants，每个tenant会贡献一定的资源，这部分贡献的资源是所谓的私有集群 (Private Cluster)。要保证这些tenants愿意去共享资源的前提就是，其在共享集群中的性能应该不差于单独拥有一个私有集群时的性能，否则tenant会更愿意去独占自己的资源。这个概念在调度领域也叫**Sharing Incentive**。

**传统的对多租户GPU集群的管理方法**：用户申请一定数量的GPU Quato。为了提高训练速度，用户通常对一个深度学习任务有Affinity Requirement，即对资源分布的要求 (例如一个需要64张卡的任务需要跑在8台机器各8张卡上)。在多租户集群中没有满足要求的资源时，资源管理器可以选择排队等待或者使用更宽松的Affinity Requirement。

**共享异常 (Sharing Anomaly)**：和内存管理中的外部碎片类似。有时候多租户集群中有足够的资源数量，但是这些资源"碎片化"，无法满足用户的Affinity要求，这样的现象被称为"共享异常"。

<img alt="Sharing Anomaly" align="center" src="./assets/osdi20-HiveD-fig1.png">

## 系统设计

<img alt="System Design" align="center" src="./assets/osdi20-HiveD-fig2.png">

HiveD采用了两层结构：第一层结构是Virtual Private Cluster虚拟私有集群，这一层为用户提供了一个私有集群的"假象"；第二层是实际的集群，从虚拟私有集群 (VC)到实际的集群中的设备之间存在着映射关系，如上图所示。

- **VC**：
    - 提出了一种新的对资源抽象的方式 -- Cell，使用这种资源抽象可以同时描述GPU资源的Quota和Affinity；
    - 在VC层面，可以使用现有的深度学习调度器来对资源进行调度。
- **From Virtual to Physical**：
    - 提出了可以保证共享安全的Dynamic Cell Allocation算法；
    - 支持low-priority jobs。

### Virtual Private Cluster with Cells

为了对私有GPU集群进行描述，HiveD定义了一个多层Cell结构的层次结构。某一层次上的Cell是具有相应拓扑结构的GPU的对应集合。在每个虚拟私有集群 (VD)上使用每个级别上的cell数来模拟/描述相应的私有集群。

<img alt="Cells" align="center" src="./assets/osdi20-HiveD-fig3.png">

在cell层次结构中，K级的cell由一组(K - 1)级cell组成，这些(K - 1)级的cell被称为伙伴单元 (Buddy Cells)；伙伴单元可以合并为下一个更高级别的cell。本文假设cell具有层次一致的可组合性：
1. 所有K级cell在满足组合对K级cell的请求方面是等价的；
2. 所有K级cell都可以拆分成相同数量的(K - 1)级cell。

### Buddy Cell Allocation Algorithm

Buddy Cell Allocation算法非常简单直观：如果需要在VC中分配一个K级cell，算法从K级开始首先检查是否有可用的K级Cell，如果有可用的cell则分配一个；否则算法将逐级下移，去检查(K + 1)级或者更高级别的cell，直到有一个可用的l级cell，其中l > K。然后，算法将递归地将一个空闲的l级cell拆分为多个较低级别的cell，直到K级cell可用为止。每次拆分都会在下一个较低级别生成一组Buddy Cells，这些cell将被添加到相对应的空闲列表中。

如果有cell被释放，该算法会按照与分配cell相反的方式把空闲的cell尽量merge起来，这样保证空闲的cell中有尽量多的更高level的cell。

<img alt="Buddy Cell Allocation Algorithm" align="center" src="./assets/osdi20-HiveD-fig4.png">

<img alt="Theorem 1" align="center" src="./assets/osdi20-HiveD-fig5.png">

> For the case where cells are heterogeneous, HiveD partitions the cluster into several pools within which cells at the same level are homogeneous, and applies Buddy Cell Allocation Algorithm in each pool.

## 总结

HiveD做的事情就是给各个tenant一个正在使用自己私有集群的假象，同时赋予tenants使用自己VC以外的资源来从共享集群中获益的能力。