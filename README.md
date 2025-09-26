# 区块链模拟器

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/your-username/ai-blockchain/pulls)

## 概述

本仓库包含一个在通义灵码（Tongyi Lingma）协助下构建的区块链模拟器，旨在模拟典型的区块链系统和共识算法。该项目旨在提供一个教育和研究平台，以了解不同的共识机制在各种网络条件下的表现。

此项目是[Linpeng Jia](https://github.com/jialinpeng)与通义灵码（Tongyi Lingma）的协作成果，展示了人类专业知识与AI编码辅助相结合如何创建复杂的分布式系统。

## 动机

创建此仓库的主要目的是利用通义灵码的强大功能构建一个全面的区块链模拟系统。该项目展示了如何使用AI编码助手来开发复杂的分布式系统，并提供了一个灵活的平台，用于试验不同的区块链共识算法。

## 功能特点

### 支持的共识算法
- **工作量证明 (PoW)** - 比特币使用的传统共识机制
- **实用拜占庭容错 (PBFT)** - 经典的BFT共识算法
- **HotStuff** - 具有流水线能力的现代BFT共识
- **Dumbo** - 异步共识算法

### 网络协议
- **直接连接 (Direct)** - 所有节点直接相连的全网状网络拓扑
- **Gossip协议 (Gossip)** - 通过随机消息传播的部分连接网络

### 模拟功能
- 可配置的节点数量、交易速率和区块大小
- 实时性能指标收集
- 交易确认时间分析
- 吞吐量(TPS)测量与可视化
- 以JSON格式导出详细的区块和交易数据
- 不同共识算法和网络协议之间的性能比较

### 可视化
- 按区块高度划分的TPS（每秒交易数）图表
- 交易确认时间分布直方图
- 自动生成性能报告

## 快速开始

### 前提条件
- Python 3.7+
- 可选：matplotlib（用于可视化功能）

### 安装
```bash
git clone https://github.com/your-username/ai-blockchain.git
cd ai-blockchain
```

### 基本用法
```python
from blockchain_simulator import BlockchainSimulator, ConsensusType, NetworkProtocol

# 创建模拟器实例
simulator = BlockchainSimulator(
    node_count=20,
    consensus_type=ConsensusType.HOTSTUFF,
    network_protocol=NetworkProtocol.GOSSIP,
    transaction_send_rate=1000.0
)

# 运行模拟
simulator.run_simulation(
    transaction_count=10000,
    blocks_to_mine=50
)
```

### 输出
模拟器生成：
- `data/` 目录中的详细区块信息
- `results/` 目录中的性能图表
- 用于分析的统计摘要

## 项目结构
```
ai-blockchain/
├── blockchain_simulator.py    # 主要模拟器实现
├── data/                      # 模拟输出数据
├── results/                   # 性能图表和图形
└── README.md                 # 本文件
```

## 配置选项

模拟器支持各种配置参数：

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `node_count` | 网络中的节点数量 | 4 |
| `consensus_type` | 要使用的共识算法 | PoW |
| `network_protocol` | 网络通信协议 | DIRECT |
| `transaction_send_rate` | 交易发送速率 (txs/sec) | None |
| `max_transactions_per_block` | 每个区块的最大交易数 | 256 |
| `transaction_size` | 每个交易的大小 (字节) | 300 |

## 贡献

欢迎贡献！请随时提交Pull Request。无论是添加新的共识算法、改进现有实现，还是增强可视化功能，您的贡献都将帮助使这个模拟器对社区更有价值。

## 许可证

该项目采用MIT许可证 - 详情请参阅[LICENSE](LICENSE)文件。

## 作者

- **Linpeng Jia** - *初始工作* - [https://github.com/jialinpeng](https://github.com/jialinpeng)
- **通义灵码** - *AI助手* - 阿里云通义实验室的一部分

## 致谢

- 本项目是在通义灵码（Tongyi Lingma）的协助下构建的，展示了AI驱动的编码助手在开发复杂分布式系统方面的能力。
- 受到需要教育工具来理解区块链共识机制及其性能特征的启发。

# 区块链模拟器

区块链模拟器 - 一个由人类开发者 [Linpeng Jia](https://github.com/jialinpeng) 与 AI 编码助手 [通义灵码](https://tongyi.aliyun.com/) 共同协作开发的区块链共识算法模拟器。

本项目是一个教育和研究平台，用于理解不同共识机制在各种网络条件下的性能表现。

## 项目结构

项目采用模块化设计，主要包含以下模块：

- `core.py` - 核心数据结构模块，包含交易、区块和节点类
- `network.py` - 网络相关模块，包含网络协议、拓扑和传输类
- `consensus.py` - 共识算法模块，包含各种共识算法的实现
- `simulator.py` - 主模拟器模块，包含区块链模拟器主类
- `main.py` - 主程序模块，包含交互式设置和程序入口
- `blockchain_simulator.py` - 向后兼容的入口点

## 功能特点

- 支持多种共识算法：PoW、PBFT、HotStuff、Dumbo
- 支持多种网络协议：DIRECT（直接广播）、GOSSIP（Gossip协议）
- 可配置节点数量、交易速率、区块大小等参数
- 实时收集性能指标并生成可视化图表
- 导出详细的区块和交易数据（JSON格式）
- 自动生成性能报告

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 直接运行（交互式配置）

```bash
python main.py
```

或

```bash
python blockchain_simulator.py
```

### 编程方式使用

```python
from simulator import BlockchainSimulator
from core import ConsensusType, NetworkProtocol

# 创建模拟器实例
simulator = BlockchainSimulator(
    node_count=20,
    consensus_type=ConsensusType.HOTSTUFF,
    network_protocol=NetworkProtocol.GOSSIP,
    transaction_send_rate=1000.0
)

# 运行模拟
simulator.run_simulation(transaction_count=10000, blocks_to_mine=50)
```

## 输出结果

模拟完成后，程序会生成以下输出：

1. 控制台输出：显示模拟过程中的关键信息和统计数据
2. JSON文件：在`data/`目录下生成详细的区块和交易数据
3. 图表：在`results/`目录下生成TPS图表和交易确认时间分布图

## 作者

- [Linpeng Jia](https://github.com/jialinpeng) - 人类开发者
- [通义灵码](https://tongyi.aliyun.com/) - AI编码助手
