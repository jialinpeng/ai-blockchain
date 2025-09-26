# 区块链模拟器

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/your-username/ai-blockchain/pulls)

## 概述

本仓库包含一个由人类开发者 [Linpeng Jia](https://github.com/jialinpeng) 与 AI 编码助手 [通义灵码](https://tongyi.aliyun.com/) 共同协作开发的区块链模拟器。该项目旨在提供一个教育和研究平台，用于理解不同共识机制在各种网络条件下的性能表现。

## 功能特点

### 支持的共识算法
- **工作量证明 (PoW)** - 比特币使用的传统共识机制，采用SHA256双重哈希算法
- **实用拜占庭容错 (PBFT)** - 经典的BFT共识算法
- **HotStuff** - 具有流水线能力的现代BFT共识
- **Dumbo** - 异步共识算法

### 网络协议
- **直接连接 (Direct)** - 所有节点直接相连的全网状网络拓扑
- **Gossip协议** - 通过随机消息传播的部分连接网络

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

#### 交互式运行
```bash
python main.py
```

#### 编程方式使用
```python
from simulator import BlockchainSimulator
from core import ConsensusType
from network import NetworkProtocol

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

## 项目结构

项目采用模块化设计，主要包含以下模块：

```
ai-blockchain/
├── core.py                 # 核心数据结构模块，包含交易、区块和节点类
├── network.py              # 网络相关模块，包含网络协议、拓扑和传输类
├── consensus.py            # 共识算法模块，包含各种共识算法的实现
├── simulator.py            # 主模拟器模块，包含区块链模拟器主类
├── main.py                 # 主程序模块，包含交互式设置和程序入口
├── requirements.txt        # 项目依赖
├── data/                   # 模拟输出数据
├── results/                # 性能图表和图形
└── README.md              # 本文件
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
| `block_interval` | 出块间隔（秒） | 0.0 |

## 贡献

欢迎提交 Pull Request 来改进这个项目。对于重大更改，请先开 issue 讨论您想要改变的内容。

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 作者

- [Linpeng Jia](https://github.com/jialinpeng) - 人类开发者
- [通义灵码](https://tongyi.aliyun.com/) - AI编码助手
