# 区块链模拟器

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/your-username/ai-blockchain/pulls)

## 概述

本仓库包含一个由人类开发者 [Linpeng Jia](https://github.com/jialinpeng) 与 AI 编码助手 [通义灵码](https://tongyi.aliyun.com/) 共同协作开发的区块链模拟器。该项目旨在提供一个教育和研究平台，用于理解不同共识机制在各种网络条件下的性能表现。

## 功能特点

### 支持的共识算法
- **工作量证明 (PoW)** - 比特币使用的传统共识机制
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
- 以压缩JSON格式导出详细的区块和交易数据
- 不同共识算法和网络协议之间的性能比较
- **Web界面** - 基于浏览器的图形化界面，用于配置和监控模拟过程

### 可视化
- 按区块高度划分的TPS（每秒交易数）图表
- 交易确认时间累积分布函数(CDF)图
- 节点间数据传输热力图
- 网络拓扑可视化
- 实时监控面板显示区块信息、统计信息和传输日志

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

#### Web界面运行
```bash
python web_explorer.py
```
然后在浏览器中访问 `http://127.0.0.1:5000`

## 项目结构

项目采用模块化设计，主要包含以下模块：

```
ai-blockchain/
├── core.py                 # 核心数据结构模块，包含交易、区块和节点类
├── network.py              # 网络相关模块，包含网络协议、拓扑和传输类
├── consensus.py            # 共识算法模块，包含各种共识算法的实现
├── simulator.py            # 主模拟器模块，包含区块链模拟器主类
├── main.py                 # 主程序模块，包含交互式设置和程序入口
├── web_explorer.py         # Web界面模块，提供基于浏览器的图形化界面
├── requirements.txt        # 项目依赖
├── run_tests.py            # 测试运行器
├── git-auto-commit.sh      # Git自动提交脚本
├── data/                   # 模拟输出数据（压缩格式）
```

## 核心组件

### 1. 核心数据结构 (core.py)
- `Transaction`: 交易类，表示一笔交易数据
- `Block`: 区块类，包含交易列表和区块元数据
- `Node`: 节点类，表示网络中的一个节点
- `ConsensusType`: 共识算法类型枚举

### 2. 网络层 (network.py)
- `NetworkProtocol`: 网络协议枚举（DIRECT, GOSSIP）
- `NetworkTopology`: 网络拓扑类，管理节点间的连接关系
- `NetworkTransport`: 网络传输类，处理消息传递

### 3. 共识算法 (consensus.py)
- `ConsensusAlgorithm`: 共识算法抽象基类
- `PoWConsensus`: 工作量证明共识实现
- `PBFTConsensus`: 实用拜占庭容错共识实现
- `HotStuffConsensus`: HotStuff共识实现
- `DumboConsensus`: Dumbo共识实现

### 4. 模拟器 (simulator.py)
- `BlockchainSimulator`: 主模拟器类，协调所有组件运行模拟

## 配置参数

模拟器支持多种配置参数以适应不同的测试场景：

- `node_count`: 节点数量（默认：4）
- `consensus_type`: 共识算法类型（默认：POW）
- `network_protocol`: 网络协议（默认：DIRECT）
- `transaction_send_rate`: 交易发送速率（txs/sec，默认：根据共识算法自动设定）
- `max_transactions_per_block`: 每个区块最大交易数（默认：256）
- `transaction_size`: 每笔交易大小（字节，默认：300）
- `block_interval`: 出块间隔（秒，默认：3.0）
- `network_bandwidth`: 网络带宽（字节/秒，默认：2500000）

## 输出数据

模拟完成后，系统会生成以下数据文件：

- `blocks.json.gz`: 区块数据压缩文件
- `network.log.gz`: 网络传输日志压缩文件
- `tps_data.json.gz`: TPS数据压缩文件
- `confirmation_times.json.gz`: 确认时间数据压缩文件

数据文件存储在 `data/` 目录下，以时间戳命名的子目录中。

## 可视化功能

当系统检测到 matplotlib 库时，会自动生成以下图表：

1. TPS图表：显示每个区块的交易处理速度
2. 确认时间CDF图：显示交易确认时间的累积分布
3. 网络拓扑图：显示节点间的连接关系
4. 网络传输热力图：显示节点间的数据传输量

## 测试

运行测试套件：

```bash
python run_tests.py
```

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 贡献

欢迎提交 Pull Request。对于重大更改，请先开 issue 讨论您想要改变的内容。

## 作者

- **Linpeng Jia** - [GitHub](https://github.com/jialinpeng)
- **通义灵码** - [Tongyi Lab](https://tongyi.aliyun.com/)

本项目由 Linpeng Jia 与 通义灵码 共同协作开发。