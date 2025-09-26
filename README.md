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
pip install -r requirements.txt
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
├── results/                # 性能图表和图形
├── templates/              # Web界面模板
├── tests/                  # 单元测试
└── README.md               # 本文件
```

## 配置选项

模拟器支持各种配置参数：

| 参数 | 描述 | 默认值 |
|-----------|-------------|---------|
| `node_count` | 网络中的节点数量 | 4 |
| `consensus_type` | 要使用的共识算法 | PoW |
| `network_protocol` | 网络通信协议 | DIRECT |
| `transaction_send_rate` | 交易发送速率 (txs/sec) | 300 |
| `max_transactions_per_block` | 每个区块的最大交易数 | 256 |
| `transaction_size` | 每个交易的大小 (字节) | 300 |
| `block_interval` | 出块间隔（秒） | 3.0 |

## Web界面功能

Web界面提供了一个直观的图形化操作环境，包括：

1. **配置面板** - 设置模拟参数，如节点数量、共识算法、网络协议等
2. **控制面板** - 启动和停止模拟过程
3. **实时日志** - 显示模拟过程中的详细日志信息
4. **区块信息** - 展示已确认区块的详细信息
5. **统计信息** - 实时显示TPS和交易确认时间图表
6. **网络拓扑** - 可视化展示节点间的连接关系
7. **传输日志** - 显示节点间的数据包传输记录

## 贡献

欢迎提交 Pull Request 来改进这个项目。对于重大更改，请先开 issue 讨论您想要改变的内容。

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 测试

本项目包含完整的单元测试套件，用于验证各个模块的功能正确性。

### 运行测试

```bash
# 运行所有测试
python run_tests.py

# 详细模式运行所有测试
python run_tests.py -v

# 运行特定模块的测试
python run_tests.py test_core
python run_tests.py test_network
```

### 测试结构

```
tests/
├── __init__.py
├── test_core.py      # 核心模块测试（Transaction、Block、Node等）
└── test_network.py   # 网络模块测试（NetworkTopology、NetworkTransport等）
```

### 测试覆盖范围

当前测试覆盖以下模块：

1. **core模块**：
   - Transaction类：交易的创建、序列化和大小计算
   - Block类：区块的创建、哈希计算、序列化和大小计算
   - Node类：节点的创建、交易和区块管理

2. **network模块**：
   - NetworkProtocol枚举：网络协议类型定义
   - NetworkTopology类：网络拓扑生成和管理
   - NetworkTransport类：网络传输和数据统计

### 添加新测试

要添加新的测试，请在`tests/`目录下创建新的测试文件，遵循以下命名约定：
- 文件名以`test_`开头，如`test_consensus.py`
- 测试类继承自`unittest.TestCase`
- 测试方法名以`test_`开头

## 作者

- [Linpeng Jia](https://github.com/jialinpeng) - 人类开发者
- [通义灵码](https://tongyi.aliyun.com/) - AI编码助手