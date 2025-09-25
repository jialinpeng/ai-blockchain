# ai-blockchain

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/your-username/ai-blockchain/pulls)

## Overview

This repository contains a blockchain simulator built with the help of Tongyi Lingma (通义灵码), designed to simulate typical blockchain systems and consensus algorithms. The project aims to provide an educational and research platform for understanding how different consensus mechanisms perform under various network conditions.

This project is a collaborative effort between [Linpeng Jia](jialinpeng@ict.ac.cn) and Tongyi Lingma (通义灵码), demonstrating how human expertise combined with AI coding assistance can create sophisticated distributed systems.

## Motivation

The primary purpose of creating this repository is to leverage the power of Tongyi Lingma to build a comprehensive blockchain simulation system. This project demonstrates how AI coding assistants can be used to develop complex distributed systems and provides a flexible platform for experimenting with different blockchain consensus algorithms.

## Features

### Supported Consensus Algorithms
- **Proof of Work (PoW)** - Traditional consensus used by Bitcoin
- **Practical Byzantine Fault Tolerance (PBFT)** - Classical BFT consensus algorithm
- **HotStuff** - Modern BFT consensus with pipelining capabilities
- **Dumbo** - Asynchronous consensus algorithm

### Network Protocols
- **Direct** - Full mesh network topology where all nodes are directly connected
- **Gossip** - Partially connected network with randomized message propagation

### Simulation Capabilities
- Configurable number of nodes, transaction rates, and block sizes
- Real-time performance metrics collection
- Transaction confirmation time analysis
- Throughput (TPS) measurement and visualization
- Detailed block and transaction data export in JSON format
- Performance comparison across different consensus algorithms and network protocols

### Visualization
- TPS (Transactions Per Second) charts by block height
- Transaction confirmation time distribution histograms
- Automatically generated performance reports

## Getting Started

### Prerequisites
- Python 3.7+
- Optional: matplotlib for visualization features

### Installation
```bash
git clone https://github.com/your-username/ai-blockchain.git
cd ai-blockchain
```

### Basic Usage
```python
from blockchain_simulator import BlockchainSimulator, ConsensusType, NetworkProtocol

# Create a simulator instance
simulator = BlockchainSimulator(
    node_count=20,
    consensus_type=ConsensusType.HOTSTUFF,
    network_protocol=NetworkProtocol.GOSSIP,
    transaction_send_rate=1000.0
)

# Run the simulation
simulator.run_simulation(
    transaction_count=10000,
    blocks_to_mine=50
)
```

### Output
The simulator generates:
- Detailed block information in `data/` directory
- Performance charts in `results/` directory
- Statistical summaries for analysis

## Project Structure
```
ai-blockchain/
├── blockchain_simulator.py    # Main simulator implementation
├── data/                      # Simulation output data
├── results/                   # Performance charts and graphs
└── README.md                 # This file
```

## Configuration Options

The simulator supports various configuration parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `node_count` | Number of nodes in the network | 4 |
| `consensus_type` | Consensus algorithm to use | PoW |
| `network_protocol` | Network communication protocol | DIRECT |
| `transaction_send_rate` | Transaction sending rate (txs/sec) | None |
| `max_transactions_per_block` | Maximum transactions per block | 256 |
| `transaction_size` | Size of each transaction (bytes) | 300 |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Whether it's adding new consensus algorithms, improving existing implementations, or enhancing the visualization capabilities, your contributions will help make this simulator more valuable for the community.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Linpeng Jia** - *Initial work* - [jialinpeng@ict.ac.cn](jialinpeng@ict.ac.cn)
- **Tongyi Lingma** - *AI Assistant* - Part of Alibaba Cloud's Tongyi Lab

## Acknowledgments

- This project was built with the assistance of Tongyi Lingma (通义灵码), demonstrating the capabilities of AI-powered coding assistants in developing complex distributed systems.
- Inspired by the need for educational tools to understand blockchain consensus mechanisms and their performance characteristics.