import json
import time
import random
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import os
from collections import defaultdict

# 尝试导入绘图库
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Plotting functionality will be disabled.")


class ConsensusType(Enum):
    POW = "pow"
    PBFT = "pbft"
    HOTSTUFF = "hotstuff"
    DUMBO = "dumbo"


class Transaction:
    def __init__(self, tx_id: str, sender: str, receiver: str, amount: float, timestamp: float, size: int = 300):
        self.tx_id = tx_id
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = timestamp
        self.size = size  # 每笔交易的大小（字节）

    def to_dict(self) -> Dict:
        return {
            'tx_id': self.tx_id,
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': self.amount,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Transaction':
        return cls(
            data['tx_id'],
            data['sender'],
            data['receiver'],
            data['amount'],
            data['timestamp']
        )

    def get_size(self) -> int:
        """获取交易大小（字节）"""
        return self.size


class Block:
    BLOCK_HEADER_SIZE = 100  # 区块头大小（字节）
    
    def __init__(self, index: int, previous_hash: str, timestamp: float, 
                 transactions: List[Transaction], nonce: int = 0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_data = f"{self.index}{self.previous_hash}{self.timestamp}" + \
                    f"{[tx.tx_id for tx in self.transactions]}{self.nonce}"
        return hashlib.sha256(block_data.encode()).hexdigest()

    def to_dict(self) -> Dict:
        return {
            'index': self.index,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'nonce': self.nonce,
            'hash': self.hash
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Block':
        transactions = [Transaction.from_dict(tx_data) for tx_data in data['transactions']]
        block = cls(
            data['index'],
            data['previous_hash'],
            data['timestamp'],
            transactions,
            data['nonce']
        )
        block.hash = data['hash']
        return block

    def get_size(self) -> int:
        """获取区块大小（字节）"""
        return self.BLOCK_HEADER_SIZE + sum(tx.get_size() for tx in self.transactions)


class NetworkProtocol(Enum):
    DIRECT = "direct"
    GOSSIP = "gossip"


class Node:
    def __init__(self, node_id: int, address: str):
        self.node_id = node_id
        self.address = address
        self.blockchain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.bandwidth = 100  # 默认带宽 (Mbps)

    def add_transaction(self, transaction: Transaction):
        """添加交易到待处理列表"""
        self.pending_transactions.append(transaction)

    def get_latest_block(self) -> Optional[Block]:
        """获取最新的区块"""
        if not self.blockchain:
            return None
        return self.blockchain[-1]

    def add_block(self, block: Block):
        """添加区块到区块链"""
        self.blockchain.append(block)

    def set_bandwidth(self, bandwidth: float):
        """设置节点带宽"""
        self.bandwidth = bandwidth


class NetworkTopology:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes
        self.connections: Dict[int, List[int]] = defaultdict(list)

    def add_connection(self, node_a_id: int, node_b_id: int):
        """添加节点间连接"""
        if node_b_id not in self.connections[node_a_id]:
            self.connections[node_a_id].append(node_b_id)
        if node_a_id not in self.connections[node_b_id]:
            self.connections[node_b_id].append(node_a_id)

    def load_from_json(self, file_path: str):
        """从JSON文件加载网络拓扑"""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # 重置连接
        self.connections = defaultdict(list)
        
        # 加载节点带宽
        if 'nodes' in data:
            for node_data in data['nodes']:
                node_id = node_data['id']
                bandwidth = node_data.get('bandwidth', 100)
                for node in self.nodes:
                    if node.node_id == node_id:
                        node.set_bandwidth(bandwidth)
        
        # 加载连接信息
        if 'connections' in data:
            for conn in data['connections']:
                self.add_connection(conn['node_a'], conn['node_b'])

    def generate_topology(self, protocol: NetworkProtocol):
        """根据传输协议动态生成网络拓扑"""
        node_count = len(self.nodes)
        if node_count <= 1:
            return
            
        # 重置连接
        self.connections = defaultdict(list)
        
        if protocol == NetworkProtocol.DIRECT:
            # 全连接拓扑 - 每个节点都与其他所有节点连接
            for i in range(node_count):
                for j in range(i + 1, node_count):
                    self.add_connection(i, j)
        elif protocol == NetworkProtocol.GOSSIP:
            # 随机部分连接拓扑 - 每个节点随机连接到约30%的其他节点
            connection_probability = 0.3
            for i in range(node_count):
                for j in range(i + 1, node_count):
                    if random.random() < connection_probability:
                        self.add_connection(i, j)
            
            # 确保每个节点至少有一个连接
            for i in range(node_count):
                if not self.connections[i]:
                    # 随机选择一个不同的节点连接
                    j = random.choice([x for x in range(node_count) if x != i])
                    self.add_connection(i, j)

    def get_neighbors(self, node_id: int) -> List[int]:
        """获取节点的邻居列表"""
        return self.connections[node_id]


class NetworkTransport:
    """网络传输层，支持不同的传输协议"""
    
    def __init__(self, topology: NetworkTopology, protocol: NetworkProtocol = NetworkProtocol.DIRECT):
        self.topology = topology
        self.protocol = protocol
    
    def broadcast_transactions(self, nodes: List[Node], transactions: List[Transaction]):
        """广播交易到网络中的节点"""
        if self.protocol == NetworkProtocol.DIRECT:
            self._direct_broadcast(nodes, transactions)
        elif self.protocol == NetworkProtocol.GOSSIP:
            self._gossip_broadcast(nodes, transactions)
    
    def _direct_broadcast(self, nodes: List[Node], transactions: List[Transaction]):
        """直接广播：将交易发送到所有节点"""
        for node in nodes:
            node.pending_transactions.extend(transactions)
    
    def _gossip_broadcast(self, nodes: List[Node], transactions: List[Transaction]):
        """Gossip广播：随机选择节点进行传播"""
        # 首先将交易发送给部分节点
        initial_nodes = random.sample(nodes, max(1, len(nodes) // 3))
        for node in initial_nodes:
            node.pending_transactions.extend(transactions)
        
        # 模拟Gossip传播过程
        for _ in range(3):  # 3轮传播
            for node in nodes:
                if node.pending_transactions:  # 如果节点有交易
                    # 随机选择邻居节点进行传播
                    neighbors = self.topology.get_neighbors(node.node_id)
                    if neighbors:
                        random_neighbor_id = random.choice(neighbors)
                        neighbor_node = next((n for n in nodes if n.node_id == random_neighbor_id), None)
                        if neighbor_node and len(neighbor_node.pending_transactions) < len(node.pending_transactions):
                            # 传播部分交易
                            tx_to_send = random.sample(
                                node.pending_transactions, 
                                min(5, len(node.pending_transactions))
                            )
                            for tx in tx_to_send:
                                if tx not in neighbor_node.pending_transactions:
                                    neighbor_node.pending_transactions.append(tx)


class ConsensusAlgorithm(ABC):
    @abstractmethod
    def validate_block(self, block: Block, previous_block: Optional[Block]) -> bool:
        """验证区块"""
        pass

    @abstractmethod
    def create_block(self, node: Node, previous_block: Optional[Block]) -> Optional[Block]:
        """创建新区块"""
        pass

    @abstractmethod
    def reach_consensus(self, nodes: List[Node], transactions: List[Transaction], transport: NetworkTransport) -> Optional[Block]:
        """在节点间达成共识"""
        pass


class PoWConsensus(ConsensusAlgorithm):
    def __init__(self, difficulty: int = 4, max_transactions_per_block: int = 256):
        self.difficulty = difficulty
        self.max_transactions_per_block = max_transactions_per_block

    def validate_block(self, block: Block, previous_block: Optional[Block]) -> bool:
        # 检查哈希是否满足难度要求
        if not block.hash.startswith('0' * self.difficulty):
            return False
            
        # 检查前一区块哈希
        if previous_block and block.previous_hash != previous_block.hash:
            return False
            
        # 验证区块哈希
        return block.hash == block.calculate_hash()

    def create_block(self, node: Node, previous_block: Optional[Block]) -> Optional[Block]:
        if not node.pending_transactions:
            return None
            
        # 取前N个交易作为示例
        transactions = node.pending_transactions[:self.max_transactions_per_block]
        previous_hash = previous_block.hash if previous_block else "0" * 64
        timestamp = time.time()
        
        nonce = 0
        block_hash = ""
        while not block_hash.startswith('0' * self.difficulty):
            nonce += 1
            block_data = f"{len(node.blockchain)}{previous_hash}{timestamp}" + \
                        f"{[tx.tx_id for tx in transactions]}{nonce}"
            block_hash = hashlib.sha256(block_data.encode()).hexdigest()
            
            # 添加一些计算延迟
            if nonce % 1000 == 0:
                time.sleep(0.001)
        
        block = Block(len(node.blockchain), previous_hash, timestamp, transactions, nonce)
        block.hash = block_hash
        return block

    def reach_consensus(self, nodes: List[Node], transactions: List[Transaction], transport: NetworkTransport) -> Optional[Block]:
        # 简化的PoW共识：第一个完成挖矿的节点获胜
        if not transactions:
            return None
            
        # 使用传输层广播交易
        transport.broadcast_transactions(nodes, transactions)
            
        # 模拟挖矿过程
        for node in nodes:
            previous_block = node.get_latest_block()
            block = self.create_block(node, previous_block)
            if block:
                # 第一个完成的节点获胜
                return block
                
        return None


class PBFTConsensus(ConsensusAlgorithm):
    def __init__(self, max_transactions_per_block: int = 256):
        self.max_transactions_per_block = max_transactions_per_block
    
    def validate_block(self, block: Block, previous_block: Optional[Block]) -> bool:
        # 检查前一区块哈希
        if previous_block and block.previous_hash != previous_block.hash:
            return False
        return True

    def create_block(self, node: Node, previous_block: Optional[Block]) -> Optional[Block]:
        if not node.pending_transactions:
            return None
            
        # 取前N个交易作为示例
        transactions = node.pending_transactions[:self.max_transactions_per_block]
        previous_hash = previous_block.hash if previous_block else "0" * 64
        timestamp = time.time()
        
        # 模拟PBFT的处理时间（包括预准备、准备和提交阶段）
        # 根据交易数量添加适当的延迟
        processing_time = len(transactions) * 0.001  # 每个交易0.001秒处理时间
        time.sleep(processing_time)
        
        block = Block(len(node.blockchain), previous_hash, timestamp, transactions)
        return block

    def reach_consensus(self, nodes: List[Node], transactions: List[Transaction], transport: NetworkTransport) -> Optional[Block]:
        if not transactions:
            return None
            
        # 使用传输层广播交易
        transport.broadcast_transactions(nodes, transactions)
        
        # 模拟网络传输延迟
        time.sleep(0.05)  # 50ms网络延迟
        
        # 选择主节点（简化版PBFT）
        primary_node = nodes[0]
        previous_block = primary_node.get_latest_block()
        block = self.create_block(primary_node, previous_block)
        return block


class HotStuffConsensus(ConsensusAlgorithm):
    def __init__(self, max_transactions_per_block: int = 256):
        self.max_transactions_per_block = max_transactions_per_block
        self.pipeline_depth = 3  # 流水线深度
        self.current_leader = 0  # 当前领导者索引
    
    def validate_block(self, block: Block, previous_block: Optional[Block]) -> bool:
        # 检查前一区块哈希
        if previous_block and block.previous_hash != previous_block.hash:
            return False
        return True

    def create_block(self, node: Node, previous_block: Optional[Block]) -> Optional[Block]:
        if not node.pending_transactions:
            return None
            
        # 取前N个交易作为示例
        transactions = node.pending_transactions[:self.max_transactions_per_block]
        previous_hash = previous_block.hash if previous_block else "0" * 64
        timestamp = time.time()
        
        # 模拟HotStuff的处理时间（包括多个投票阶段）
        # 根据交易数量添加适当的延迟，确保处理时间不会太短
        # 基础延迟确保即使交易很少也不会处理得太快
        base_processing_time = 0.05  # 基础处理时间50ms
        transaction_processing_time = len(transactions) * 0.0005  # 每个交易0.5ms处理时间
        
        # 模拟HotStuff三阶段投票的处理时间
        # 流水线设计使并行处理成为可能，所以不需要乘以pipeline_depth
        three_phase_voting_time = 0.03  # 三阶段投票时间
        
        processing_time = base_processing_time + transaction_processing_time + three_phase_voting_time
        time.sleep(processing_time)
        
        block = Block(len(node.blockchain), previous_hash, timestamp, transactions)
        return block

    def reach_consensus(self, nodes: List[Node], transactions: List[Transaction], transport: NetworkTransport) -> Optional[Block]:
        if not transactions:
            return None
            
        # 使用传输层广播交易
        transport.broadcast_transactions(nodes, transactions)
        
        # 模拟网络传输延迟
        time.sleep(0.03)  # 30ms网络延迟
        
        # HotStuff实现：领导者轮换
        leader_node = nodes[self.current_leader]
        previous_block = leader_node.get_latest_block()
        block = self.create_block(leader_node, previous_block)
        
        # 轮换领导者
        self.current_leader = (self.current_leader + 1) % len(nodes)
        
        # 模拟流水线效果：如果流水线中有多个区块正在处理，则整体处理时间会减少
        # 流水线通过并行处理提高吞吐量，而不是增加单个区块的处理时间
        # 这里我们不添加额外的延迟，而是通过领导者轮换来体现流水线效果
        
        return block


class DumboConsensus(ConsensusAlgorithm):
    def __init__(self, max_transactions_per_block: int = 256):
        self.max_transactions_per_block = max_transactions_per_block
    
    def validate_block(self, block: Block, previous_block: Optional[Block]) -> bool:
        # 检查前一区块哈希
        if previous_block and block.previous_hash != previous_block.hash:
            return False
        return True

    def create_block(self, node: Node, previous_block: Optional[Block]) -> Optional[Block]:
        if not node.pending_transactions:
            return None
            
        # 取前N个交易作为示例
        transactions = node.pending_transactions[:self.max_transactions_per_block]
        previous_hash = previous_block.hash if previous_block else "0" * 64
        timestamp = time.time()
        
        # 模拟Dumbo的处理时间（异步共识）
        # 根据交易数量添加适当的延迟
        processing_time = len(transactions) * 0.002  # 每个交易0.002秒处理时间
        time.sleep(processing_time)
        
        block = Block(len(node.blockchain), previous_hash, timestamp, transactions)
        return block

    def reach_consensus(self, nodes: List[Node], transactions: List[Transaction], transport: NetworkTransport) -> Optional[Block]:
        if not transactions:
            return None
            
        # 使用传输层广播交易
        transport.broadcast_transactions(nodes, transactions)
        
        # 模拟网络传输延迟
        time.sleep(0.1)  # 100ms网络延迟
        
        # Dumbo简化实现：选择第一个节点作为领导者
        leader_node = nodes[0]
        previous_block = leader_node.get_latest_block()
        block = self.create_block(leader_node, previous_block)
        return block


class BlockchainSimulator:
    def __init__(self, node_count: int = 4, consensus_type: ConsensusType = ConsensusType.POW, 
                 network_protocol: NetworkProtocol = NetworkProtocol.DIRECT,
                 transaction_send_rate: Optional[float] = None,
                 max_transactions_per_block: int = 256,
                 transaction_size: int = 300):
        self.node_count = node_count
        self.consensus_type = consensus_type
        self.network_protocol = network_protocol
        self.transaction_send_rate = transaction_send_rate  # 交易发送速率 (txs/sec)
        self.max_transactions_per_block = max_transactions_per_block  # 每个区块最大交易数
        self.transaction_size = transaction_size  # 每笔交易大小（字节）
        self.nodes: List[Node] = []
        self.topology: Optional[NetworkTopology] = None
        self.transport: Optional[NetworkTransport] = None
        self.consensus: Optional[ConsensusAlgorithm] = None
        self.block_tps_history: List[Dict] = []  # 存储区块TPS历史记录
        self.transaction_confirm_times: List[float] = []  # 存储交易确认时间
        self._initialize_nodes()
        self._initialize_topology()
        self._initialize_consensus()

    def _initialize_nodes(self):
        """初始化节点"""
        self.nodes = []
        for i in range(self.node_count):
            node = Node(i, f"192.168.1.{i+1}")
            self.nodes.append(node)

    def _initialize_topology(self):
        """初始化网络拓扑"""
        self.topology = NetworkTopology(self.nodes)
        self.topology.generate_topology(self.network_protocol)
        self.transport = NetworkTransport(self.topology, self.network_protocol)

    def _initialize_consensus(self):
        """初始化共识算法"""
        if self.consensus_type == ConsensusType.POW:
            self.consensus = PoWConsensus(max_transactions_per_block=self.max_transactions_per_block)
        elif self.consensus_type == ConsensusType.PBFT:
            self.consensus = PBFTConsensus(max_transactions_per_block=self.max_transactions_per_block)
        elif self.consensus_type == ConsensusType.HOTSTUFF:
            self.consensus = HotStuffConsensus(max_transactions_per_block=self.max_transactions_per_block)
        elif self.consensus_type == ConsensusType.DUMBO:
            self.consensus = DumboConsensus(max_transactions_per_block=self.max_transactions_per_block)

    def set_consensus_algorithm(self, consensus_type: ConsensusType):
        """设置共识算法"""
        self.consensus_type = consensus_type
        self._initialize_consensus()

    def set_network_protocol(self, protocol: NetworkProtocol):
        """设置网络传输协议"""
        self.network_protocol = protocol
        if self.topology:
            self.topology.generate_topology(protocol)
            self.transport = NetworkTransport(self.topology, protocol)

    def set_topology(self, topology: NetworkTopology):
        """设置网络拓扑"""
        self.topology = topology
        self.transport = NetworkTransport(topology, self.network_protocol)

    def load_topology_from_json(self, file_path: str):
        """从JSON文件加载网络拓扑"""
        self.topology = NetworkTopology(self.nodes)
        self.topology.load_from_json(file_path)
        self.transport = NetworkTransport(self.topology, self.network_protocol)

    def generate_transactions(self, count: int) -> List[Transaction]:
        """生成随机交易"""
        transactions = []
        for i in range(count):
            tx = Transaction(
                f"tx_{int(time.time()*1000000)}_{i}",
                f"user_{random.randint(1, 100)}",
                f"user_{random.randint(1, 100)}",
                random.uniform(0.1, 1000.0),
                time.time(),
                self.transaction_size  # 使用设置的交易大小
            )
            transactions.append(tx)
        return transactions

    def submit_transaction(self, node_id: int, transaction: Transaction):
        """向指定节点提交交易"""
        if 0 <= node_id < len(self.nodes):
            self.nodes[node_id].add_transaction(transaction)

    def set_transaction_send_rate(self, rate: float):
        """设置交易发送速率（txs/sec）"""
        self.transaction_send_rate = rate

    def run_simulation(self, transaction_count: int = 100, blocks_to_mine: int = 10):
        """运行模拟"""
        print(f"Starting blockchain simulation with {self.node_count} nodes")
        print(f"Consensus: {self.consensus_type.value}")
        print(f"Network Protocol: {self.network_protocol.value}")
        
        # 生成交易
        transactions = self.generate_transactions(transaction_count)
        
        # 记录交易提交时间
        submit_times = {tx.tx_id: tx.timestamp for tx in transactions}
        
        # 如果设置了交易发送速率，则按速率发送交易并在发送过程中进行共识
        if self.transaction_send_rate is not None:
            print(f"Sending transactions at rate: {self.transaction_send_rate} txs/sec")
            self._send_transactions_and_mine_concurrently(transactions, submit_times, blocks_to_mine)
        else:
            # 原有的区块挖掘逻辑
            self._mine_blocks_sequentially(transactions, submit_times, blocks_to_mine)
        
        print("Simulation completed")
        
        # 保存区块和交易的详细信息到文件
        self._save_block_details()
        
        # 绘制TPS图表和交易确认时间分布图
        self._plot_tps_chart()
        self._plot_confirmation_time_distribution()

    def _mine_blocks_sequentially(self, transactions: List[Transaction], submit_times: Dict[str, float], blocks_to_mine: int):
        """按顺序挖掘区块（原有逻辑）"""
        transaction_count = len(transactions)
        
        for block_index in range(blocks_to_mine):
            print(f"Mining block {block_index + 1}/{blocks_to_mine}")
            
            # 获取该轮次需要处理的交易
            start_idx = block_index * (transaction_count // blocks_to_mine)
            end_idx = min((block_index + 1) * (transaction_count // blocks_to_mine), transaction_count)
            round_transactions = transactions[start_idx:end_idx]
            
            # 将交易广播到所有节点
            for node in self.nodes:
                node.pending_transactions.extend(round_transactions)
            
            # 记录开始时间
            start_time = time.time()
            
            # 达成共识并创建区块
            block = self.consensus.reach_consensus(self.nodes, round_transactions, self.transport)
            
            if block:
                # 记录结束时间
                end_time = time.time()
                
                # 计算TPS
                block_time = end_time - start_time
                tps = len(block.transactions) / block_time if block_time > 0 else 0
                
                # 计算区块大小
                block_size = block.get_size()
                
                # 将区块添加到所有节点
                for node in self.nodes:
                    node.add_block(block)
                
                # 记录交易确认时间
                confirm_time = time.time()
                for tx in block.transactions:
                    if tx.tx_id in submit_times:
                        self.transaction_confirm_times.append(confirm_time - submit_times[tx.tx_id])
                
                # 记录区块信息
                block_info = {
                    'block_height': block_index,
                    'transaction_count': len(block.transactions),
                    'block_size': block_size,
                    'block_time': block_time,
                    'tps': tps
                }
                self.block_tps_history.append(block_info)
                
                print(f"Block {block_index}: {len(block.transactions)} transactions, "
                      f"Size: {block_size} bytes, Time: {block_time:.2f}s, TPS: {tps:.2f}")
            else:
                print(f"Block {block_index}: No block created")

    def _send_transactions_and_mine_concurrently(self, transactions: List[Transaction], submit_times: Dict[str, float], blocks_to_mine: int):
        """并发发送交易和挖矿"""
        # 计算每个区块平均需要的交易数
        transactions_per_block = self.max_transactions_per_block
        
        # 计算每个交易之间的时间间隔
        interval = 1.0 / self.transaction_send_rate if self.transaction_send_rate and self.transaction_send_rate > 0 else 0.01
        
        # 记录开始发送交易的时间
        start_sending_time = time.time()
        
        # 用于记录已发送的交易数量
        sent_transactions = 0
        total_transactions = len(transactions)
        
        # 记录已创建的区块数量和已打包的交易数量
        blocks_mined = 0
        transactions_mined = 0
        
        # 记录上次挖矿时间，用于控制挖矿频率不超过交易发送速率
        last_mine_time = start_sending_time
        
        # 开始发送交易并同时挖矿
        while sent_transactions < total_transactions and blocks_mined < blocks_to_mine and transactions_mined < total_transactions:
            # 计算当前应该发送的交易批次
            batch_end = min(sent_transactions + max(1, int(self.transaction_send_rate)), total_transactions)
            
            # 发送一批交易
            for i in range(sent_transactions, batch_end):
                # 检查是否已达到区块数量限制或交易总数限制
                if blocks_mined >= blocks_to_mine or transactions_mined >= total_transactions:
                    break
                    
                # 计算应该发送的时间点
                expected_time = (i - sent_transactions) * interval
                
                # 获取当前时间
                current_time = time.time()
                
                # 计算需要等待的时间
                elapsed_time = current_time - start_sending_time
                wait_time = expected_time - elapsed_time
                
                # 如果需要等待，则等待
                if wait_time > 0:
                    time.sleep(wait_time)
                
                # 随机选择一个节点发送交易
                node = random.choice(self.nodes)
                node.add_transaction(transactions[i])
                
                # 更新已发送交易数
                sent_transactions = i + 1
                
                # 检查是否可以创建区块（当节点有足够的交易时）
                # 同时确保挖矿频率不超过交易发送速率
                current_time = time.time()
                time_since_last_mine = current_time - last_mine_time
                min_time_between_mines = 1.0 / self.transaction_send_rate if self.transaction_send_rate else 0.1
                
                if (len(node.pending_transactions) >= transactions_per_block and 
                    time_since_last_mine >= min_time_between_mines):
                    block = self._mine_one_block(node, submit_times)
                    if block:
                        blocks_mined += 1
                        transactions_mined += len(block.transactions)
                        last_mine_time = current_time
                        # 检查是否已达到区块数量限制或交易总数限制
                        if blocks_mined >= blocks_to_mine or transactions_mined >= total_transactions:
                            break
            
            # 如果已达到区块数量限制或交易总数限制，则退出循环
            if blocks_mined >= blocks_to_mine or transactions_mined >= total_transactions:
                break
                
            # 定期检查所有节点是否可以创建区块
            # 同样确保挖矿频率不超过交易发送速率
            current_time = time.time()
            time_since_last_mine = current_time - last_mine_time
            min_time_between_mines = 1.0 / self.transaction_send_rate if self.transaction_send_rate else 0.1
            
            for node in self.nodes:
                if blocks_mined >= blocks_to_mine or transactions_mined >= total_transactions:
                    break
                if (len(node.pending_transactions) >= transactions_per_block and 
                    time_since_last_mine >= min_time_between_mines):
                    block = self._mine_one_block(node, submit_times)
                    if block:
                        blocks_mined += 1
                        transactions_mined += len(block.transactions)
                        last_mine_time = current_time
        
        # 处理剩余的交易（如果还需要更多区块且未达到交易总数限制）
        # 同样确保挖矿频率不超过交易发送速率
        current_time = time.time()
        time_since_last_mine = current_time - last_mine_time
        min_time_between_mines = 1.0 / self.transaction_send_rate if self.transaction_send_rate else 0.1
        
        if blocks_mined < blocks_to_mine and time_since_last_mine >= min_time_between_mines:
            for node in self.nodes:
                if blocks_mined >= blocks_to_mine or transactions_mined >= total_transactions:
                    break
                if node.pending_transactions:
                    block = self._mine_one_block(node, submit_times)
                    if block:
                        blocks_mined += 1
                        transactions_mined += len(block.transactions)
                        if transactions_mined >= total_transactions:
                            break

    def _mine_one_block(self, node: Node, submit_times: Dict[str, float]):
        """挖掘一个区块"""
        # 获取节点的待处理交易（不超过最大交易数限制）
        transactions_to_mine = node.pending_transactions[:self.max_transactions_per_block]
        
        # 从待处理交易中移除已选中的交易
        node.pending_transactions = node.pending_transactions[len(transactions_to_mine):]
        
        # 如果没有交易需要处理，则不创建区块
        if not transactions_to_mine:
            return None
        
        # 记录开始时间
        start_time = time.time()
        
        # 达成共识并创建区块
        block = self.consensus.reach_consensus(self.nodes, transactions_to_mine, self.transport)
        
        if block:
            # 记录结束时间
            end_time = time.time()
            
            # 计算TPS（基于实际的时间窗口）
            block_time = end_time - start_time
            tps = len(block.transactions) / block_time if block_time > 0 else 0
            
            # 计算区块大小
            block_size = block.get_size()
            
            # 将区块添加到所有节点
            for n in self.nodes:
                n.add_block(block)
            
            # 记录交易确认时间
            confirm_time = time.time()
            for tx in block.transactions:
                if tx.tx_id in submit_times:
                    self.transaction_confirm_times.append(confirm_time - submit_times[tx.tx_id])
            
            # 记录区块信息
            block_info = {
                'block_height': len(self.nodes[0].blockchain) - 1,  # 区块高度为当前区块链长度减1
                'transaction_count': len(block.transactions),
                'block_size': block_size,
                'block_time': block_time,
                'tps': tps
            }
            self.block_tps_history.append(block_info)
            
            # 获取区块高度用于输出
            block_height = len(self.nodes[0].blockchain) - 1
            
            # 计算截至当前区块的总交易数
            total_transactions = sum(len(b.transactions) for b in self.nodes[0].blockchain)
            
            print(f"Block #{block_height} mined: {len(block.transactions)} transactions, "
                  f"Total: {total_transactions} transactions, "
                  f"Size: {block_size} bytes, Time: {block_time:.2f}s, TPS: {tps:.2f}")
            
            return block
        return None

    def _send_transactions_at_rate(self, transactions: List[Transaction]):
        """按照指定速率发送交易"""
        if not self.transaction_send_rate or self.transaction_send_rate <= 0:
            return
            
        # 计算每个交易之间的时间间隔
        interval = 1.0 / self.transaction_send_rate
        
        for i, tx in enumerate(transactions):
            # 计算应该发送的时间点
            expected_time = i * interval
            
            # 获取当前时间
            current_time = time.time()
            
            # 计算需要等待的时间
            elapsed_time = current_time - transactions[0].timestamp
            wait_time = expected_time - elapsed_time
            
            # 如果需要等待，则等待
            if wait_time > 0:
                time.sleep(wait_time)
            
            # 随机选择一个节点发送交易
            node = random.choice(self.nodes)
            node.add_transaction(tx)

    def _plot_confirmation_time_distribution(self):
        """绘制交易确认时间分布图并保存到results文件夹"""
        if not HAS_MATPLOTLIB:
            print("Skipping confirmation time distribution plotting due to missing matplotlib")
            return
            
        if not self.transaction_confirm_times:
            print("No transaction confirmation times to plot")
            return
            
        # 创建results目录
        os.makedirs('results', exist_ok=True)
        
        # 绘制直方图
        plt.figure(figsize=(10, 6))
        plt.hist(self.transaction_confirm_times, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Confirmation Time (seconds)')
        plt.ylabel('Frequency')
        plt.title(f'Transaction Confirmation Time Distribution ({self.consensus_type.value.upper()} Consensus)')
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        filename = f"results/confirmation_time_distribution_{self.consensus_type.value.lower()}_{self.network_protocol.value.lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图表，不显示
        
        print(f"Confirmation time distribution chart saved to {filename}")

    def _save_block_details(self):
        """将区块和交易的详细信息以JSON格式保存到data文件夹下的文件中"""
        # 创建data目录
        os.makedirs('data', exist_ok=True)
        
        # 获取第一个节点的区块链（所有节点应该有相同的区块链）
        if not self.nodes:
            print("没有节点")
            return
            
        blockchain = self.nodes[0].blockchain
        
        # 准备要写入文件的JSON数据
        block_data = []
        for block in blockchain:
            block_info = {
                "hash": block.hash,
                "previous_hash": block.previous_hash,
                "timestamp": block.timestamp,
                "transaction_count": len(block.transactions),
                "block_size": block.get_size(),  # 添加区块大小
                "nonce": block.nonce,
                "transactions": []
            }
            
            # 添加交易详情
            for tx in block.transactions:
                tx_info = {
                    "id": tx.tx_id,
                    "sender": tx.sender,
                    "receiver": tx.receiver,
                    "amount": tx.amount,
                    "timestamp": tx.timestamp,
                    "transaction_size": tx.get_size()  # 添加交易大小
                }
                block_info["transactions"].append(tx_info)
            
            block_data.append(block_info)
        
        # 添加统计信息
        total_blocks = len(blockchain)
        total_transactions = sum(len(block.transactions) for block in blockchain)
        avg_tps = sum(info['tps'] for info in self.block_tps_history) / len(self.block_tps_history) if self.block_tps_history else 0
        avg_confirmation_time = sum(self.transaction_confirm_times) / len(self.transaction_confirm_times) if self.transaction_confirm_times else 0
        total_data_size = sum(block.get_size() for block in blockchain)  # 总数据大小
        
        # 组装完整的JSON数据
        data = {
            "blocks": block_data,
            "statistics": {
                "total_blocks": total_blocks,
                "total_transactions": total_transactions,
                "average_tps": avg_tps,
                "average_confirmation_time": avg_confirmation_time,
                "total_data_size_bytes": total_data_size,  # 总数据大小（字节）
                "average_block_size": total_data_size / total_blocks if total_blocks > 0 else 0  # 平均区块大小
            }
        }
        
        # 写入JSON文件
        filename = f"data/block_details_{self.consensus_type.value}_{self.network_protocol.value}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"区块详细信息已保存到 {filename}")

    def _plot_tps_chart(self):
        """绘制TPS图表并保存到results文件夹"""
        if not HAS_MATPLOTLIB:
            print("Skipping TPS chart plotting due to missing matplotlib")
            return
            
        if not self.block_tps_history:
            print("No data to plot")
            return
            
        # 创建results目录
        os.makedirs('results', exist_ok=True)
        
        # 提取数据
        block_heights = [block['block_height'] for block in self.block_tps_history]
        tps_values = [block['tps'] for block in self.block_tps_history]
        
        # 绘制图表
        plt.figure(figsize=(10, 6))
        plt.plot(block_heights, tps_values, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.xlabel('Block Height')
        plt.ylabel('TPS (Transactions Per Second)')
        plt.title(f'Blockchain TPS by Block Height ({self.consensus_type.value.upper()} Consensus)')
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        filename = f"results/tps_chart_{self.consensus_type.value.lower()}_{self.network_protocol.value.lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图表，不显示
        
        print(f"TPS chart saved to {filename}")


# 使用示例
if __name__ == "__main__":
    # 创建模拟器实例
    simulator = BlockchainSimulator(
        node_count=20,
        consensus_type=ConsensusType.POW,
        network_protocol=NetworkProtocol.GOSSIP,
        transaction_send_rate=1000.0,  # 1000 transactions per second
        max_transactions_per_block=256,  # 每个区块最多256笔交易
        transaction_size=300  # 每笔交易300字节
    )
    
    # 运行模拟
    simulator.run_simulation(transaction_count=10000, blocks_to_mine=50)
