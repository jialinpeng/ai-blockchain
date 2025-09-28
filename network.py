import random
import time
from collections import defaultdict
from enum import Enum
from typing import List, Dict, Optional
from core import Node, Transaction, Block


class NetworkProtocol(Enum):
    """
    网络协议枚举
    """
    DIRECT = "direct"      # 直接连接（全连接）
    GOSSIP = "gossip"      # Gossip协议


class NetworkTopology:
    """
    网络拓扑类，用于定义节点间的连接关系
    """
    
    def __init__(self, node_count: int, protocol: NetworkProtocol):
        """
        初始化网络拓扑
        
        Args:
            node_count: 节点数量
            protocol: 网络协议类型
        """
        self.node_count = node_count
        self.protocol = protocol
        self.adjacency_list: Dict[int, List[int]] = defaultdict(list)
        self.nodes: List[Node] = []
        self._build_topology()

    def _build_topology(self):
        """
        构建网络拓扑结构
        """
        if self.protocol == NetworkProtocol.DIRECT:
            # 全连接网络：每个节点都与其他所有节点直接连接
            for i in range(self.node_count):
                for j in range(self.node_count):
                    if i != j:
                        self.adjacency_list[i].append(j)
        elif self.protocol == NetworkProtocol.GOSSIP:
            # 部分连接网络：每个节点随机连接到部分其他节点
            for i in range(self.node_count):
                # 每个节点连接到大约30%的其他节点
                connection_count = max(1, int(self.node_count * 0.3))
                available_nodes = [j for j in range(self.node_count) if j != i]
                self.adjacency_list[i] = random.sample(available_nodes, min(connection_count, len(available_nodes)))

    def apply_to_nodes(self, nodes: List[Node]):
        """
        将网络拓扑应用到节点列表
        
        Args:
            nodes: 节点列表
        """
        self.nodes = nodes

    def get_neighbors(self, node_id: int) -> List[int]:
        """
        获取指定节点的邻居节点列表
        
        Args:
            node_id: 节点ID
            
        Returns:
            邻居节点ID列表
        """
        return self.adjacency_list.get(node_id, [])


class NetworkTransport:
    """
    网络传输类，用于在节点间传输数据
    """
    
    def __init__(self, topology: NetworkTopology, protocol: NetworkProtocol):
        """
        初始化网络传输对象
        
        Args:
            topology: 网络拓扑对象
            protocol: 网络协议类型
        """
        self.topology = topology
        self.protocol = protocol
        # 跟踪节点间的数据传输量
        self.data_transfer_stats = defaultdict(lambda: defaultdict(int))
        # 存储传输日志
        self.transfer_logs = []

    def broadcast_transactions(self, sender_id: int, transactions: List[Transaction]):
        """
        广播交易到所有节点
        
        Args:
            sender_id: 发送者节点ID
            transactions: 交易列表
        """
        if not self.topology or not self.topology.nodes:
            return
            
        nodes = self.topology.nodes
        transaction_count = len(transactions)
        
        if self.protocol == NetworkProtocol.DIRECT:
            # 直接广播给所有节点
            for node in nodes:
                if node.node_id != sender_id:  # 不广播给自己
                    node.pending_transactions.extend(transactions)
                    # 记录数据传输量
                    self.data_transfer_stats[sender_id][node.node_id] += transaction_count
                    # 记录传输日志
                    self.transfer_logs.append({
                        'timestamp': time.time(),
                        'sender': sender_id,
                        'receiver': node.node_id,
                        'transaction_count': transaction_count,
                        'protocol': 'DIRECT',
                        'action': 'broadcast_transactions'
                    })
        elif self.protocol == NetworkProtocol.GOSSIP:
            self._gossip_transactions(sender_id, transactions)

    def _gossip_transactions(self, sender_id: int, transactions: List[Transaction], max_gossip_rounds: int = 3):
        """
        使用Gossip协议传播交易
        
        Args:
            sender_id: 发送者节点ID
            transactions: 交易列表
            max_gossip_rounds: 最大Gossip传播轮数
        """
        if not self.topology or not self.topology.nodes:
            return
            
        nodes = self.topology.nodes
        sender_node = next((n for n in nodes if n.node_id == sender_id), None)
        if not sender_node:
            return
            
        # 记录初始传输日志
        self.transfer_logs.append({
            'timestamp': time.time(),
            'sender': sender_id,
            'receiver': 'multiple',
            'transaction_count': len(transactions),
            'protocol': 'GOSSIP',
            'action': 'initial_gossip_broadcast'
        })
            
        # 首先将交易发送给部分节点
        initial_nodes = random.sample(nodes, max(1, len(nodes) // 3))
        for node in initial_nodes:
            if node.node_id != sender_id:  # 不发送给自己
                for tx in transactions:
                    if tx not in node.pending_transactions:
                        node.pending_transactions.append(tx)
        
        # 记录初始传输
        for node in initial_nodes:
            if node.node_id != sender_id:  # 不记录给自己发送
                self.data_transfer_stats[sender_id][node.node_id] += len(transactions)
                # 记录传输日志
                self.transfer_logs.append({
                    'timestamp': time.time(),
                    'sender': sender_id,
                    'receiver': node.node_id,
                    'transaction_count': len(transactions),
                    'protocol': 'GOSSIP',
                    'action': 'initial_gossip_send'
                })
        
        # 模拟Gossip传播过程
        for round_num in range(max_gossip_rounds):  # 指定轮数传播
            self.transfer_logs.append({
                'timestamp': time.time(),
                'sender': 'system',
                'receiver': 'all',
                'transaction_count': 0,
                'protocol': 'GOSSIP',
                'action': f'gossip_round_{round_num+1}_start'
            })
            
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
                            sent_count = 0
                            for tx in tx_to_send:
                                if tx not in neighbor_node.pending_transactions:
                                    neighbor_node.pending_transactions.append(tx)
                                    sent_count += 1
                            # 记录数据传输
                            if sent_count > 0:
                                self.data_transfer_stats[node.node_id][neighbor_node.node_id] += sent_count
                                # 记录传输日志
                                self.transfer_logs.append({
                                    'timestamp': time.time(),
                                    'sender': node.node_id,
                                    'receiver': neighbor_node.node_id,
                                    'transaction_count': sent_count,
                                    'protocol': 'GOSSIP',
                                    'action': f'gossip_round_{round_num+1}_propagate'
                                })
            
            self.transfer_logs.append({
                'timestamp': time.time(),
                'sender': 'system',
                'receiver': 'all',
                'transaction_count': 0,
                'protocol': 'GOSSIP',
                'action': f'gossip_round_{round_num+1}_end'
            })

    def broadcast_block(self, sender_id: int, block: Block):
        """
        广播区块到所有节点
        
        Args:
            sender_id: 发送者节点ID
            block: 区块对象
        """
        if not self.topology or not self.topology.nodes:
            return
            
        nodes = self.topology.nodes
        
        if self.protocol == NetworkProtocol.DIRECT:
            # 直接广播给所有节点
            for node in nodes:
                if node.node_id != sender_id:  # 不广播给自己
                    # 检查区块是否已存在于节点的区块链中
                    if not any(b.hash == block.hash for b in node.blockchain):
                        node.add_block(block)
                        # 记录数据传输量（区块大小以字节为单位）
                        block_size = block.get_size()
                        self.data_transfer_stats[sender_id][node.node_id] += block_size
                        # 记录传输日志
                        self.transfer_logs.append({
                            'timestamp': time.time(),
                            'sender': sender_id,
                            'receiver': node.node_id,
                            'block_size': block_size,
                            'block_hash': block.hash[:16],  # 只记录前16个字符
                            'protocol': 'DIRECT',
                            'action': 'broadcast_block'
                        })
        elif self.protocol == NetworkProtocol.GOSSIP:
            self._gossip_block(sender_id, block)

    def _gossip_block(self, sender_id: int, block: Block):
        """
        使用Gossip协议传播区块
        
        Args:
            sender_id: 发送者节点ID
            block: 区块对象
        """
        if not self.topology or not self.topology.nodes:
            return
            
        nodes = self.topology.nodes
        block_size = block.get_size()
        
        # 记录初始传输日志
        self.transfer_logs.append({
            'timestamp': time.time(),
            'sender': sender_id,
            'receiver': 'multiple',
            'block_size': block_size,
            'block_hash': block.hash[:16],
            'protocol': 'GOSSIP',
            'action': 'initial_gossip_block_broadcast'
        })
        
        # 首先将区块发送给部分节点
        initial_nodes = random.sample(nodes, max(1, len(nodes) // 3))
        for node in initial_nodes:
            if node.node_id != sender_id:  # 不发送给自己
                # 检查区块是否已存在于节点的区块链中
                if not any(b.hash == block.hash for b in node.blockchain):
                    node.add_block(block)
                    # 记录数据传输量
                    self.data_transfer_stats[sender_id][node.node_id] += block_size
                    # 记录传输日志
                    self.transfer_logs.append({
                        'timestamp': time.time(),
                        'sender': sender_id,
                        'receiver': node.node_id,
                        'block_size': block_size,
                        'block_hash': block.hash[:16],
                        'protocol': 'GOSSIP',
                        'action': 'initial_gossip_block_send'
                    })

    def get_data_transfer_stats(self) -> Dict:
        """
        获取数据传输统计信息
        
        Returns:
            数据传输统计信息
        """
        return dict(self.data_transfer_stats)
        
    def get_transfer_logs(self) -> List[Dict]:
        """
        获取传输日志
        
        Returns:
            传输日志列表
        """
        return self.transfer_logs