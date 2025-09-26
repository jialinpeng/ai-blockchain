import json
import time
import random
from typing import List, Dict
from collections import defaultdict
from enum import Enum

from core import Node, Transaction, Block


class NetworkProtocol(Enum):
    """
    网络协议类型枚举
    """
    DIRECT = "direct"
    GOSSIP = "gossip"


class NetworkTopology:
    """
    网络拓扑类，用于管理节点间的连接关系
    """
    
    def __init__(self, nodes: List[Node]):
        """
        初始化网络拓扑
        
        Args:
            nodes: 节点列表
        """
        self.nodes = nodes
        self.adjacency_list: Dict[int, List[int]] = defaultdict(list)

    def generate_topology(self, protocol: NetworkProtocol):
        """
        根据协议类型生成网络拓扑
        
        Args:
            protocol: 网络协议类型
        """
        node_count = len(self.nodes)
        self.adjacency_list.clear()
        
        if protocol == NetworkProtocol.DIRECT:
            # 全连接拓扑
            for i in range(node_count):
                for j in range(node_count):
                    if i != j:
                        self.adjacency_list[i].append(j)
        elif protocol == NetworkProtocol.GOSSIP:
            # 随机部分连接拓扑
            for i in range(node_count):
                # 每个节点连接到约30%的其他节点
                connection_count = max(1, int(node_count * 0.3))
                other_nodes = list(range(node_count))
                other_nodes.remove(i)
                connected_nodes = random.sample(other_nodes, min(connection_count, len(other_nodes)))
                self.adjacency_list[i].extend(connected_nodes)

    def get_neighbors(self, node_id: int) -> List[int]:
        """
        获取节点的邻居节点列表
        
        Args:
            node_id: 节点ID
            
        Returns:
            邻居节点ID列表
        """
        return self.adjacency_list.get(node_id, [])

    def to_dict(self, data_stats: Dict = None) -> Dict:
        """
        将网络拓扑转换为字典
        
        Returns:
            网络拓扑信息字典
        """
        result = {
            'nodes': [node.node_id for node in self.nodes],
            'adjacency_list': dict(self.adjacency_list)
        }
        
        if data_stats:
            result['data_stats'] = data_stats
            
        return result

    def load_from_json(self, file_path: str):
        """
        从JSON文件加载网络拓扑
        
        Args:
            file_path: JSON文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.adjacency_list.clear()
        for node_id, neighbors in data['adjacency_list'].items():
            self.adjacency_list[int(node_id)] = neighbors


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
        
        # 模拟Gossip传播过程
        for _ in range(max_gossip_rounds):  # 指定轮数传播
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

    def get_data_transfer_stats(self) -> Dict:
        """
        获取数据传输统计信息
        
        Returns:
            数据传输统计信息
        """
        return dict(self.data_transfer_stats)