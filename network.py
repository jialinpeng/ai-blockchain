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

    def to_dict(self) -> Dict:
        """
        将网络拓扑转换为字典
        
        Returns:
            网络拓扑信息字典
        """
        return {
            'nodes': [node.node_id for node in self.nodes],
            'adjacency_list': dict(self.adjacency_list)
        }

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

    def broadcast_transactions(self, nodes: List[Node], transactions: List[Transaction]):
        """
        广播交易到所有节点
        
        Args:
            nodes: 节点列表
            transactions: 交易列表
        """
        if self.protocol == NetworkProtocol.DIRECT:
            # 直接广播给所有节点
            for node in nodes:
                node.pending_transactions.extend(transactions)
        elif self.protocol == NetworkProtocol.GOSSIP:
            self._gossip_transactions(nodes, transactions)

    def _gossip_transactions(self, nodes: List[Node], transactions: List[Transaction]):
        """
        使用Gossip协议传播交易
        
        Args:
            nodes: 节点列表
            transactions: 交易列表
        """
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