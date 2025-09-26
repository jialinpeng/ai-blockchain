import time
import hashlib
from typing import List, Dict
from enum import Enum
from collections import defaultdict


class ConsensusType(Enum):
    POW = "pow"
    PBFT = "pbft"
    HOTSTUFF = "hotstuff"
    DUMBO = "dumbo"


class Transaction:
    """
    交易类，表示区块链中的一笔交易
    """
    
    def __init__(self, tx_id: str, sender: str, receiver: str, amount: float, timestamp: float, size: int = 300):
        """
        初始化交易对象
        
        Args:
            tx_id: 交易ID
            sender: 发送方
            receiver: 接收方
            amount: 金额
            timestamp: 时间戳
            size: 交易大小（字节）
        """
        self.tx_id = tx_id
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.timestamp = timestamp
        self.size = size  # 每笔交易的大小（字节）

    def to_dict(self) -> Dict:
        """
        将交易对象转换为字典
        
        Returns:
            交易信息字典
        """
        return {
            'tx_id': self.tx_id,
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': self.amount,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Transaction':
        """
        从字典创建交易对象
        
        Args:
            data: 交易信息字典
            
        Returns:
            Transaction: 交易对象
        """
        return cls(
            data['tx_id'],
            data['sender'],
            data['receiver'],
            data['amount'],
            data['timestamp']
        )

    def get_size(self) -> int:
        """
        获取交易大小（字节）
        
        Returns:
            交易大小（字节）
        """
        return self.size


class Block:
    """
    区块类，表示区块链中的一个区块
    """
    
    BLOCK_HEADER_SIZE = 100  # 区块头大小（字节）
    
    def __init__(self, index: int, previous_hash: str, timestamp: float, 
                 transactions: List[Transaction], nonce: int = 0):
        """
        初始化区块对象
        
        Args:
            index: 区块索引
            previous_hash: 前一个区块的哈希值
            timestamp: 时间戳
            transactions: 交易列表
            nonce: 随机数
        """
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """
        计算区块哈希值
        
        Returns:
            区块哈希值
        """
        block_data = f"{self.index}{self.previous_hash}{self.timestamp}" + \
                    f"{[tx.tx_id for tx in self.transactions]}{self.nonce}"
        return hashlib.sha256(block_data.encode()).hexdigest()

    def to_dict(self) -> Dict:
        """
        将区块对象转换为字典
        
        Returns:
            区块信息字典
        """
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
        """
        从字典创建区块对象
        
        Args:
            data: 区块信息字典
            
        Returns:
            Block: 区块对象
        """
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
        """
        获取区块大小（字节）
        
        Returns:
            区块大小（字节）
        """
        return self.BLOCK_HEADER_SIZE + sum(tx.get_size() for tx in self.transactions)


class Node:
    """
    节点类，表示区块链网络中的一个节点
    """
    
    def __init__(self, node_id: int, ip_address: str, network_bandwidth: int = 2500000):
        """
        初始化节点对象
        
        Args:
            node_id: 节点ID
            ip_address: IP地址
            network_bandwidth: 网络带宽（字节/秒），默认为2500000字节/秒（20Mbps）
        """
        self.node_id = node_id
        self.ip_address = ip_address
        self.network_bandwidth = network_bandwidth  # 网络带宽（字节/秒）
        self.blockchain: List[Block] = []
        self.pending_transactions: List[Transaction] = []

    def add_block(self, block: Block):
        """
        添加区块到区块链
        
        Args:
            block: 区块对象
        """
        self.blockchain.append(block)

    def add_transaction(self, transaction: Transaction):
        """
        添加交易到待处理交易列表
        
        Args:
            transaction: 交易对象
        """
        self.pending_transactions.append(transaction)

    def get_latest_block(self):
        """
        获取最新的区块
        
        Returns:
            最新的区块对象，如果没有区块则返回None
        """
        return self.blockchain[-1] if self.blockchain else None
