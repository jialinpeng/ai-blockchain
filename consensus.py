import time
import hashlib
import random
from abc import ABC, abstractmethod
from typing import List, Optional

from core import Node, Block, Transaction
from network import NetworkTransport


class ConsensusAlgorithm(ABC):
    """
    共识算法抽象基类
    """
    
    @abstractmethod
    def validate_block(self, block: Block, previous_block: Optional[Block]) -> bool:
        """
        验证区块
        
        Args:
            block: 待验证的区块
            previous_block: 前一个区块
            
        Returns:
            bool: 验证结果
        """
        pass

    @abstractmethod
    def create_block(self, node: Node, previous_block: Optional[Block]) -> Optional[Block]:
        """
        创建新区块
        
        Args:
            node: 节点对象
            previous_block: 前一个区块
            
        Returns:
            Block: 新创建的区块，如果无法创建则返回None
        """
        pass

    @abstractmethod
    def reach_consensus(self, nodes: List[Node], transactions: List[Transaction], transport: NetworkTransport) -> Optional[Block]:
        """
        在节点间达成共识
        
        Args:
            nodes: 节点列表
            transactions: 交易列表
            transport: 网络传输对象
            
        Returns:
            Block: 达成共识的区块，如果没有达成则返回None
        """
        pass


class PoWConsensus(ConsensusAlgorithm):
    """
    工作量证明共识算法实现
    """
    
    def __init__(self, difficulty: int = 4, max_transactions_per_block: int = 256):
        """
        初始化PoW共识算法
        
        Args:
            difficulty: 挖矿难度
            max_transactions_per_block: 每个区块最大交易数
        """
        self.difficulty = difficulty
        self.max_transactions_per_block = max_transactions_per_block

    def validate_block(self, block: Block, previous_block: Optional[Block]) -> bool:
        """
        验证区块
        
        Args:
            block: 待验证的区块
            previous_block: 前一个区块
            
        Returns:
            bool: 验证结果
        """
        if previous_block and block.previous_hash != previous_block.hash:
            return False
        return block.hash.startswith('0' * self.difficulty)

    def create_block(self, node: Node, previous_block: Optional[Block]) -> Optional[Block]:
        """
        创建新区块
        
        Args:
            node: 节点对象
            previous_block: 前一个区块
            
        Returns:
            Block: 新创建的区块，如果无法创建则返回None
        """
        if not node.pending_transactions:
            return None
            
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
        """
        在节点间达成共识
        
        Args:
            nodes: 节点列表
            transactions: 交易列表
            transport: 网络传输对象
            
        Returns:
            Block: 达成共识的区块，如果没有达成则返回None
        """
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
    """
    实用拜占庭容错共识算法实现
    """
    
    def __init__(self, max_transactions_per_block: int = 256):
        """
        初始化PBFT共识算法
        
        Args:
            max_transactions_per_block: 每个区块最大交易数
        """
        self.max_transactions_per_block = max_transactions_per_block
    
    def validate_block(self, block: Block, previous_block: Optional[Block]) -> bool:
        """
        验证区块
        
        Args:
            block: 待验证的区块
            previous_block: 前一个区块
            
        Returns:
            bool: 验证结果
        """
        if previous_block and block.previous_hash != previous_block.hash:
            return False
        return True

    def create_block(self, node: Node, previous_block: Optional[Block]) -> Optional[Block]:
        """
        创建新区块
        
        Args:
            node: 节点对象
            previous_block: 前一个区块
            
        Returns:
            Block: 新创建的区块，如果无法创建则返回None
        """
        if not node.pending_transactions:
            return None
            
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
        """
        在节点间达成共识
        
        Args:
            nodes: 节点列表
            transactions: 交易列表
            transport: 网络传输对象
            
        Returns:
            Block: 达成共识的区块，如果没有达成则返回None
        """
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
    """
    HotStuff共识算法实现
    """
    
    def __init__(self, max_transactions_per_block: int = 256):
        """
        初始化HotStuff共识算法
        
        Args:
            max_transactions_per_block: 每个区块最大交易数
        """
        self.max_transactions_per_block = max_transactions_per_block
        self.pipeline_depth = 3  # 流水线深度
        self.current_leader = 0  # 当前领导者索引
    
    def validate_block(self, block: Block, previous_block: Optional[Block]) -> bool:
        """
        验证区块
        
        Args:
            block: 待验证的区块
            previous_block: 前一个区块
            
        Returns:
            bool: 验证结果
        """
        if previous_block and block.previous_hash != previous_block.hash:
            return False
        return True

    def create_block(self, node: Node, previous_block: Optional[Block]) -> Optional[Block]:
        """
        创建新区块
        
        Args:
            node: 节点对象
            previous_block: 前一个区块
            
        Returns:
            Block: 新创建的区块，如果无法创建则返回None
        """
        if not node.pending_transactions:
            return None
            
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
        """
        在节点间达成共识
        
        Args:
            nodes: 节点列表
            transactions: 交易列表
            transport: 网络传输对象
            
        Returns:
            Block: 达成共识的区块，如果没有达成则返回None
        """
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
    """
    Dumbo共识算法实现
    """
    
    def __init__(self, max_transactions_per_block: int = 256):
        """
        初始化Dumbo共识算法
        
        Args:
            max_transactions_per_block: 每个区块最大交易数
        """
        self.max_transactions_per_block = max_transactions_per_block
    
    def validate_block(self, block: Block, previous_block: Optional[Block]) -> bool:
        """
        验证区块
        
        Args:
            block: 待验证的区块
            previous_block: 前一个区块
            
        Returns:
            bool: 验证结果
        """
        if previous_block and block.previous_hash != previous_block.hash:
            return False
        return True

    def create_block(self, node: Node, previous_block: Optional[Block]) -> Optional[Block]:
        """
        创建新区块
        
        Args:
            node: 节点对象
            previous_block: 前一个区块
            
        Returns:
            Block: 新创建的区块，如果无法创建则返回None
        """
        if not node.pending_transactions:
            return None
            
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
        """
        在节点间达成共识
        
        Args:
            nodes: 节点列表
            transactions: 交易列表
            transport: 网络传输对象
            
        Returns:
            Block: 达成共识的区块，如果没有达成则返回None
        """
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