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
    
    def __init__(self, max_transactions_per_block: int = 256):
        self.max_transactions_per_block = max_transactions_per_block

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
    工作量证明共识算法实现（比特币风格）
    """
    
    def __init__(self, difficulty: int = 4, max_transactions_per_block: int = 256):
        """
        初始化PoW共识算法
        
        Args:
            difficulty: 挖矿难度（前导零的个数）
            max_transactions_per_block: 每个区块最大交易数
        """
        self.difficulty = difficulty
        self.max_transactions_per_block = max_transactions_per_block
        # 构造目标值，例如难度为4时，目标值为0x0000FFFF...FFFF
        self.target = (1 << (256 - difficulty * 4)) - 1

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
            
        # 验证区块哈希是否满足难度要求
        block_header = self._construct_block_header(block)
        block_hash = hashlib.sha256(hashlib.sha256(block_header.encode()).digest()).hexdigest()
        hash_value = int(block_hash, 16)
        
        return hash_value <= self.target and block_hash == block.hash

    def create_block(self, node: Node, previous_block: Optional[Block]) -> Optional[Block]:
        """
        创建新区块（比特币风格的PoW）
        
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
        timestamp = int(time.time())
        
        # 创建区块（初始nonce为0）
        block = Block(len(node.blockchain), previous_hash, timestamp, transactions, 0)
        
        # 进行工作量证明计算
        nonce = 0
        block_hash = None
        
        while True:
            # 构造区块头
            block_header = self._construct_block_header(block, nonce)
            
            # 双重SHA256哈希
            hash_result = hashlib.sha256(hashlib.sha256(block_header.encode()).digest()).hexdigest()
            hash_value = int(hash_result, 16)
            
            # 检查是否满足难度要求
            if hash_value <= self.target:
                block_hash = hash_result
                break
                
            # 增加nonce并继续计算
            nonce += 1
            block.nonce = nonce
            
            # 每计算10000次暂停一下，避免占用过多CPU
            if nonce % 10000 == 0:
                time.sleep(0.001)
        
        # 设置最终的哈希值和nonce
        block.hash = block_hash
        block.nonce = nonce
        return block

    def _construct_block_header(self, block: Block, nonce: int = None) -> str:
        """
        构造区块头字符串
        
        Args:
            block: 区块对象
            nonce: 随机数
            
        Returns:
            str: 区块头字符串
        """
        nonce = nonce if nonce is not None else block.nonce
        tx_data = "".join([tx.tx_id for tx in block.transactions])
        block_header = f"{block.index}{block.previous_hash}{block.timestamp}{tx_data}{nonce}"
        return block_header

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
            print("[PoW] 无待处理交易，跳过挖矿")
            return None
            
        print(f"[PoW] 开始工作量证明共识，节点数: {len(nodes)}, 交易数: {len(transactions)}")
        
        # 使用传输层广播交易
        transport.broadcast_transactions(nodes, transactions)
        print(f"[PoW] 已广播 {len(transactions)} 笔交易到网络")
            
        # 模拟挖矿过程
        print("[PoW] 各节点开始挖矿...")
        for i, node in enumerate(nodes):
            previous_block = node.get_latest_block()
            block = self.create_block(node, previous_block)
            if block:
                print(f"[PoW] 节点 {node.node_id} 成功挖矿，nonce: {block.nonce}")
                print(f"[PoW] 区块哈希: {block.hash[:16]}...")
                return block
                
        print("[PoW] 所有节点挖矿失败")
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
            print("[PBFT] 无待处理交易，跳过共识")
            return None
            
        print(f"[PBFT] 开始实用拜占庭容错共识，节点数: {len(nodes)}, 交易数: {len(transactions)}")
        
        # 使用传输层广播交易
        transport.broadcast_transactions(nodes, transactions)
        print(f"[PBFT] 已广播 {len(transactions)} 笔交易到网络")
        
        # 模拟网络传输延迟
        time.sleep(0.05)  # 50ms网络延迟
        
        # 选择主节点（简化版PBFT）
        primary_node = nodes[0]
        print(f"[PBFT] 主节点: {primary_node.node_id}")
        
        # 预准备阶段
        print("[PBFT] 预准备阶段 - 主节点广播提案")
        time.sleep(0.02)
        
        # 准备阶段
        print("[PBFT] 准备阶段 - 节点验证并广播准备消息")
        prepared_count = len(nodes)  # 简化实现，假设所有节点都准备就绪
        time.sleep(0.02)
        print(f"[PBFT] {prepared_count}/{len(nodes)} 节点完成准备")
        
        # 提交阶段
        print("[PBFT] 提交阶段 - 节点收集准备消息并广播提交消息")
        committed_count = len(nodes)  # 简化实现，假设所有节点都提交
        time.sleep(0.02)
        print(f"[PBFT] {committed_count}/{len(nodes)} 节点完成提交")
        
        # 决策
        previous_block = primary_node.get_latest_block()
        block = self.create_block(primary_node, previous_block)
        if block:
            print(f"[PBFT] 共识达成，区块高度: {block.index}, 区块哈希: {block.hash[:16]}...")
        else:
            print("[PBFT] 共识失败")
            
        return block


class HotStuffConsensus(ConsensusAlgorithm):
    """
    HotStuff共识算法实现
    """
    
    def __init__(self, max_transactions_per_block: int = 256, pipeline_depth: int = 3):
        """
        初始化HotStuff共识算法
        
        Args:
            max_transactions_per_block: 每个区块最大交易数
            pipeline_depth: 流水线深度
        """
        super().__init__(max_transactions_per_block)
        self.pipeline_depth = pipeline_depth
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
            print("[HotStuff] 无待处理交易，跳过共识")
            return None
            
        print(f"[HotStuff] 开始HotStuff共识，节点数: {len(nodes)}, 交易数: {len(transactions)}")
        
        # 使用传输层广播交易
        transport.broadcast_transactions(nodes, transactions)
        print(f"[HotStuff] 已广播 {len(transactions)} 笔交易到网络")
        
        # 模拟网络传输延迟
        time.sleep(0.03)  # 30ms网络延迟
        
        # HotStuff实现：领导者轮换
        leader_node = nodes[self.current_leader]
        print(f"[HotStuff] 当前领导者: 节点 {leader_node.node_id}")
        
        # 提议阶段
        print("[HotStuff] 领导者提议新区块")
        time.sleep(0.02)
        
        # 投票阶段
        print("[HotStuff] 节点对提议进行投票")
        vote_count = len(nodes)  # 简化实现，假设所有节点都投票
        time.sleep(0.02)
        print(f"[HotStuff] 收到 {vote_count} 票")
        
        # 提交阶段
        print("[HotStuff] 达成2f+1投票，提交区块")
        time.sleep(0.02)
        
        previous_block = leader_node.get_latest_block()
        block = self.create_block(leader_node, previous_block)
        
        # 轮换领导者
        self.current_leader = (self.current_leader + 1) % len(nodes)
        print(f"[HotStuff] 领导者轮换至节点 {self.current_leader}")
        
        if block:
            print(f"[HotStuff] 共识达成，区块高度: {block.index}, 区块哈希: {block.hash[:16]}...")
        else:
            print("[HotStuff] 共识失败")
            
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