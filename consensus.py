import hashlib
import time
import random
import threading
import queue
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from core import Block, Transaction, Node


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
    def reach_consensus(self, nodes: List[Node], transactions: List[Transaction], transport: 'NetworkTransport') -> Optional[Block]:
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
        # 检查前一个区块哈希是否匹配
        if previous_block and block.previous_hash != previous_block.hash:
            return False
            
        # 验证区块哈希是否满足难度要求
        block_header = self._construct_block_header(block)
        block_hash = hashlib.sha256(hashlib.sha256(block_header.encode()).digest()).hexdigest()
        hash_value = int(block_hash, 16)
        
        # 检查工作量证明是否有效
        if hash_value > self.target or block_hash != block.hash:
            return False
            
        # 验证通过
        return True

    def create_block(self, node: Node, previous_block: Optional[Block]) -> Optional[Block]:
        """
        创建新区块（比特币风格的PoW）
        
        Args:
            node: 节点对象
            previous_block: 前一个区块
            
        Returns:
            Block: 新创建的区块，如果无法创建则返回None
        """
        # 即使没有交易也要创建区块（空区块机制）
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

    def reach_consensus(self, nodes: List[Node], transactions: List[Transaction], transport: 'NetworkTransport') -> Optional[Block]:
        """
        在节点间达成共识（使用并发挖矿方式）
        
        Args:
            nodes: 节点列表
            transactions: 交易列表
            transport: 网络传输对象
            
        Returns:
            Block: 达成共识的区块，如果没有达成则返回None
        """
        # 不管有没有交易都继续执行（支持空块）
        print(f"[PoW] 开始工作量证明共识，节点数: {len(nodes)}, 交易数: {len(transactions)}")
        
        # 使用传输层广播交易
        if transport and transactions:
            # 选择一个随机节点作为发送者
            sender_node = random.choice(nodes)
            transport.broadcast_transactions(sender_node.node_id, transactions)
            print(f"[PoW] 已广播 {len(transactions)} 笔交易到网络")
            
        # 将交易分发给所有节点
        if transactions:
            for node in nodes:
                node.pending_transactions.extend(transactions)
            
        # 模拟挖矿过程 - 所有节点同时开始挖矿
        print("[PoW] 各节点开始挖矿...")
        
        # 使用线程让所有节点同时挖矿
        result_queue = queue.Queue()
        threads = []
        
        def mine_block(node, result_queue):
            previous_block = node.get_latest_block()
            block = self.create_block(node, previous_block)
            if block:
                result_queue.put((node, block))
        
        # 启动所有节点的挖矿线程
        for i, node in enumerate(nodes):
            thread = threading.Thread(target=mine_block, args=(node, result_queue))
            threads.append(thread)
            thread.start()
        
        # 等待第一个完成的节点
        winner_node = None
        winner_block = None
        try:
            winner_node, winner_block = result_queue.get(timeout=30)  # 30秒超时
            print(f"[PoW] 节点 {winner_node.node_id} 成功挖矿，nonce: {winner_block.nonce}")
            print(f"[PoW] 区块哈希: {winner_block.hash[:16]}...")
        except queue.Empty:
            print("[PoW] 挖矿超时，没有节点成功挖到区块")
        
        # 等待所有线程结束
        for thread in threads:
            thread.join()
        
        # 将获胜区块添加到获胜节点的区块链中（这样我们就能通过区块链识别挖矿节点）
        if winner_node and winner_block:
            winner_node.add_block(winner_block)
            
            # 广播获胜区块到所有其他节点
            if transport:
                transport.broadcast_block(winner_node.node_id, winner_block)
                
            # 其他节点接收并验证区块
            for node in nodes:
                if node != winner_node:
                    # 验证区块
                    latest_block = node.get_latest_block()
                    if self.validate_block(winner_block, latest_block):
                        # 验证通过，添加到区块链
                        node.add_block(winner_block)
                        print(f"[PoW] 节点 {node.node_id} 接受新区块 {winner_block.index}")
                    else:
                        print(f"[PoW] 节点 {node.node_id} 拒绝新区块 {winner_block.index}（验证失败）")
        
        return winner_block


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
        # 即使没有交易也要创建区块（空区块机制）
        transactions = node.pending_transactions[:self.max_transactions_per_block]
        previous_hash = previous_block.hash if previous_block else "0" * 64
        timestamp = time.time()
        
        block = Block(len(node.blockchain), previous_hash, timestamp, transactions)
        return block

    def reach_consensus(self, nodes: List[Node], transactions: List[Transaction], transport: 'NetworkTransport') -> Optional[Block]:
        """
        在节点间达成共识
        
        Args:
            nodes: 节点列表
            transactions: 交易列表
            transport: 网络传输对象
            
        Returns:
            Block: 达成共识的区块，如果没有达成则返回None
        """
        # 不管有没有交易都继续执行（支持空块）
        print(f"[PBFT] 开始实用拜占庭容错共识，节点数: {len(nodes)}, 交易数: {len(transactions)}")
        
        # 使用传输层广播交易
        if transport and transactions:
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
            
            # 广播区块到所有节点
            if transport:
                transport.broadcast_block(primary_node.node_id, block)
            
            # 所有节点接收并验证区块
            for node in nodes:
                if self.validate_block(block, node.get_latest_block()):
                    node.add_block(block)
                    print(f"[PBFT] 节点 {node.node_id} 添加新区块 {block.index}")
                else:
                    print(f"[PBFT] 节点 {node.node_id} 拒绝新区块 {block.index}（验证失败）")
            
            return block
        else:
            print("[PBFT] 共识失败，无法创建区块")
            
        return None


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
        self.leader_index = 0  # 领导者节点索引
    
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
        # 即使没有交易也要创建区块（空区块机制）
        transactions = node.pending_transactions[:self.max_transactions_per_block]
        previous_hash = previous_block.hash if previous_block else "0" * 64
        timestamp = time.time()
        
        block = Block(len(node.blockchain), previous_hash, timestamp, transactions)
        return block

    def reach_consensus(self, nodes: List[Node], transactions: List[Transaction], transport: 'NetworkTransport') -> Optional[Block]:
        """
        在节点间达成共识
        
        Args:
            nodes: 节点列表
            transactions: 交易列表
            transport: 网络传输对象
            
        Returns:
            Block: 达成共识的区块，如果没有达成则返回None
        """
        # 不管有没有交易都继续执行（支持空块）
        print(f"[HotStuff] 开始HotStuff共识，节点数: {len(nodes)}, 交易数: {len(transactions)}")
        
        # 轮换领导者
        leader_node = nodes[self.leader_index % len(nodes)]
        self.leader_index += 1
        print(f"[HotStuff] 领导者节点: {leader_node.node_id}")
        
        # 领导者创建区块
        previous_block = leader_node.get_latest_block()
        block = self.create_block(leader_node, previous_block)
        
        if block:
            print(f"[HotStuff] 区块创建成功，区块高度: {block.index}, 区块哈希: {block.hash[:16]}...")
            
            # 广播区块到所有节点
            if transport:
                transport.broadcast_block(leader_node.node_id, block)
            
            # 所有节点接收并验证区块
            for node in nodes:
                if self.validate_block(block, node.get_latest_block()):
                    node.add_block(block)
                    print(f"[HotStuff] 节点 {node.node_id} 添加新区块 {block.index}")
                else:
                    print(f"[HotStuff] 节点 {node.node_id} 拒绝新区块 {block.index}（验证失败）")
            
            return block
        else:
            print("[HotStuff] 区块创建失败")
            
        return None


class DumboConsensus(ConsensusAlgorithm):
    """
    Dumbo共识算法实现（简化版）
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
        # 即使没有交易也要创建区块（空区块机制）
        transactions = node.pending_transactions[:self.max_transactions_per_block]
        previous_hash = previous_block.hash if previous_block else "0" * 64
        timestamp = time.time()
        
        # 模拟Dumbo的处理时间（异步共识）
        # 根据交易数量添加适当的延迟
        processing_time = len(transactions) * 0.002  # 每个交易0.002秒处理时间
        time.sleep(processing_time)
        
        block = Block(len(node.blockchain), previous_hash, timestamp, transactions)
        return block

    def reach_consensus(self, nodes: List[Node], transactions: List[Transaction], transport: 'NetworkTransport') -> Optional[Block]:
        """
        在节点间达成共识
        
        Args:
            nodes: 节点列表
            transactions: 交易列表
            transport: 网络传输对象
            
        Returns:
            Block: 达成共识的区块，如果没有达成则返回None
        """
        # 不管有没有交易都继续执行（支持空块）
        print(f"[Dumbo] 开始Dumbo共识，节点数: {len(nodes)}, 交易数: {len(transactions)}")
        
        # 使用传输层广播交易
        if transport and transactions:
            transport.broadcast_transactions(nodes, transactions)
            print(f"[Dumbo] 已广播 {len(transactions)} 笔交易到网络")
        
        # 模拟网络传输延迟
        time.sleep(0.1)  # 100ms网络延迟
        
        # Dumbo简化实现：选择第一个节点作为领导者
        leader_node = nodes[0]
        previous_block = leader_node.get_latest_block()
        block = self.create_block(leader_node, previous_block)
        
        if block:
            print(f"[Dumbo] 共识达成，区块高度: {block.index}, 区块哈希: {block.hash[:16]}...")
            
            # 广播区块到所有节点
            if transport:
                transport.broadcast_block(leader_node.node_id, block)
            
            # 所有节点接收并验证区块
            for node in nodes:
                if self.validate_block(block, node.get_latest_block()):
                    node.add_block(block)
                    print(f"[Dumbo] 节点 {node.node_id} 添加新区块 {block.index}")
                else:
                    print(f"[Dumbo] 节点 {node.node_id} 拒绝新区块 {block.index}（验证失败）")
            
            return block
        else:
            print("[Dumbo] 共识失败，无法创建区块")
            
        return None