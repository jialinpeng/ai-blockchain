import unittest
import time
from core import ConsensusType, Transaction, Block, Node


class TestTransaction(unittest.TestCase):
    """测试Transaction类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.timestamp = time.time()
        self.tx = Transaction(
            tx_id="tx_001",
            sender="Alice",
            receiver="Bob",
            amount=10.5,
            timestamp=self.timestamp,
            size=300
        )
    
    def test_transaction_initialization(self):
        """测试交易初始化"""
        self.assertEqual(self.tx.tx_id, "tx_001")
        self.assertEqual(self.tx.sender, "Alice")
        self.assertEqual(self.tx.receiver, "Bob")
        self.assertEqual(self.tx.amount, 10.5)
        self.assertEqual(self.tx.timestamp, self.timestamp)
        self.assertEqual(self.tx.size, 300)
    
    def test_transaction_to_dict(self):
        """测试交易转字典方法"""
        tx_dict = self.tx.to_dict()
        self.assertIsInstance(tx_dict, dict)
        self.assertEqual(tx_dict['tx_id'], "tx_001")
        self.assertEqual(tx_dict['sender'], "Alice")
        self.assertEqual(tx_dict['receiver'], "Bob")
        self.assertEqual(tx_dict['amount'], 10.5)
        self.assertEqual(tx_dict['timestamp'], self.timestamp)
    
    def test_transaction_get_size(self):
        """测试获取交易大小方法"""
        self.assertEqual(self.tx.get_size(), 300)


class TestBlock(unittest.TestCase):
    """测试Block类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.timestamp = time.time()
        self.transactions = [
            Transaction("tx_001", "Alice", "Bob", 10.5, self.timestamp),
            Transaction("tx_002", "Bob", "Charlie", 5.2, self.timestamp)
        ]
        self.block = Block(
            index=1,
            previous_hash="0" * 64,
            timestamp=self.timestamp,
            transactions=self.transactions,
            nonce=12345
        )
    
    def test_block_initialization(self):
        """测试区块初始化"""
        self.assertEqual(self.block.index, 1)
        self.assertEqual(self.block.previous_hash, "0" * 64)
        self.assertEqual(self.block.timestamp, self.timestamp)
        self.assertEqual(self.block.transactions, self.transactions)
        self.assertEqual(self.block.nonce, 12345)
        self.assertIsNotNone(self.block.hash)
    
    def test_block_calculate_hash(self):
        """测试区块哈希计算"""
        hash_value = self.block.calculate_hash()
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 64)  # SHA256哈希长度
    
    def test_block_to_dict(self):
        """测试区块转字典方法"""
        block_dict = self.block.to_dict()
        self.assertIsInstance(block_dict, dict)
        self.assertEqual(block_dict['index'], 1)
        self.assertEqual(block_dict['previous_hash'], "0" * 64)
        self.assertEqual(len(block_dict['transactions']), 2)
        self.assertEqual(block_dict['nonce'], 12345)
    
    def test_block_get_size(self):
        """测试获取区块大小方法"""
        # 区块头大小(100) + 两笔交易大小(各300字节)
        expected_size = 100 + 300 + 300
        self.assertEqual(self.block.get_size(), expected_size)


class TestNode(unittest.TestCase):
    """测试Node类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.node = Node(
            node_id=1,
            ip_address="192.168.1.1",
            network_bandwidth=2500000
        )
    
    def test_node_initialization(self):
        """测试节点初始化"""
        self.assertEqual(self.node.node_id, 1)
        self.assertEqual(self.node.ip_address, "192.168.1.1")
        self.assertEqual(self.node.network_bandwidth, 2500000)
        self.assertIsInstance(self.node.blockchain, list)
        self.assertIsInstance(self.node.pending_transactions, list)
        self.assertEqual(len(self.node.blockchain), 0)
        self.assertEqual(len(self.node.pending_transactions), 0)
    
    def test_node_add_transaction(self):
        """测试节点添加交易"""
        tx = Transaction("tx_001", "Alice", "Bob", 10.5, time.time())
        self.node.add_transaction(tx)
        self.assertEqual(len(self.node.pending_transactions), 1)
        self.assertEqual(self.node.pending_transactions[0], tx)
    
    def test_node_add_block(self):
        """测试节点添加区块"""
        timestamp = time.time()
        transactions = [Transaction("tx_001", "Alice", "Bob", 10.5, timestamp)]
        block = Block(0, "0" * 64, timestamp, transactions)
        self.node.add_block(block)
        self.assertEqual(len(self.node.blockchain), 1)
        self.assertEqual(self.node.blockchain[0], block)
    
    def test_node_get_latest_block(self):
        """测试获取最新区块"""
        # 空区块链
        self.assertIsNone(self.node.get_latest_block())
        
        # 添加一个区块后
        timestamp = time.time()
        transactions = [Transaction("tx_001", "Alice", "Bob", 10.5, timestamp)]
        block = Block(0, "0" * 64, timestamp, transactions)
        self.node.add_block(block)
        latest_block = self.node.get_latest_block()
        self.assertEqual(latest_block, block)


class TestConsensusType(unittest.TestCase):
    """测试ConsensusType枚举"""
    
    def test_consensus_type_values(self):
        """测试共识类型枚举值"""
        self.assertEqual(ConsensusType.POW.value, "pow")
        self.assertEqual(ConsensusType.PBFT.value, "pbft")
        self.assertEqual(ConsensusType.HOTSTUFF.value, "hotstuff")
        self.assertEqual(ConsensusType.DUMBO.value, "dumbo")


if __name__ == '__main__':
    unittest.main()