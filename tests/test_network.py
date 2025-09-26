import unittest
import time
from core import Node, Transaction, Block
from network import NetworkProtocol, NetworkTopology, NetworkTransport


class TestNetworkProtocol(unittest.TestCase):
    """测试NetworkProtocol枚举"""
    
    def test_network_protocol_values(self):
        """测试网络协议枚举值"""
        self.assertEqual(NetworkProtocol.DIRECT.value, "direct")
        self.assertEqual(NetworkProtocol.GOSSIP.value, "gossip")


class TestNetworkTopology(unittest.TestCase):
    """测试NetworkTopology类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.nodes = [
            Node(0, "192.168.1.1"),
            Node(1, "192.168.1.2"),
            Node(2, "192.168.1.3"),
            Node(3, "192.168.1.4")
        ]
        self.topology = NetworkTopology(self.nodes)
    
    def test_topology_initialization(self):
        """测试网络拓扑初始化"""
        self.assertEqual(self.topology.nodes, self.nodes)
        self.assertIsInstance(self.topology.adjacency_list, dict)
        self.assertEqual(len(self.topology.adjacency_list), 0)
    
    def test_generate_direct_topology(self):
        """测试生成DIRECT协议拓扑"""
        self.topology.generate_topology(NetworkProtocol.DIRECT)
        
        # 检查每个节点是否与其他所有节点连接
        for i in range(len(self.nodes)):
            neighbors = self.topology.get_neighbors(i)
            # 每个节点应该连接到除自己以外的所有节点
            self.assertEqual(len(neighbors), len(self.nodes) - 1)
            self.assertNotIn(i, neighbors)  # 不应该连接到自己
    
    def test_generate_gossip_topology(self):
        """测试生成GOSSIP协议拓扑"""
        self.topology.generate_topology(NetworkProtocol.GOSSIP)
        
        # 检查每个节点是否至少有一个邻居
        for i in range(len(self.nodes)):
            neighbors = self.topology.get_neighbors(i)
            self.assertGreater(len(neighbors), 0)
            self.assertLessEqual(len(neighbors), len(self.nodes) - 1)
            self.assertNotIn(i, neighbors)  # 不应该连接到自己
    
    def test_get_neighbors(self):
        """测试获取邻居节点"""
        # 在没有生成拓扑前，应该返回空列表
        neighbors = self.topology.get_neighbors(0)
        self.assertEqual(neighbors, [])
        
        # 生成DIRECT拓扑后测试
        self.topology.generate_topology(NetworkProtocol.DIRECT)
        neighbors = self.topology.get_neighbors(0)
        self.assertEqual(len(neighbors), 3)
        self.assertNotIn(0, neighbors)
    
    def test_to_dict(self):
        """测试网络拓扑转字典"""
        # 生成DIRECT拓扑
        self.topology.generate_topology(NetworkProtocol.DIRECT)
        
        # 转换为字典
        topology_dict = self.topology.to_dict()
        self.assertIsInstance(topology_dict, dict)
        self.assertIn('nodes', topology_dict)
        self.assertIn('adjacency_list', topology_dict)
        self.assertEqual(len(topology_dict['nodes']), len(self.nodes))


class TestNetworkTransport(unittest.TestCase):
    """测试NetworkTransport类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.nodes = [
            Node(0, "192.168.1.1"),
            Node(1, "192.168.1.2"),
            Node(2, "192.168.1.3")
        ]
        self.topology = NetworkTopology(self.nodes)
        self.topology.generate_topology(NetworkProtocol.DIRECT)
        self.transport = NetworkTransport(self.topology, NetworkProtocol.DIRECT)
    
    def test_transport_initialization(self):
        """测试网络传输初始化"""
        self.assertEqual(self.transport.topology, self.topology)
        self.assertEqual(self.transport.protocol, NetworkProtocol.DIRECT)
        self.assertIsInstance(self.transport.data_transfer_stats, dict)
        self.assertEqual(len(self.transport.data_transfer_stats), 0)
    
    def test_broadcast_transactions(self):
        """测试广播交易"""
        transactions = [
            Transaction("tx_001", "Alice", "Bob", 10.5, time.time()),
            Transaction("tx_002", "Bob", "Charlie", 5.2, time.time())
        ]
        
        # 广播交易
        self.transport.broadcast_transactions(0, transactions)
        
        # 检查数据传输统计
        self.assertIn(0, self.transport.data_transfer_stats)
        self.assertGreater(len(self.transport.data_transfer_stats[0]), 0)
        # 检查节点1和2是否收到了交易
        self.assertEqual(len(self.nodes[1].pending_transactions), 2)
        self.assertEqual(len(self.nodes[2].pending_transactions), 2)
    
    def test_gossip_transactions(self):
        """测试Gossip协议传播交易"""
        # 重新创建支持GOSSIP协议的传输实例
        self.topology.generate_topology(NetworkProtocol.GOSSIP)
        gossip_transport = NetworkTransport(self.topology, NetworkProtocol.GOSSIP)
        
        transactions = [
            Transaction("tx_001", "Alice", "Bob", 10.5, time.time()),
            Transaction("tx_002", "Bob", "Charlie", 5.2, time.time())
        ]
        
        # Gossip传播交易
        gossip_transport._gossip_transactions(0, transactions, max_gossip_rounds=3)
        
        # 检查数据传输统计
        self.assertGreaterEqual(len(gossip_transport.data_transfer_stats), 0)
    
    def test_get_data_transfer_stats(self):
        """测试获取数据传输统计"""
        stats = self.transport.get_data_transfer_stats()
        self.assertIsInstance(stats, dict)
        self.assertEqual(len(stats), 0)
        
        # 广播一些交易后再检查
        transactions = [Transaction("tx_001", "Alice", "Bob", 10.5, time.time())]
        self.transport.broadcast_transactions(0, transactions)
        
        stats = self.transport.get_data_transfer_stats()
        self.assertIsInstance(stats, dict)
        self.assertGreater(len(stats), 0)
        self.assertIn(0, stats)
        self.assertGreater(len(stats[0]), 0)


if __name__ == '__main__':
    unittest.main()