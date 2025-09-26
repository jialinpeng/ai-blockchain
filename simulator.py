import json
import time
import random
import os
import gzip
from typing import List, Dict, Optional

# 尝试导入绘图库
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Plotting functionality will be disabled.")

from core import ConsensusType, Transaction, Block, Node
from network import NetworkProtocol, NetworkTopology, NetworkTransport
from consensus import ConsensusAlgorithm, PoWConsensus, PBFTConsensus, HotStuffConsensus, DumboConsensus


class BlockchainSimulator:
    """
    区块链模拟器主类
    """
    
    def __init__(self, node_count: int = 4, consensus_type: ConsensusType = ConsensusType.POW, 
                 network_protocol: NetworkProtocol = NetworkProtocol.DIRECT,
                 transaction_send_rate: Optional[float] = None,
                 max_transactions_per_block: int = 256,
                 transaction_size: int = 300,
                 block_interval: float = 3.0,
                 network_bandwidth: int = 2500000):
        """
        初始化区块链模拟器
        
        Args:
            node_count: 节点数量
            consensus_type: 共识算法类型
            network_protocol: 网络协议类型
            transaction_send_rate: 交易发送速率 (txs/sec)
            max_transactions_per_block: 每个区块最大交易数
            transaction_size: 每笔交易大小（字节）
            block_interval: 出块间隔（秒），默认为3秒
            network_bandwidth: 网络带宽（字节/秒），默认为2500000字节/秒（20Mbps）
        """
        self.node_count = node_count
        self.consensus_type = consensus_type
        self.network_protocol = network_protocol
        self.transaction_send_rate = transaction_send_rate  # 交易发送速率 (txs/sec)
        self.max_transactions_per_block = max_transactions_per_block  # 每个区块最大交易数
        self.transaction_size = transaction_size  # 每笔交易大小（字节）
        self.block_interval = block_interval  # 出块间隔（秒）
        self.network_bandwidth = network_bandwidth  # 网络带宽（字节/秒）
        self.nodes: List[Node] = []
        self.topology: Optional[NetworkTopology] = None
        self.transport: Optional[NetworkTransport] = None
        self.consensus: Optional[ConsensusAlgorithm] = None
        self.block_tps_history: List[Dict] = []  # 存储区块TPS历史记录
        self.transaction_confirm_times: List[float] = []  # 存储交易确认时间
        
        # 初始化网络拓扑和传输层
        self.topology = NetworkTopology(self.nodes)
        self.topology.generate_topology(network_protocol)
        self.transport = NetworkTransport(self.topology, network_protocol)
        
        # 初始化共识算法
        if consensus_type == ConsensusType.POW:
            self.consensus = PoWConsensus(max_transactions_per_block=self.max_transactions_per_block)
        elif consensus_type == ConsensusType.PBFT:
            self.consensus = PBFTConsensus(max_transactions_per_block=self.max_transactions_per_block)
        elif consensus_type == ConsensusType.HOTSTUFF:
            self.consensus = HotStuffConsensus(max_transactions_per_block=self.max_transactions_per_block)
        elif consensus_type == ConsensusType.DUMBO:
            self.consensus = DumboConsensus(max_transactions_per_block=self.max_transactions_per_block)

        # 设置上一个区块的确认时间为模拟开始时间，以确保第一个区块也遵守出块间隔
        self._last_block_confirm_time = time.time()
        
        # 最后初始化节点
        self._initialize_nodes()

    def _initialize_nodes(self):
        """初始化节点"""
        self.nodes = []
        for i in range(self.node_count):
            node = Node(i, f"192.168.1.{i+1}", self.network_bandwidth)
            self.nodes.append(node)


    def set_consensus_algorithm(self, consensus_type: ConsensusType):
        """
        设置共识算法
        
        Args:
            consensus_type: 共识算法类型
        """
        self.consensus_type = consensus_type
        self._initialize_consensus()

    def set_network_protocol(self, protocol: NetworkProtocol):
        """
        设置网络传输协议
        
        Args:
            protocol: 网络协议类型
        """
        self.network_protocol = protocol
        if self.topology:
            self.topology.generate_topology(protocol)
            self.transport = NetworkTransport(self.topology, protocol)

    def set_topology(self, topology: NetworkTopology):
        """
        设置网络拓扑
        
        Args:
            topology: 网络拓扑对象
        """
        self.topology = topology
        self.transport = NetworkTransport(topology, self.network_protocol)

    def load_topology_from_json(self, file_path: str):
        """
        从JSON文件加载网络拓扑
        
        Args:
            file_path: JSON文件路径
        """
        self.topology = NetworkTopology(self.nodes)
        self.topology.load_from_json(file_path)
        self.transport = NetworkTransport(self.topology, self.network_protocol)

    def generate_transactions(self, count: int) -> List[Transaction]:
        """
        生成随机交易
        
        Args:
            count: 交易数量
            
        Returns:
            交易列表
        """
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
        """
        向指定节点提交交易
        
        Args:
            node_id: 节点ID
            transaction: 交易对象
        """
        if 0 <= node_id < len(self.nodes):
            self.nodes[node_id].add_transaction(transaction)

    def set_transaction_send_rate(self, rate: float):
        """
        设置交易发送速率（txs/sec）
        
        Args:
            rate: 交易发送速率
        """
        self.transaction_send_rate = rate

    def set_block_interval(self, interval: float):
        """
        设置出块间隔
        
        Args:
            interval: 出块间隔（秒）
        """
        self.block_interval = interval

    def _wait_for_next_block_interval(self, current_time: float):
        """
        等待到下一个区块的时间间隔点
        
        Args:
            current_time: 当前时间
        """
        if self.block_interval > 0:
            # 如果是第一个区块且_last_block_confirm_time为0，则使用当前时间作为起始点
            if self._last_block_confirm_time == 0:
                self._last_block_confirm_time = current_time
                # 对于第一个区块，等待一个完整的出块间隔
                time.sleep(self.block_interval)
            else:
                # 对于后续区块，计算并等待剩余时间
                elapsed_time = current_time - self._last_block_confirm_time
                wait_time = self.block_interval - elapsed_time
                
                if wait_time > 0:
                    time.sleep(wait_time)
                
    def _update_block_confirm_time(self, confirm_time: float):
        """
        更新区块确认时间
        
        Args:
            confirm_time: 区块确认时间
        """
        self._last_block_confirm_time = confirm_time

    def run_simulation(self, transaction_count: int = 100, blocks_to_mine: int = 10):
        """
        运行模拟
        
        Args:
            transaction_count: 交易数量
            blocks_to_mine: 需要挖掘的区块数量
        """
        print(f"Starting blockchain simulation with {self.node_count} nodes")
        print(f"Consensus: {self.consensus_type.value}")
        print(f"Network Protocol: {self.network_protocol.value}")
        
        # 生成交易
        transactions = self.generate_transactions(transaction_count)
        
        # 记录交易提交时间
        submit_times = {tx.tx_id: tx.timestamp for tx in transactions}
        
        # 只使用并发模式：一边发送交易一边进行共识
        if self.transaction_send_rate is not None:
            print(f"Sending transactions at rate: {self.transaction_send_rate} txs/sec")
        else:
            print("Transaction send rate not set, using default rate of 1000 txs/sec")
            self.transaction_send_rate = 1000.0
            
        self._send_transactions_and_mine_concurrently(transactions, submit_times, blocks_to_mine)
        
        print("Simulation completed")
        
        # 保存区块和交易的详细信息到文件
        self._save_block_details()
        
        # 绘制TPS图表和交易确认时间分布图
        self._plot_tps_chart()
        self._plot_confirmation_time_distribution()
        
        # 绘制节点间数据传输图表
        self._plot_node_data_transfer()

    def _send_transactions_and_mine_concurrently(self, transactions: List[Transaction], submit_times: Dict[str, float], blocks_to_mine: int):
        """
        并发发送交易和挖矿
        
        Args:
            transactions: 交易列表
            submit_times: 交易提交时间字典
            blocks_to_mine: 需要挖掘的区块数量
        """
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
        while sent_transactions < total_transactions and blocks_mined < blocks_to_mine:
            # 计算当前应该发送的交易批次
            batch_end = min(sent_transactions + max(1, int(self.transaction_send_rate / 10)), total_transactions)
            
            # 发送一批交易
            for i in range(sent_transactions, batch_end):
                # 检查是否已达到区块数量限制
                if blocks_mined >= blocks_to_mine:
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
                        # 检查是否已达到区块数量限制
                        if blocks_mined >= blocks_to_mine:
                            break
            
            # 如果已达到区块数量限制，则退出循环
            if blocks_mined >= blocks_to_mine:
                break
                
            # 定期检查所有节点是否可以创建区块
            # 同样确保挖矿频率不超过交易发送速率
            current_time = time.time()
            time_since_last_mine = current_time - last_mine_time
            min_time_between_mines = 1.0 / self.transaction_send_rate if self.transaction_send_rate else 0.1
            
            for node in self.nodes:
                if blocks_mined >= blocks_to_mine:
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
        
        while blocks_mined < blocks_to_mine:
            for node in self.nodes:
                if blocks_mined >= blocks_to_mine:
                    break
                if node.pending_transactions and time_since_last_mine >= min_time_between_mines:
                    block = self._mine_one_block(node, submit_times)
                    if block:
                        blocks_mined += 1
                        transactions_mined += len(block.transactions)
                        last_mine_time = time.time()
                        time_since_last_mine = 0
                        break
            
            # 如果本轮没有挖到区块，短暂等待再继续
            if blocks_mined < blocks_to_mine:
                time.sleep(0.01)
                time_since_last_mine = time.time() - last_mine_time

        print(f"Simulation finished. Blocks mined: {blocks_mined}, Transactions processed: {transactions_mined}")

    def _mine_one_block(self, node: Node, submit_times: Dict[str, float]):
        """
        挖掘一个区块
        
        Args:
            node: 节点对象
            submit_times: 交易提交时间字典
            
        Returns:
            Block: 挖掘出的区块，如果没有挖掘出则返回None
        """
        # 记录开始挖矿的时间
        mining_start_time = time.time()
        
        # 等待到下一个区块的时间间隔点
        self._wait_for_next_block_interval(mining_start_time)
        
        # 获取节点的待处理交易（不超过最大交易数限制）
        transactions_to_mine = node.pending_transactions[:self.max_transactions_per_block]
        
        # 从待处理交易中移除已选中的交易
        node.pending_transactions = node.pending_transactions[len(transactions_to_mine):]
        
        # 如果没有交易需要处理，则不创建区块
        if not transactions_to_mine:
            return None
        
        # 达成共识并创建区块
        block = self.consensus.reach_consensus(self.nodes, transactions_to_mine, self.transport)
        
        if block:
            # 记录区块确认时间
            block_confirm_time = time.time()
            
            # 计算TPS（基于区块间隔时间窗口）
            if self._last_block_confirm_time > 0 and self.block_interval > 0:
                # 使用区块间隔时间窗口计算TPS
                time_window = block_confirm_time - self._last_block_confirm_time
                tps = len(block.transactions) / time_window if time_window > 0 else 0
            else:
                # 使用区块处理时间计算TPS
                block_processing_time = block_confirm_time - mining_start_time
                tps = len(block.transactions) / block_processing_time if block_processing_time > 0 else 0
            
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
            
            # 更新上一个区块的确认时间
            self._update_block_confirm_time(block_confirm_time)
            
            # 记录区块信息
            block_info = {
                'block_height': len(self.nodes[0].blockchain) - 1,  # 区块高度为当前区块链长度减1
                'transaction_count': len(block.transactions),
                'block_size': block_size,
                'block_time': block_confirm_time - mining_start_time,  # 区块处理时间
                'tps': tps
            }
            self.block_tps_history.append(block_info)
            
            # 获取区块高度用于输出
            block_height = len(self.nodes[0].blockchain) - 1
            
            # 计算截至当前区块的总交易数
            total_transactions = sum(len(b.transactions) for b in self.nodes[0].blockchain)
            
            print(f"Block #{block_height} mined: {len(block.transactions)} transactions, "
                  f"Total: {total_transactions} transactions, "
                  f"Size: {block_size} bytes, Time: {block_confirm_time - mining_start_time:.2f}s, TPS: {tps:.2f}")
            
            return block
        return None

    def _plot_confirmation_time_distribution(self):
        """
        绘制交易确认时间CDF图并保存到results文件夹
        """
        if not HAS_MATPLOTLIB:
            print("Skipping confirmation time distribution plotting due to missing matplotlib")
            return
            
        if not self.transaction_confirm_times:
            print("No transaction confirmation times to plot")
            return
            
        # 创建results目录
        os.makedirs('results', exist_ok=True)
        
        # 计算CDF数据
        sorted_times = np.sort(self.transaction_confirm_times)
        y_vals = np.arange(len(sorted_times)) / float(len(sorted_times))
        
        # 绘制CDF图
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_times, y_vals, linewidth=2)
        plt.xlabel('Confirmation Time (seconds)')
        plt.ylabel('CDF')
        plt.title(f'Transaction Confirmation Time CDF ({self.consensus_type.value.upper()} Consensus)')
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        filename = f"results/confirmation_time_cdf_{self.consensus_type.value.lower()}_{self.network_protocol.value.lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图表，不显示
        
        print(f"Confirmation time CDF chart saved to {filename}")

    def _plot_node_data_transfer(self):
        """
        绘制节点间数据传输总量图表并保存到results文件夹
        """
        if not HAS_MATPLOTLIB:
            print("Skipping node data transfer plotting due to missing matplotlib")
            return
            
        if not self.transport:
            print("No transport layer available for data transfer plotting")
            return
            
        # 获取数据传输统计信息
        data_stats = self.transport.get_data_transfer_stats()
        
        # 如果没有数据传输统计信息，则跳过绘图
        if not data_stats:
            print("No data transfer statistics to plot")
            return
            
        # 创建results目录
        os.makedirs('results', exist_ok=True)
        
        # 准备绘图数据
        nodes = list(range(self.node_count))
        node_labels = [f'Node {i}' for i in nodes]
        
        # 创建一个矩阵来存储节点间的数据传输量（以交易数量为单位）
        transfer_matrix = np.zeros((self.node_count, self.node_count))
        
        # 填充传输矩阵
        for sender, receivers in data_stats.items():
            for receiver, amount in receivers.items():
                if sender >= 0 and receiver < self.node_count:  # 排除初始传输(-1)
                    transfer_matrix[sender][receiver] = amount
        
        # 将交易数量转换为数据量（MB）
        # 每个交易的大小由transaction_size参数确定，转换为MB单位
        data_matrix_mb = (transfer_matrix * self.transaction_size) / (1024 * 1024)
        
        # 创建热力图
        plt.figure(figsize=(10, 8))
        im = plt.imshow(data_matrix_mb, cmap='Blues', aspect='auto')
        
        # 判断是否显示详细信息（当节点数较少时显示）
        show_details = self.node_count <= 20
        
        if show_details:
            # 设置坐标轴标签
            plt.xticks(ticks=nodes, labels=node_labels, rotation=45)
            plt.yticks(ticks=nodes, labels=node_labels)
            
            # 在每个格子中添加数值标签（以MB为单位）
            for i in range(self.node_count):
                for j in range(self.node_count):
                    if data_matrix_mb[i, j] > 0:
                        # 显示MB单位的数值
                        data_amount_mb = data_matrix_mb[i, j]
                        text = plt.text(j, i, f'{data_amount_mb:.2f}',
                                      ha="center", va="center", 
                                      color="black" if data_matrix_mb[i, j] < np.max(data_matrix_mb)/2 else "white")
        else:
            # 节点数较多时，不显示具体的标签和数值
            plt.xticks(ticks=nodes[::max(1, len(nodes)//10)], labels=[])  # 只显示部分刻度，不显示标签
            plt.yticks(ticks=nodes[::max(1, len(nodes)//10)], labels=[])
        
        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('Data Transfer Amount (MB)', rotation=270, labelpad=20)
        
        # 设置图表标题和其他属性
        plt.title(f'Node Data Transfer Statistics ({self.consensus_type.value.upper()} Consensus, {self.network_protocol.value.upper()} Protocol)')
        plt.xlabel('Receiver Node')
        plt.ylabel('Sender Node')
        plt.tight_layout()
        
        # 保存图表
        filename = f"results/node_data_transfer_{self.consensus_type.value.lower()}_{self.network_protocol.value.lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图表，不显示
        
        print(f"Node data transfer chart saved to {filename}")

    def _save_block_details(self):
        """
        将区块和交易的详细信息以压缩JSON格式保存到data文件夹下的文件中
        """
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
        
        # 使用gzip压缩写入JSON文件
        filename = f"data/block_details_{self.consensus_type.value}_{self.network_protocol.value}.json.gz"
        with gzip.open(filename, 'wt', encoding='utf-8') as f:
            json.dump(data, f, separators=(',', ':'), ensure_ascii=False)
        
        print(f"区块详细信息已保存到 {filename} (已压缩)")

    def _plot_tps_chart(self):
        """
        绘制TPS图表并保存到results文件夹
        """
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