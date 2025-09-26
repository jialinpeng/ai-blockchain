from core import ConsensusType
from network import NetworkProtocol
from simulator import BlockchainSimulator


def interactive_setup():
    """
    交互式设置模拟器参数
    
    Returns:
        dict: 包含所有配置参数的字典
    """
    print("=" * 60)
    print("           区块链模拟器 - 由Linpeng Jia与通义灵码共同协作开发")
    print("=" * 60)
    print("本项目是一个教育和研究平台，用于理解不同共识机制在各种网络条件下的性能表现。")
    print()
    
    # 设置节点数量
    while True:
        try:
            node_count = int(input("请输入节点数量 (默认为20): ") or "20")
            if node_count > 0:
                break
            else:
                print("节点数量必须大于0，请重新输入。")
        except ValueError:
            print("请输入有效的整数。")
    
    # 设置共识算法
    consensus_options = {
        "1": ConsensusType.POW,
        "2": ConsensusType.PBFT,
        "3": ConsensusType.HOTSTUFF,
        "4": ConsensusType.DUMBO
    }
    print("\n可选的共识算法:")
    print("1. POW (工作量证明)")
    print("2. PBFT (实用拜占庭容错)")
    print("3. HOTSTUFF (HotStuff共识)")
    print("4. DUMBO (Dumbo共识)")
    
    while True:
        consensus_choice = input("请选择共识算法 (默认为1): ") or "1"
        if consensus_choice in consensus_options:
            consensus_type = consensus_options[consensus_choice]
            break
        else:
            print("无效选择，请输入1-4之间的数字。")
    
    # 设置网络协议
    protocol_options = {
        "1": NetworkProtocol.DIRECT,
        "2": NetworkProtocol.GOSSIP
    }
    print("\n可选的网络协议:")
    print("1. DIRECT (直接广播)")
    print("2. GOSSIP (Gossip协议)")
    
    while True:
        protocol_choice = input("请选择网络协议 (默认为2): ") or "2"
        if protocol_choice in protocol_options:
            network_protocol = protocol_options[protocol_choice]
            break
        else:
            print("无效选择，请输入1或2。")
    
    # 设置交易发送速率
    while True:
        try:
            rate_input = input("请输入交易发送速率 (txs/sec，输入0或回车表示不设置): ") or "0"
            transaction_send_rate = float(rate_input) if float(rate_input) > 0 else None
            break
        except ValueError:
            print("请输入有效的数字。")
    
    # 设置每个区块最大交易数
    while True:
        try:
            max_tx_per_block = int(input("请输入每个区块最大交易数 (默认为256): ") or "256")
            if max_tx_per_block > 0:
                break
            else:
                print("交易数必须大于0，请重新输入。")
        except ValueError:
            print("请输入有效的整数。")
    
    # 设置每笔交易大小
    while True:
        try:
            transaction_size = int(input("请输入每笔交易大小 (字节，默认为300): ") or "300")
            if transaction_size > 0:
                break
            else:
                print("交易大小必须大于0，请重新输入。")
        except ValueError:
            print("请输入有效的整数。")
    
    # 设置出块间隔
    while True:
        try:
            block_interval = float(input("请输入出块间隔 (秒，默认为0表示无间隔): ") or "0")
            if block_interval >= 0:
                break
            else:
                print("出块间隔必须大于等于0，请重新输入。")
        except ValueError:
            print("请输入有效的数字。")
    
    # 设置交易数量
    while True:
        try:
            transaction_count = int(input("请输入模拟交易总数 (默认为10000): ") or "10000")
            if transaction_count > 0:
                break
            else:
                print("交易总数必须大于0，请重新输入。")
        except ValueError:
            print("请输入有效的整数。")
    
    # 设置挖掘区块数
    while True:
        try:
            blocks_to_mine = int(input("请输入需要挖掘的区块数 (默认为50): ") or "50")
            if blocks_to_mine > 0:
                break
            else:
                print("区块数必须大于0，请重新输入。")
        except ValueError:
            print("请输入有效的整数。")
    
    return {
        "node_count": node_count,
        "consensus_type": consensus_type,
        "network_protocol": network_protocol,
        "transaction_send_rate": transaction_send_rate,
        "max_transactions_per_block": max_tx_per_block,
        "transaction_size": transaction_size,
        "block_interval": block_interval,
        "transaction_count": transaction_count,
        "blocks_to_mine": blocks_to_mine
    }


# 使用示例
if __name__ == "__main__":
    print("=" * 60)
    print("           区块链模拟器 - 由Linpeng Jia与通义灵码共同协作开发")
    print("=" * 60)
    print("1. 交互式设置参数")
    print("2. 使用默认参数")
    
    choice = input("请选择启动方式 (默认为1): ") or "1"
    
    if choice == "2":
        # 使用默认参数
        simulator = BlockchainSimulator(
            node_count=20,
            consensus_type=ConsensusType.POW,
            network_protocol=NetworkProtocol.GOSSIP,
            transaction_send_rate=1000.0,  # 1000 transactions per second
            max_transactions_per_block=256,  # 每个区块最多256笔交易
            transaction_size=300,  # 每笔交易300字节
            block_interval=0.0  # 无出块间隔
        )
        
        # 运行模拟
        simulator.run_simulation(transaction_count=10000, blocks_to_mine=50)
    else:
        # 交互式设置参数
        params = interactive_setup()
        
        # 创建模拟器实例
        simulator = BlockchainSimulator(
            node_count=params["node_count"],
            consensus_type=params["consensus_type"],
            network_protocol=params["network_protocol"],
            transaction_send_rate=params["transaction_send_rate"],
            max_transactions_per_block=params["max_transactions_per_block"],
            transaction_size=params["transaction_size"],
            block_interval=params["block_interval"]
        )
        
        # 运行模拟
        simulator.run_simulation(
            transaction_count=params["transaction_count"],
            blocks_to_mine=params["blocks_to_mine"]
        )