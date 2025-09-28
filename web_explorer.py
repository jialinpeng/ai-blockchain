import json
import threading
import time
from flask import Flask, render_template, jsonify, request
from core import ConsensusType, Transaction, Block
from network import NetworkProtocol
from simulator import BlockchainSimulator

app = Flask(__name__, static_folder='static', template_folder='templates')

# 全局变量存储模拟器实例和状态
simulator = None
is_running = False
simulation_thread = None
simulation_config = {}

# 存储实时数据
real_time_data = {
    'blocks': [],
    'tps_data': [],
    'confirmation_times': [],
    'network_topology': {},
    'stats': {}
}

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """获取或设置配置"""
    global simulator, simulation_config
    
    if request.method == 'POST':
        data = request.json
        simulation_config = {
            'node_count': int(data.get('node_count', 4)),
            'consensus_type': data.get('consensus_type', 'pow'),
            'network_protocol': data.get('network_protocol', 'direct'),
            'transaction_send_rate': float(data.get('transaction_send_rate', 1000.0)),
            'max_transactions_per_block': int(data.get('max_transactions_per_block', 256)),
            'transaction_size': int(data.get('transaction_size', 300)),
            'block_interval': float(data.get('block_interval', 3.0)),
            'network_bandwidth': int(data.get('network_bandwidth', 2500000)),
            'transaction_count': int(data.get('transaction_count', 1000)),
            'blocks_to_mine': int(data.get('blocks_to_mine', 10))
        }
        return jsonify({'status': 'success', 'config': simulation_config})
    
    return jsonify(simulation_config)

@app.route('/api/start', methods=['POST'])
def start_simulation():
    """启动模拟"""
    global simulator, is_running, simulation_thread
    
    if is_running:
        return jsonify({'status': 'error', 'message': '模拟已在运行中'})
    
    try:
        # 创建模拟器实例
        simulator = BlockchainSimulator(
            node_count=simulation_config.get('node_count', 4),
            consensus_type=ConsensusType(simulation_config.get('consensus_type', 'pow')),
            network_protocol=NetworkProtocol(simulation_config.get('network_protocol', 'direct')),
            transaction_send_rate=simulation_config.get('transaction_send_rate', 1000.0),
            max_transactions_per_block=simulation_config.get('max_transactions_per_block', 256),
            transaction_size=simulation_config.get('transaction_size', 300),
            block_interval=simulation_config.get('block_interval', 3.0),
            network_bandwidth=simulation_config.get('network_bandwidth', 2500000)
        )
        
        # 更新状态
        is_running = True
        
        # 在新线程中运行模拟
        simulation_thread = threading.Thread(
            target=run_simulation,
            args=(simulation_config.get('transaction_count', 1000), 
                  simulation_config.get('blocks_to_mine', 10)),
            daemon=True
        )
        simulation_thread.start()
        
        return jsonify({'status': 'success', 'message': '模拟已启动'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """停止模拟"""
    global is_running
    
    is_running = False
    return jsonify({'status': 'success', 'message': '模拟已停止'})

@app.route('/api/status')
def get_status():
    """获取当前状态"""
    return jsonify({
        'is_running': is_running,
        'config': simulation_config
    })

@app.route('/api/blocks')
def get_blocks():
    """获取区块信息"""
    if not simulator or not simulator.nodes:
        return jsonify({
            'blocks': [],
            'total': 0
        })
    
    # 收集所有节点的区块，确保完整性
    all_blocks = []
    processed_hashes = set()  # 用于跟踪已处理的区块哈希
    
    for node in simulator.nodes:
        for block in node.blockchain:
            # 检查是否已经处理过该区块
            if block.hash in processed_hashes:
                continue
            
            # 添加到已处理集合
            processed_hashes.add(block.hash)
            
            # 使用区块索引作为高度，确保连续性
            height = block.index
            
            # 查找挖矿节点（即添加该区块的节点）
            miner_id = None
            for n in simulator.nodes:
                # 检查节点的区块链中是否包含该区块，并且该区块是最新添加的
                if len(n.blockchain) > height and n.blockchain[height].hash == block.hash:
                    # 进一步检查该节点是否是挖矿节点（区块索引应该匹配）
                    if len(n.blockchain) == height + 1:
                        miner_id = n.node_id
                        break
            
            all_blocks.append({
                'height': height,
                'hash': block.hash,
                'previous_hash': block.previous_hash,
                'timestamp': block.timestamp,
                'transaction_count': len(block.transactions),
                'size': block.get_size(),
                'nonce': block.nonce,
                'miner_id': miner_id  # 添加矿工节点ID
            })
    
    # 按高度排序，确保连续性
    all_blocks.sort(key=lambda x: x['height'])
    
    # 分页处理
    page = request.args.get('page', 1, type=int)
    size = request.args.get('size', 20, type=int)  # 默认每页20个区块
    
    start = (page - 1) * size
    end = start + size
    
    paginated_blocks = all_blocks[start:end]
    
    return jsonify({
        'blocks': paginated_blocks,
        'total': len(all_blocks)
    })

@app.route('/api/stats')
def get_stats():
    """获取统计信息"""
    if not simulator:
        return jsonify({})
    
    stats = {
        'tps_history': simulator.block_tps_history,
        'confirmation_times': simulator.transaction_confirm_times,
        'node_count': len(simulator.nodes) if simulator.nodes else 0,
        'block_count': len(simulator.nodes[0].blockchain) if simulator.nodes else 0  # 添加区块总数
    }
    
    # 网络拓扑信息
    if simulator.topology:
        stats['topology'] = {
            'nodes': [node.node_id for node in simulator.nodes],
            'connections': dict(simulator.topology.adjacency_list)
        }
    
    return jsonify(stats)

@app.route('/api/network')
def get_network():
    """获取网络信息"""
    if not simulator or not simulator.topology:
        return jsonify({})
    
    network_data = {
        'nodes': [node.node_id for node in simulator.nodes],
        'connections': dict(simulator.topology.adjacency_list)
    }
    
    # 数据传输统计
    if simulator.transport:
        data_stats = simulator.transport.get_data_transfer_stats()
        network_data['data_transfer'] = data_stats
    
    return jsonify(network_data)

@app.route('/api/logs')
def get_logs():
    """获取传输日志"""
    if not simulator or not simulator.transport:
        return jsonify([])
    
    # 获取传输日志
    if hasattr(simulator.transport, 'get_transfer_logs'):
        logs = simulator.transport.get_transfer_logs()
        return jsonify(logs)
    
    return jsonify([])

def run_simulation(transaction_count, blocks_to_mine):
    """运行模拟（在单独线程中）"""
    global is_running
    
    try:
        simulator.run_simulation(transaction_count, blocks_to_mine)
    except Exception as e:
        print(f"模拟运行出错: {str(e)}")
    finally:
        is_running = False

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)