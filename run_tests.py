#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
区块链模拟器测试运行脚本
========================

本脚本用于运行区块链模拟器项目的所有单元测试。

使用方法:
    python run_tests.py                    # 运行所有测试
    python run_tests.py -v                 # 详细模式运行所有测试
    python run_tests.py test_core          # 运行core模块测试
    python run_tests.py test_network       # 运行network模块测试
"""

import sys
import unittest
import argparse
import os


def discover_and_run_tests(test_module=None, verbosity=1):
    """
    发现并运行测试
    
    Args:
        test_module (str): 要测试的模块名称，如果为None则运行所有测试
        verbosity (int): 测试输出详细程度
    """
    # 添加项目根目录到Python路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # 创建测试加载器
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if test_module:
        # 运行指定模块的测试
        try:
            # 尝试导入测试模块
            module = __import__(f'tests.{test_module}', fromlist=[test_module])
            suite.addTests(loader.loadTestsFromModule(module))
        except ImportError as e:
            print(f"无法导入测试模块 {test_module}: {e}")
            return False
    else:
        # 自动发现所有测试
        test_dir = os.path.join(project_root, 'tests')
        if os.path.exists(test_dir):
            suite.addTests(loader.discover(test_dir, pattern='test_*.py'))
        else:
            print(f"测试目录 {test_dir} 不存在")
            return False
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # 返回测试结果
    return result.wasSuccessful()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='区块链模拟器单元测试运行器')
    parser.add_argument('module', nargs='?', help='要测试的模块名称（可选）')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出模式')
    
    args = parser.parse_args()
    
    # 设置详细程度
    verbosity = 2 if args.verbose else 1
    
    print("=" * 60)
    print("           区块链模拟器单元测试")
    print("=" * 60)
    
    if args.module:
        print(f"运行模块测试: {args.module}")
    else:
        print("运行所有测试")
    
    print(f"详细模式: {'开启' if args.verbose else '关闭'}")
    print("-" * 60)
    
    # 运行测试
    success = discover_and_run_tests(args.module, verbosity)
    
    print("-" * 60)
    if success:
        print("所有测试通过!")
        sys.exit(0)
    else:
        print("部分测试失败!")
        sys.exit(1)


if __name__ == '__main__':
    main()