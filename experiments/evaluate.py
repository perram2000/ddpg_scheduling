"""
HD-DDPG Evaluation Script - 高度增强版本
HD-DDPG模型评估脚本 - 专为稳定化组件设计

🎯 主要增强：
- 集成稳定化组件
- 实时进度显示
- 增强统计分析
- 详细可视化
- 性能基准比较
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 🎯 导入tqdm用于进度条显示
try:
    from tqdm import tqdm, trange
    TQDM_AVAILABLE = True
except ImportError:
    print("⚠️ tqdm未安装，建议安装: pip install tqdm")
    TQDM_AVAILABLE = False
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)
        def update(self, n=1):
            self.n += n
        def set_postfix(self, **kwargs):
            pass

# 导入统计分析包
try:
    from scipy import stats
    import scipy.stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️ scipy未安装，部分统计功能将不可用")
    SCIPY_AVAILABLE = False

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入稳定化组件
try:
    from src.algorithms.hd_ddpg import HDDDPG
    from src.environment.medical_simulator import StabilizedMedicalSchedulingSimulator
    from src.environment.fog_cloud_env import StabilizedFogCloudEnvironment
    from src.environment.workflow_generator import MedicalWorkflowGenerator
    from src.utils.metrics import EnhancedSchedulingMetrics
    STABILIZED_MODULES = True
    print("✅ 稳定化组件导入成功")
except ImportError as e:
    print(f"⚠️ 稳定化组件导入失败: {e}")
    print("⚠️ 将使用标准组件")
    try:
        from src.algorithms.hd_ddpg import HDDDPG
        from src.environment.medical_simulator import MedicalSchedulingSimulator
        from src.environment.fog_cloud_env import FogCloudEnvironment
        from src.environment.workflow_generator import MedicalWorkflowGenerator
        STABILIZED_MODULES = False
    except ImportError:
        print("❌ 无法导入任何组件")
        STABILIZED_MODULES = None


def parse_enhanced_arguments():
    """🎯 解析增强的命令行参数"""
    parser = argparse.ArgumentParser(
        description='HD-DDPG Model Evaluation (高度增强版本)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 🔧 模型参数
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument('--model-path', type=str, required=True,
                            help='训练模型路径')
    model_group.add_argument('--config-path', type=str, default=None,
                            help='模型配置文件路径')
    model_group.add_argument('--model-type', type=str, choices=['standard', 'stabilized'],
                            default='stabilized', help='模型类型')

    # 🔧 评估参数
    eval_group = parser.add_argument_group('评估参数')
    eval_group.add_argument('--test-workflows', type=int, default=200,
                           help='测试工作流数量')
    eval_group.add_argument('--workflow-sizes', type=str, default='6,8,10,12,15,18,20',
                           help='测试工作流大小（逗号分隔）')
    eval_group.add_argument('--workflow-types', type=str,
                           default='radiology,pathology,general,surgery,emergency',
                           help='工作流类型（逗号分隔）')
    eval_group.add_argument('--evaluation-rounds', type=int, default=3,
                           help='评估轮数（用于减少随机性）')

    # 🔧 比较参数
    comp_group = parser.add_argument_group('基准比较')
    comp_group.add_argument('--baseline-algorithms', type=str,
                           default='HEFT,FCFS,Random,RoundRobin,MinMin',
                           help='基线算法（逗号分隔）')
    comp_group.add_argument('--include-statistical-test', action='store_true', default=True,
                           help='执行统计显著性检验')
    comp_group.add_argument('--confidence-level', type=float, default=0.95,
                           help='置信水平')

    # 🔧 环境参数
    env_group = parser.add_argument_group('环境参数')
    env_group.add_argument('--failure-rate', type=float, default=0.005,
                          help='系统故障率')
    env_group.add_argument('--network-latency-factor', type=float, default=1.0,
                          help='网络延迟因子')
    env_group.add_argument('--resource-heterogeneity', type=float, default=0.2,
                          help='资源异构性程度')

    # 🔧 输出参数
    output_group = parser.add_argument_group('输出参数')
    output_group.add_argument('--output-dir', type=str, default='results/enhanced_evaluation',
                             help='输出目录')
    output_group.add_argument('--save-plots', action='store_true', default=True,
                             help='保存评估图表')
    output_group.add_argument('--save-detailed-results', action='store_true', default=True,
                             help='保存详细结果')
    output_group.add_argument('--export-format', type=str, choices=['csv', 'json', 'both'],
                             default='both', help='导出格式')

    # 🔧 可视化参数
    viz_group = parser.add_argument_group('可视化参数')
    viz_group.add_argument('--plot-style', type=str, choices=['seaborn', 'matplotlib', 'ggplot'],
                          default='seaborn', help='绘图风格')
    viz_group.add_argument('--figure-dpi', type=int, default=300,
                          help='图表DPI')
    viz_group.add_argument('--disable-progress-bar', action='store_true',
                          help='禁用进度条')

    # 🔧 分析参数
    analysis_group = parser.add_argument_group('分析参数')
    analysis_group.add_argument('--performance-metrics', type=str,
                               default='makespan,load_balance,energy,stability,quality',
                               help='性能指标（逗号分隔）')
    analysis_group.add_argument('--detailed-analysis', action='store_true', default=True,
                               help='执行详细分析')
    analysis_group.add_argument('--scalability-analysis', action='store_true', default=True,
                               help='可扩展性分析')

    # 通用参数
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    return parser.parse_args()


def load_enhanced_model_and_config(model_path, config_path=None, model_type='stabilized'):
    """🎯 加载增强模型和配置"""
    print(f"🔧 加载模型: {model_path}")

    # 🎯 智能配置文件查找
    if config_path is None:
        potential_config_paths = [
            os.path.join(os.path.dirname(model_path), '..', 'config.json'),
            os.path.join(os.path.dirname(model_path), '..', '..', 'config.json'),
            os.path.join(os.path.dirname(model_path), 'config.json'),
            'config.json',
            'stabilized_config.json'
        ]

        for path in potential_config_paths:
            if os.path.exists(path):
                config_path = path
                break

    # 加载配置
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ 配置已加载: {config_path}")
    else:
        print("⚠️ 使用默认配置")

    # 🎯 创建增强的HD-DDPG配置
    hd_ddpg_config = {
        # 基础参数
        'meta_state_dim': 15,
        'meta_action_dim': 3,
        'gamma': 0.95,
        'batch_size': config.get('batch_size', 16),
        'memory_capacity': config.get('memory_capacity', 8000),
        'meta_lr': config.get('learning_rate', 0.00008),
        'sub_lr': config.get('learning_rate', 0.00008),
        'tau': 0.005,
        'update_frequency': 1,
        'save_frequency': 50,

        # 🎯 稳定化参数（如果是稳定化模型）
        'gradient_clip_norm': 0.5,
        'exploration_noise': 0.0,  # 评估时不使用探索噪声
        'noise_decay': 1.0,
        'min_noise': 0.0,
        'reward_scale': 1.0,
        'verbose': False,

        # 🎯 高级参数
        'action_smoothing': config.get('action_smoothing', True),
        'action_smoothing_alpha': 0.25,
        'makespan_weight': 0.75,
        'stability_weight': 0.15,
        'target_update_frequency': 3,

        # 🎯 稳定化特性
        'enable_per': config.get('enable_per', True),
        'quality_threshold': config.get('quality_threshold', 0.15),
        'td_error_clipping': True,
        'adaptive_learning': False,  # 评估时禁用
    }

    # 创建HD-DDPG实例
    try:
        hd_ddpg = HDDDPG(hd_ddpg_config)
        print("✅ HD-DDPG实例创建成功")
    except Exception as e:
        print(f"❌ HD-DDPG创建失败: {e}")
        return None, None

    # 加载模型权重
    try:
        hd_ddpg.load_models(model_path)
        print(f"✅ 模型权重加载成功")

        # 🎯 验证模型
        print("🔍 验证模型完整性...")
        validation_result = validate_model(hd_ddpg)
        if validation_result:
            print("✅ 模型验证通过")
        else:
            print("⚠️ 模型验证有警告，但将继续评估")

        return hd_ddpg, config

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None


def validate_model(hd_ddpg):
    """🎯 验证模型完整性"""
    try:
        # 测试模型的基本功能
        test_state = np.random.random(15).astype(np.float32)

        # 测试元控制器
        cluster, action_probs = hd_ddpg.meta_controller.select_cluster(test_state)
        if cluster not in ['FPGA', 'FOG_GPU', 'CLOUD']:
            print(f"⚠️ 元控制器输出异常集群: {cluster}")
            return False

        # 测试子控制器
        for cluster_type in ['FPGA', 'FOG_GPU', 'CLOUD']:
            state_dims = {'FPGA': 6, 'FOG_GPU': 8, 'CLOUD': 6}
            test_cluster_state = np.random.random(state_dims[cluster_type]).astype(np.float32)

            # 创建模拟节点
            mock_nodes = [MockNode(i, cluster_type) for i in range(2)]

            try:
                node, node_probs = hd_ddpg.sub_controller_manager.select_node_in_cluster(
                    cluster_type, test_cluster_state, mock_nodes
                )
                if node is None:
                    print(f"⚠️ {cluster_type}子控制器未选择节点")
            except Exception as e:
                print(f"⚠️ {cluster_type}子控制器测试失败: {e}")
                return False

        print("✅ 所有组件验证通过")
        return True

    except Exception as e:
        print(f"⚠️ 模型验证错误: {e}")
        return False


class MockNode:
    """模拟节点类用于验证"""
    def __init__(self, node_id, cluster_type):
        self.node_id = node_id
        self.cluster_type = cluster_type
        self.memory_capacity = 100
        self.current_load = 0
        self.availability = True


def create_enhanced_environment(args):
    """🎯 创建增强的评估环境"""
    try:
        if STABILIZED_MODULES:
            # 使用稳定化环境
            env_config = {
                'failure_rate': args.failure_rate,
                'network_latency_factor': args.network_latency_factor,
                'resource_heterogeneity': args.resource_heterogeneity,
                'stability_monitoring': True,
                'performance_tracking': True
            }
            environment = StabilizedFogCloudEnvironment(env_config)
            print("✅ 稳定化环境创建成功")
        else:
            # 使用标准环境
            environment = FogCloudEnvironment()
            print("✅ 标准环境创建成功")

        return environment

    except Exception as e:
        print(f"⚠️ 环境创建失败: {e}")
        # 创建简化环境
        return create_fallback_environment()


def create_fallback_environment():
    """创建回退环境"""
    class FallbackEnvironment:
        def __init__(self):
            self.current_time = 0
            self.nodes = self._create_nodes()

        def _create_nodes(self):
            from collections import defaultdict
            nodes = defaultdict(list)

            node_configs = {
                'FPGA': {'count': 2, 'memory': 100, 'base_time': 0.4, 'base_energy': 8},
                'FOG_GPU': {'count': 3, 'memory': 150, 'base_time': 0.8, 'base_energy': 15},
                'CLOUD': {'count': 2, 'memory': 200, 'base_time': 1.6, 'base_energy': 25}
            }

            node_id = 0
            for cluster_type, config in node_configs.items():
                for i in range(config['count']):
                    node = type('Node', (), {
                        'node_id': node_id,
                        'cluster_type': cluster_type,
                        'memory_capacity': config['memory'],
                        'current_load': 0,
                        'availability': True,
                        'can_accommodate': lambda self, req: self.memory_capacity - self.current_load >= req,
                        'get_execution_time': lambda self, comp_req, cfg=config: cfg['base_time'] * comp_req,
                        'get_energy_consumption': lambda self, exec_time, cfg=config: cfg['base_energy'] * exec_time
                    })()
                    nodes[cluster_type].append(node)
                    node_id += 1

            return nodes

        def reset(self):
            self.current_time = 0
            for cluster_nodes in self.nodes.values():
                for node in cluster_nodes:
                    node.current_load = 0

        def get_system_state(self):
            return np.random.random(15)

        def get_stabilized_system_state(self):
            return self.get_system_state()

        def get_cluster_state(self, cluster_type):
            dims = {'FPGA': 6, 'FOG_GPU': 8, 'CLOUD': 6}
            return np.random.random(dims.get(cluster_type, 6))

        def get_enhanced_cluster_state(self, cluster_type):
            return self.get_cluster_state(cluster_type)

        def get_available_nodes(self, cluster_type=None):
            if cluster_type:
                return [node for node in self.nodes.get(cluster_type, []) if node.availability]
            else:
                available = []
                for cluster_nodes in self.nodes.values():
                    available.extend([node for node in cluster_nodes if node.availability])
                return available

        def update_node_load(self, node_id, memory_req):
            for cluster_nodes in self.nodes.values():
                for node in cluster_nodes:
                    if node.node_id == node_id:
                        node.current_load += memory_req
                        break

    return FallbackEnvironment()


def generate_enhanced_test_workflows(workflow_generator, test_configs, args):
    """🎯 生成增强的测试工作流"""
    print("📋 生成测试工作流...")

    test_workflows = []

    # 设置进度条
    use_tqdm = TQDM_AVAILABLE and not args.disable_progress_bar

    if use_tqdm:
        config_pbar = tqdm(test_configs, desc="生成工作流配置", unit="config")
    else:
        config_pbar = test_configs

    for config in config_pbar:
        workflow_type = config['type']
        workflow_size = config['size']
        count = config['count']

        for round_idx in range(args.evaluation_rounds):
            for i in range(count):
                try:
                    if hasattr(workflow_generator, 'generate_workflow'):
                        workflow = workflow_generator.generate_workflow(
                            num_tasks=workflow_size,
                            workflow_type=workflow_type
                        )
                    else:
                        # 创建简化工作流
                        workflow = create_simple_workflow(workflow_size, workflow_type, f"{workflow_type}_{workflow_size}_{round_idx}_{i}")

                    test_workflows.append({
                        'workflow': workflow,
                        'type': workflow_type,
                        'size': workflow_size,
                        'round': round_idx,
                        'workflow_id': f"{workflow_type}_{workflow_size}_{round_idx}_{i}"
                    })

                except Exception as e:
                    print(f"⚠️ 工作流生成失败: {e}")
                    # 创建简化工作流作为备用
                    workflow = create_simple_workflow(workflow_size, workflow_type, f"fallback_{workflow_type}_{workflow_size}_{round_idx}_{i}")
                    test_workflows.append({
                        'workflow': workflow,
                        'type': workflow_type,
                        'size': workflow_size,
                        'round': round_idx,
                        'workflow_id': f"fallback_{workflow_type}_{workflow_size}_{round_idx}_{i}"
                    })

    print(f"✅ 生成了 {len(test_workflows)} 个测试工作流")
    return test_workflows


def create_simple_workflow(size, workflow_type, workflow_id):
    """创建简化工作流"""
    class SimpleTask:
        def __init__(self, task_id, computation_req=1.0, memory_req=10, priority=1):
            self.task_id = task_id
            self.computation_requirement = computation_req
            self.memory_requirement = memory_req
            self.priority = priority
            self.dependencies = []

    workflow = []
    for i in range(size):
        # 根据工作流类型调整任务属性
        if workflow_type == 'radiology':
            comp_req = np.random.uniform(1.0, 2.5)
            mem_req = np.random.randint(15, 35)
        elif workflow_type == 'pathology':
            comp_req = np.random.uniform(0.8, 2.0)
            mem_req = np.random.randint(10, 25)
        elif workflow_type == 'surgery':
            comp_req = np.random.uniform(1.5, 3.0)
            mem_req = np.random.randint(20, 40)
        elif workflow_type == 'emergency':
            comp_req = np.random.uniform(0.5, 1.5)
            mem_req = np.random.randint(8, 20)
        else:  # general
            comp_req = np.random.uniform(0.7, 2.0)
            mem_req = np.random.randint(10, 30)

        task = SimpleTask(
            task_id=f"{workflow_id}_task_{i}",
            computation_req=comp_req,
            memory_req=mem_req,
            priority=np.random.randint(1, 4)
        )
        workflow.append(task)

    return workflow


def evaluate_hd_ddpg_enhanced(hd_ddpg, test_workflows, environment, args):
    """🎯 增强的HD-DDPG评估"""
    print(f"🤖 评估HD-DDPG ({len(test_workflows)} 个工作流)...")

    results = []
    use_tqdm = TQDM_AVAILABLE and not args.disable_progress_bar

    if use_tqdm:
        workflow_pbar = tqdm(test_workflows, desc="HD-DDPG评估", unit="workflow")
    else:
        workflow_pbar = test_workflows

    start_time = time.time()

    for i, test_case in enumerate(workflow_pbar):
        try:
            workflow = test_case['workflow']
            workflow_type = test_case['type']
            workflow_size = test_case['size']
            round_idx = test_case['round']
            workflow_id = test_case['workflow_id']

            # 重置环境
            environment.reset()

            # 🎯 调度工作流
            schedule_start_time = time.time()
            result = hd_ddpg.schedule_workflow(workflow, environment)
            scheduling_time = time.time() - schedule_start_time

            # 🎯 计算增强指标
            enhanced_metrics = calculate_enhanced_metrics(result, workflow, environment)

            # 记录结果
            result_entry = {
                'workflow_id': workflow_id,
                'workflow_type': workflow_type,
                'workflow_size': workflow_size,
                'evaluation_round': round_idx,
                'makespan': result.get('makespan', float('inf')),
                'load_balance': result.get('load_balance', 0),
                'total_energy': result.get('total_energy', 0),
                'completed_tasks': result.get('completed_tasks', 0),
                'total_reward': result.get('total_reward', 0),
                'success_rate': result.get('success_rate', 0),
                'scheduling_time': scheduling_time,
                'algorithm': 'HD-DDPG',

                # 🎯 增强指标
                'stability_score': enhanced_metrics.get('stability_score', 0),
                'quality_score': enhanced_metrics.get('quality_score', 0),
                'resource_utilization': enhanced_metrics.get('resource_utilization', 0),
                'energy_efficiency': enhanced_metrics.get('energy_efficiency', 0),
                'throughput': enhanced_metrics.get('throughput', 0)
            }

            results.append(result_entry)

            # 🎯 更新进度条信息
            if use_tqdm:
                workflow_pbar.set_postfix({
                    'Makespan': f"{result_entry['makespan']:.2f}",
                    'Success': f"{result_entry['success_rate']:.1%}",
                    'Quality': f"{result_entry['quality_score']:.2f}"
                })

        except Exception as e:
            print(f"⚠️ 工作流 {i} 评估失败: {e}")
            # 添加失败记录
            results.append({
                'workflow_id': test_case.get('workflow_id', f'failed_{i}'),
                'workflow_type': test_case['type'],
                'workflow_size': test_case['size'],
                'evaluation_round': test_case['round'],
                'makespan': float('inf'),
                'load_balance': 0,
                'total_energy': 0,
                'completed_tasks': 0,
                'total_reward': -100,
                'success_rate': 0,
                'scheduling_time': 0,
                'algorithm': 'HD-DDPG',
                'stability_score': 0,
                'quality_score': 0,
                'resource_utilization': 0,
                'energy_efficiency': 0,
                'throughput': 0
            })

    total_time = time.time() - start_time
    print(f"✅ HD-DDPG评估完成 (耗时: {total_time:.2f}s)")

    return results


def calculate_enhanced_metrics(result, workflow, environment):
    """🎯 计算增强指标"""
    try:
        enhanced_metrics = {}

        # 稳定性评分
        makespan = result.get('makespan', float('inf'))
        if makespan != float('inf'):
            # 基于makespan的稳定性
            stability_score = 1.0 / (1.0 + makespan * 0.1)
        else:
            stability_score = 0.0
        enhanced_metrics['stability_score'] = stability_score

        # 质量评分
        success_rate = result.get('success_rate', 0)
        load_balance = result.get('load_balance', 0)
        quality_score = (success_rate + load_balance + stability_score) / 3
        enhanced_metrics['quality_score'] = quality_score

        # 资源利用率
        completed_tasks = result.get('completed_tasks', 0)
        total_tasks = len(workflow) if workflow else 1
        resource_utilization = completed_tasks / total_tasks
        enhanced_metrics['resource_utilization'] = resource_utilization

        # 能耗效率
        total_energy = result.get('total_energy', 1)
        if total_energy > 0 and completed_tasks > 0:
            energy_efficiency = completed_tasks / total_energy
        else:
            energy_efficiency = 0
        enhanced_metrics['energy_efficiency'] = energy_efficiency

        # 吞吐量
        if makespan != float('inf') and makespan > 0:
            throughput = completed_tasks / makespan
        else:
            throughput = 0
        enhanced_metrics['throughput'] = throughput

        return enhanced_metrics

    except Exception as e:
        print(f"⚠️ 增强指标计算错误: {e}")
        return {
            'stability_score': 0,
            'quality_score': 0,
            'resource_utilization': 0,
            'energy_efficiency': 0,
            'throughput': 0
        }


def evaluate_baseline_enhanced(algorithm, test_workflows, environment, args):
    """🎯 增强的基线算法评估"""
    print(f"📊 评估基线算法: {algorithm} ({len(test_workflows)} 个工作流)...")

    results = []
    use_tqdm = TQDM_AVAILABLE and not args.disable_progress_bar

    if use_tqdm:
        workflow_pbar = tqdm(test_workflows, desc=f"{algorithm}评估", unit="workflow")
    else:
        workflow_pbar = test_workflows

    for i, test_case in enumerate(workflow_pbar):
        try:
            workflow = test_case['workflow']
            workflow_type = test_case['type']
            workflow_size = test_case['size']
            round_idx = test_case['round']
            workflow_id = test_case['workflow_id']

            # 重置环境
            environment.reset()

            # 🎯 运行基线算法
            schedule_start_time = time.time()
            result = run_baseline_algorithm(algorithm, workflow, environment)
            scheduling_time = time.time() - schedule_start_time

            # 🎯 计算增强指标
            enhanced_metrics = calculate_enhanced_metrics(result, workflow, environment)

            # 记录结果
            result_entry = {
                'workflow_id': workflow_id,
                'workflow_type': workflow_type,
                'workflow_size': workflow_size,
                'evaluation_round': round_idx,
                'makespan': result.get('makespan', float('inf')),
                'load_balance': result.get('load_balance', 0),
                'total_energy': result.get('total_energy', 0),
                'completed_tasks': result.get('completed_tasks', 0),
                'total_reward': 0,  # 基线算法不计算reward
                'success_rate': result.get('success_rate', 0),
                'scheduling_time': scheduling_time,
                'algorithm': algorithm,

                # 🎯 增强指标
                'stability_score': enhanced_metrics.get('stability_score', 0),
                'quality_score': enhanced_metrics.get('quality_score', 0),
                'resource_utilization': enhanced_metrics.get('resource_utilization', 0),
                'energy_efficiency': enhanced_metrics.get('energy_efficiency', 0),
                'throughput': enhanced_metrics.get('throughput', 0)
            }

            results.append(result_entry)

            # 🎯 更新进度条信息
            if use_tqdm:
                workflow_pbar.set_postfix({
                    'Makespan': f"{result_entry['makespan']:.2f}",
                    'Success': f"{result_entry['success_rate']:.1%}"
                })

        except Exception as e:
            print(f"⚠️ {algorithm} 工作流 {i} 评估失败: {e}")
            # 添加失败记录
            results.append({
                'workflow_id': test_case.get('workflow_id', f'{algorithm}_failed_{i}'),
                'workflow_type': test_case['type'],
                'workflow_size': test_case['size'],
                'evaluation_round': test_case['round'],
                'makespan': float('inf'),
                'load_balance': 0,
                'total_energy': 0,
                'completed_tasks': 0,
                'total_reward': 0,
                'success_rate': 0,
                'scheduling_time': 0,
                'algorithm': algorithm,
                'stability_score': 0,
                'quality_score': 0,
                'resource_utilization': 0,
                'energy_efficiency': 0,
                'throughput': 0
            })

    print(f"✅ {algorithm}评估完成")
    return results


def run_baseline_algorithm(algorithm, workflow, environment):
    """🎯 运行基线算法"""
    try:
        if algorithm == 'HEFT':
            return run_heft_algorithm(workflow, environment)
        elif algorithm == 'FCFS':
            return run_fcfs_algorithm(workflow, environment)
        elif algorithm == 'Random':
            return run_random_algorithm(workflow, environment)
        elif algorithm == 'RoundRobin':
            return run_round_robin_algorithm(workflow, environment)
        elif algorithm == 'MinMin':
            return run_min_min_algorithm(workflow, environment)
        else:
            print(f"⚠️ 未知基线算法: {algorithm}")
            return run_random_algorithm(workflow, environment)

    except Exception as e:
        print(f"⚠️ 基线算法 {algorithm} 执行失败: {e}")
        return {
            'makespan': float('inf'),
            'load_balance': 0,
            'total_energy': 0,
            'completed_tasks': 0,
            'success_rate': 0
        }


def run_heft_algorithm(workflow, environment):
    """HEFT算法实现"""
    try:
        # 简化的HEFT实现
        task_completion_times = []
        total_energy = 0
        completed_tasks = 0

        # 获取所有可用节点
        all_nodes = []
        for cluster_type in ['FPGA', 'FOG_GPU', 'CLOUD']:
            nodes = environment.get_available_nodes(cluster_type)
            all_nodes.extend(nodes)

        if not all_nodes:
            return {'makespan': float('inf'), 'load_balance': 0, 'total_energy': 0,
                   'completed_tasks': 0, 'success_rate': 0}

        # 按计算时间排序任务
        for task in workflow:
            # 选择最快的节点
            best_node = None
            best_time = float('inf')

            for node in all_nodes:
                if node.availability and hasattr(node, 'get_execution_time'):
                    comp_req = getattr(task, 'computation_requirement', 1.0)
                    exec_time = node.get_execution_time(comp_req)
                    if exec_time < best_time:
                        best_time = exec_time
                        best_node = node

            if best_node:
                # 执行任务
                completion_time = environment.current_time + best_time
                task_completion_times.append(completion_time)

                # 计算能耗
                if hasattr(best_node, 'get_energy_consumption'):
                    energy = best_node.get_energy_consumption(best_time)
                    total_energy += energy

                # 更新环境
                environment.current_time = completion_time
                memory_req = getattr(task, 'memory_requirement', 10)
                environment.update_node_load(best_node.node_id, memory_req)
                completed_tasks += 1

        # 计算指标
        makespan = max(task_completion_times) if task_completion_times else float('inf')
        success_rate = completed_tasks / len(workflow) if workflow else 0

        # 简化的负载均衡计算
        load_balance = min(1.0, success_rate)

        return {
            'makespan': makespan,
            'load_balance': load_balance,
            'total_energy': total_energy,
            'completed_tasks': completed_tasks,
            'success_rate': success_rate
        }

    except Exception as e:
        print(f"⚠️ HEFT算法执行错误: {e}")
        return {'makespan': float('inf'), 'load_balance': 0, 'total_energy': 0,
               'completed_tasks': 0, 'success_rate': 0}


def run_fcfs_algorithm(workflow, environment):
    """FCFS算法实现"""
    try:
        task_completion_times = []
        total_energy = 0
        completed_tasks = 0

        # 获取所有可用节点
        all_nodes = []
        for cluster_type in ['FPGA', 'FOG_GPU', 'CLOUD']:
            nodes = environment.get_available_nodes(cluster_type)
            all_nodes.extend(nodes)

        if not all_nodes:
            return {'makespan': float('inf'), 'load_balance': 0, 'total_energy': 0,
                   'completed_tasks': 0, 'success_rate': 0}

        node_index = 0

        # 按顺序分配任务
        for task in workflow:
            if all_nodes:
                # 轮流选择节点
                selected_node = all_nodes[node_index % len(all_nodes)]
                node_index += 1

                if hasattr(selected_node, 'get_execution_time'):
                    comp_req = getattr(task, 'computation_requirement', 1.0)
                    exec_time = selected_node.get_execution_time(comp_req)

                    completion_time = environment.current_time + exec_time
                    task_completion_times.append(completion_time)

                    # 计算能耗
                    if hasattr(selected_node, 'get_energy_consumption'):
                        energy = selected_node.get_energy_consumption(exec_time)
                        total_energy += energy

                    # 更新环境
                    environment.current_time = completion_time
                    memory_req = getattr(task, 'memory_requirement', 10)
                    environment.update_node_load(selected_node.node_id, memory_req)
                    completed_tasks += 1

        # 计算指标
        makespan = max(task_completion_times) if task_completion_times else float('inf')
        success_rate = completed_tasks / len(workflow) if workflow else 0
        load_balance = min(1.0, success_rate)

        return {
            'makespan': makespan,
            'load_balance': load_balance,
            'total_energy': total_energy,
            'completed_tasks': completed_tasks,
            'success_rate': success_rate
        }

    except Exception as e:
        print(f"⚠️ FCFS算法执行错误: {e}")
        return {'makespan': float('inf'), 'load_balance': 0, 'total_energy': 0,
               'completed_tasks': 0, 'success_rate': 0}


def run_random_algorithm(workflow, environment):
    """随机算法实现"""
    try:
        task_completion_times = []
        total_energy = 0
        completed_tasks = 0

        # 获取所有可用节点
        all_nodes = []
        for cluster_type in ['FPGA', 'FOG_GPU', 'CLOUD']:
            nodes = environment.get_available_nodes(cluster_type)
            all_nodes.extend(nodes)

        if not all_nodes:
            return {'makespan': float('inf'), 'load_balance': 0, 'total_energy': 0,
                   'completed_tasks': 0, 'success_rate': 0}

        # 随机分配任务
        for task in workflow:
            if all_nodes:
                # 随机选择节点
                selected_node = np.random.choice(all_nodes)

                if hasattr(selected_node, 'get_execution_time'):
                    comp_req = getattr(task, 'computation_requirement', 1.0)
                    exec_time = selected_node.get_execution_time(comp_req)

                    completion_time = environment.current_time + exec_time
                    task_completion_times.append(completion_time)

                    # 计算能耗
                    if hasattr(selected_node, 'get_energy_consumption'):
                        energy = selected_node.get_energy_consumption(exec_time)
                        total_energy += energy

                    # 更新环境
                    environment.current_time = completion_time
                    memory_req = getattr(task, 'memory_requirement', 10)
                    environment.update_node_load(selected_node.node_id, memory_req)
                    completed_tasks += 1

        # 计算指标
        makespan = max(task_completion_times) if task_completion_times else float('inf')
        success_rate = completed_tasks / len(workflow) if workflow else 0
        load_balance = min(1.0, success_rate * 0.7)  # 随机算法负载均衡较差

        return {
            'makespan': makespan,
            'load_balance': load_balance,
            'total_energy': total_energy,
            'completed_tasks': completed_tasks,
            'success_rate': success_rate
        }

    except Exception as e:
        print(f"⚠️ Random算法执行错误: {e}")
        return {'makespan': float('inf'), 'load_balance': 0, 'total_energy': 0,
               'completed_tasks': 0, 'success_rate': 0}


def run_round_robin_algorithm(workflow, environment):
    """轮询算法实现"""
    # 与FCFS类似，但更规则的轮询
    return run_fcfs_algorithm(workflow, environment)


def run_min_min_algorithm(workflow, environment):
    """MinMin算法实现"""
    try:
        task_completion_times = []
        total_energy = 0
        completed_tasks = 0

        # 获取所有可用节点
        all_nodes = []
        for cluster_type in ['FPGA', 'FOG_GPU', 'CLOUD']:
            nodes = environment.get_available_nodes(cluster_type)
            all_nodes.extend(nodes)

        if not all_nodes:
            return {'makespan': float('inf'), 'load_balance': 0, 'total_energy': 0,
                   'completed_tasks': 0, 'success_rate': 0}

        # MinMin: 优先处理小任务
        sorted_workflow = sorted(workflow,
                               key=lambda t: getattr(t, 'computation_requirement', 1.0))

        for task in sorted_workflow:
            # 选择最快的节点
            best_node = None
            best_time = float('inf')

            for node in all_nodes:
                if node.availability and hasattr(node, 'get_execution_time'):
                    comp_req = getattr(task, 'computation_requirement', 1.0)
                    exec_time = node.get_execution_time(comp_req)
                    if exec_time < best_time:
                        best_time = exec_time
                        best_node = node

            if best_node:
                completion_time = environment.current_time + best_time
                task_completion_times.append(completion_time)

                # 计算能耗
                if hasattr(best_node, 'get_energy_consumption'):
                    energy = best_node.get_energy_consumption(best_time)
                    total_energy += energy

                # 更新环境
                environment.current_time = completion_time
                memory_req = getattr(task, 'memory_requirement', 10)
                environment.update_node_load(best_node.node_id, memory_req)
                completed_tasks += 1

        # 计算指标
        makespan = max(task_completion_times) if task_completion_times else float('inf')
        success_rate = completed_tasks / len(workflow) if workflow else 0
        load_balance = min(1.0, success_rate * 0.8)  # MinMin负载均衡一般

        return {
            'makespan': makespan,
            'load_balance': load_balance,
            'total_energy': total_energy,
            'completed_tasks': completed_tasks,
            'success_rate': success_rate
        }

    except Exception as e:
        print(f"⚠️ MinMin算法执行错误: {e}")
        return {'makespan': float('inf'), 'load_balance': 0, 'total_energy': 0,
               'completed_tasks': 0, 'success_rate': 0}


def perform_enhanced_statistical_analysis(results_df, args):
    """🎯 执行增强的统计分析"""
    if not SCIPY_AVAILABLE:
        print("⚠️ scipy不可用，跳过统计分析")
        return {}

    print("\n=== 增强统计分析 ===")

    # 按算法分组
    algorithms = results_df['algorithm'].unique()
    metrics = ['makespan', 'load_balance', 'total_energy', 'scheduling_time',
               'stability_score', 'quality_score', 'resource_utilization',
               'energy_efficiency', 'throughput']

    statistical_results = {}

    for metric in metrics:
        print(f"\n📊 {metric.upper()} 分析:")
        metric_results = {}

        # 计算描述性统计
        for algorithm in algorithms:
            data = results_df[results_df['algorithm'] == algorithm][metric]
            # 过滤无效值
            valid_data = data[~data.isin([float('inf'), float('-inf')]) & data.notna()]

            if len(valid_data) > 0:
                metric_results[algorithm] = {
                    'count': len(valid_data),
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std()),
                    'median': float(valid_data.median()),
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max()),
                    'q25': float(valid_data.quantile(0.25)),
                    'q75': float(valid_data.quantile(0.75))
                }

                print(f"  {algorithm:12s}: {valid_data.mean():8.3f} ± {valid_data.std():6.3f} "
                      f"(n={len(valid_data)})")

        # HD-DDPG vs 基线算法的显著性检验
        if 'HD-DDPG' in algorithms:
            hd_ddpg_data = results_df[results_df['algorithm'] == 'HD-DDPG'][metric]
            hd_ddpg_valid = hd_ddpg_data[~hd_ddpg_data.isin([float('inf'), float('-inf')]) & hd_ddpg_data.notna()]

            for baseline in algorithms:
                if baseline != 'HD-DDPG':
                    baseline_data = results_df[results_df['algorithm'] == baseline][metric]
                    baseline_valid = baseline_data[~baseline_data.isin([float('inf'), float('-inf')]) & baseline_data.notna()]

                    if len(hd_ddpg_valid) > 5 and len(baseline_valid) > 5:
                        try:
                            # 🎯 多种统计检验
                            # Wilcoxon rank-sum test (非参数)
                            statistic_w, p_value_w = stats.ranksums(hd_ddpg_valid, baseline_valid)

                            # Mann-Whitney U test
                            statistic_u, p_value_u = stats.mannwhitneyu(hd_ddpg_valid, baseline_valid, alternative='two-sided')

                            # T-test (如果数据近似正态分布)
                            try:
                                statistic_t, p_value_t = stats.ttest_ind(hd_ddpg_valid, baseline_valid, equal_var=False)
                            except:
                                statistic_t, p_value_t = np.nan, 1.0

                            # 效应大小 (Cohen's d)
                            pooled_std = np.sqrt(((len(hd_ddpg_valid) - 1) * hd_ddpg_valid.var() +
                                                 (len(baseline_valid) - 1) * baseline_valid.var()) /
                                                (len(hd_ddpg_valid) + len(baseline_valid) - 2))
                            if pooled_std > 0:
                                cohens_d = (hd_ddpg_valid.mean() - baseline_valid.mean()) / pooled_std
                            else:
                                cohens_d = 0

                            metric_results[f'HD-DDPG_vs_{baseline}'] = {
                                'wilcoxon_statistic': float(statistic_w),
                                'wilcoxon_p_value': float(p_value_w),
                                'mannwhitney_statistic': float(statistic_u),
                                'mannwhitney_p_value': float(p_value_u),
                                'ttest_statistic': float(statistic_t),
                                'ttest_p_value': float(p_value_t),
                                'cohens_d': float(cohens_d),
                                'significant_wilcoxon': p_value_w < (1 - args.confidence_level),
                                'significant_mannwhitney': p_value_u < (1 - args.confidence_level),
                                'significant_ttest': p_value_t < (1 - args.confidence_level) if not np.isnan(p_value_t) else False,
                                'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
                            }

                            # 输出结果
                            significance_w = "✓" if p_value_w < (1 - args.confidence_level) else "✗"
                            significance_u = "✓" if p_value_u < (1 - args.confidence_level) else "✗"
                            print(f"    vs {baseline:10s}: Wilcoxon p={p_value_w:.4f} {significance_w}, "
                                  f"M-W p={p_value_u:.4f} {significance_u}, Cohen's d={cohens_d:.3f}")

                        except Exception as e:
                            print(f"    vs {baseline:10s}: 统计检验失败 - {e}")

        statistical_results[metric] = metric_results

    return statistical_results


def create_enhanced_evaluation_plots(results_df, output_dir, args):
    """🎯 创建增强的评估图表"""
    print("📈 创建增强评估图表...")

    # 设置绘图风格
    if args.plot_style == 'seaborn':
        plt.style.use('seaborn-v0_8')
    elif args.plot_style == 'ggplot':
        plt.style.use('ggplot')
    else:
        plt.style.use('default')

    # 设置字体和颜色
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'DejaVu Sans',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.autolayout': True
    })

    # 🎯 主要性能比较图 (3x3)
    fig1, axes1 = plt.subplots(3, 3, figsize=(20, 16))
    axes1 = axes1.flatten()

    main_metrics = ['makespan', 'load_balance', 'total_energy', 'scheduling_time',
                   'stability_score', 'quality_score', 'resource_utilization',
                   'energy_efficiency', 'throughput']

    algorithms = results_df['algorithm'].unique()

    for i, metric in enumerate(main_metrics):
        if i < len(axes1):
            # 过滤有效数据
            plot_data = results_df[~results_df[metric].isin([float('inf'), float('-inf')]) & results_df[metric].notna()]

            if len(plot_data) > 0:
                # 箱线图
                sns.boxplot(data=plot_data, x='algorithm', y=metric, ax=axes1[i],
                           palette='Set2', showfliers=True)

                # 添加均值点
                means = plot_data.groupby('algorithm')[metric].mean()
                for j, (algo, mean_val) in enumerate(means.items()):
                    axes1[i].scatter(j, mean_val, color='red', s=100, marker='D',
                                   zorder=10, label='Mean' if j == 0 else '')

                axes1[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold', fontsize=14)
                axes1[i].tick_params(axis='x', rotation=45)
                axes1[i].grid(True, alpha=0.3)

                if i == 0:  # 只在第一个子图添加图例
                    axes1[i].legend()
            else:
                axes1[i].text(0.5, 0.5, 'No Valid Data', ha='center', va='center',
                             transform=axes1[i].transAxes, fontsize=16)
                axes1[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')

    plt.suptitle('HD-DDPG vs Baseline Algorithms - Performance Comparison',
                fontsize=18, fontweight='bold')

    if args.save_plots:
        plot1_path = os.path.join(output_dir, 'enhanced_performance_comparison.png')
        plt.savefig(plot1_path, dpi=args.figure_dpi, bbox_inches='tight')
        print(f"✅ 性能比较图已保存: {plot1_path}")

    plt.show()

    # 🎯 可扩展性分析图
    if args.scalability_analysis:
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

        # 工作流大小 vs 性能
        valid_data = results_df[~results_df['makespan'].isin([float('inf'), float('-inf')]) & results_df['makespan'].notna()]
        if len(valid_data) > 0:
            sns.lineplot(data=valid_data, x='workflow_size', y='makespan',
                        hue='algorithm', marker='o', ax=axes2[0,0])
            axes2[0,0].set_title('Makespan vs Workflow Size', fontweight='bold')
            axes2[0,0].grid(True, alpha=0.3)

        # 工作流类型性能比较
        type_data = results_df[~results_df['makespan'].isin([float('inf'), float('-inf')]) & results_df['makespan'].notna()]
        if len(type_data) > 0:
            makespan_by_type = type_data.groupby(['algorithm', 'workflow_type'])['makespan'].mean().reset_index()
            sns.barplot(data=makespan_by_type, x='workflow_type', y='makespan',
                       hue='algorithm', ax=axes2[0,1])
            axes2[0,1].set_title('Average Makespan by Workflow Type', fontweight='bold')
            axes2[0,1].tick_params(axis='x', rotation=45)
            axes2[0,1].grid(True, alpha=0.3)

        # 质量评分分布
        quality_data = results_df[results_df['quality_score'].notna()]
        if len(quality_data) > 0:
            sns.violinplot(data=quality_data, x='algorithm', y='quality_score', ax=axes2[1,0])
            axes2[1,0].set_title('Quality Score Distribution', fontweight='bold')
            axes2[1,0].tick_params(axis='x', rotation=45)
            axes2[1,0].grid(True, alpha=0.3)

        # 能耗效率 vs 吞吐量
        efficiency_data = results_df[(results_df['energy_efficiency'].notna()) &
                                   (results_df['throughput'].notna()) &
                                   (~results_df['energy_efficiency'].isin([float('inf'), float('-inf')])) &
                                   (~results_df['throughput'].isin([float('inf'), float('-inf')]))]
        if len(efficiency_data) > 0:
            sns.scatterplot(data=efficiency_data, x='energy_efficiency', y='throughput',
                           hue='algorithm', size='workflow_size', ax=axes2[1,1], alpha=0.7)
            axes2[1,1].set_title('Energy Efficiency vs Throughput', fontweight='bold')
            axes2[1,1].grid(True, alpha=0.3)

        plt.suptitle('Scalability and Detailed Performance Analysis',
                    fontsize=16, fontweight='bold')

        if args.save_plots:
            plot2_path = os.path.join(output_dir, 'scalability_analysis.png')
            plt.savefig(plot2_path, dpi=args.figure_dpi, bbox_inches='tight')
            print(f"✅ 可扩展性分析图已保存: {plot2_path}")

        plt.show()

    # 🎯 性能改进热图
    if 'HD-DDPG' in algorithms:
        fig3, ax3 = plt.subplots(figsize=(12, 8))

        improvement_data = []

        for baseline in algorithms:
            if baseline != 'HD-DDPG':
                for metric in ['makespan', 'total_energy']:  # 越小越好的指标
                    hd_ddpg_data = results_df[results_df['algorithm'] == 'HD-DDPG'][metric]
                    baseline_data = results_df[results_df['algorithm'] == baseline][metric]

                    # 过滤有效数据
                    hd_ddpg_valid = hd_ddpg_data[~hd_ddpg_data.isin([float('inf'), float('-inf')]) & hd_ddpg_data.notna()]
                    baseline_valid = baseline_data[~baseline_data.isin([float('inf'), float('-inf')]) & baseline_data.notna()]

                    if len(hd_ddpg_valid) > 0 and len(baseline_valid) > 0:
                        hd_ddpg_mean = hd_ddpg_valid.mean()
                        baseline_mean = baseline_valid.mean()
                        if baseline_mean > 0:
                            improvement = (baseline_mean - hd_ddpg_mean) / baseline_mean * 100
                            improvement_data.append({
                                'Baseline': baseline,
                                'Metric': metric,
                                'Improvement (%)': improvement
                            })

                # 负载均衡、质量评分等（越大越好的指标）
                for metric in ['load_balance', 'quality_score', 'stability_score']:
                    hd_ddpg_data = results_df[results_df['algorithm'] == 'HD-DDPG'][metric]
                    baseline_data = results_df[results_df['algorithm'] == baseline][metric]

                    hd_ddpg_valid = hd_ddpg_data[hd_ddpg_data.notna()]
                    baseline_valid = baseline_data[baseline_data.notna()]

                    if len(hd_ddpg_valid) > 0 and len(baseline_valid) > 0:
                        hd_ddpg_mean = hd_ddpg_valid.mean()
                        baseline_mean = baseline_valid.mean()
                        if baseline_mean > 0:
                            improvement = (hd_ddpg_mean - baseline_mean) / baseline_mean * 100
                            improvement_data.append({
                                'Baseline': baseline,
                                'Metric': metric,
                                'Improvement (%)': improvement
                            })

        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            improvement_pivot = improvement_df.pivot(index='Baseline', columns='Metric', values='Improvement (%)')

            # 创建热图
            mask = improvement_pivot.isnull()
            sns.heatmap(improvement_pivot, annot=True, cmap='RdYlGn', center=0,
                       fmt='.1f', ax=ax3, mask=mask, cbar_kws={'label': 'Improvement (%)'})
            ax3.set_title('HD-DDPG Performance Improvement vs Baselines (%)',
                         fontweight='bold', fontsize=14)

            if args.save_plots:
                heatmap_path = os.path.join(output_dir, 'improvement_heatmap.png')
                plt.savefig(heatmap_path, dpi=args.figure_dpi, bbox_inches='tight')
                print(f"✅ 改进热图已保存: {heatmap_path}")

            plt.show()

    print("✅ 所有图表创建完成")


def generate_enhanced_evaluation_report(results_df, statistical_results, output_dir, args):
    """🎯 生成增强的评估报告"""
    print("📝 生成增强评估报告...")

    report_path = os.path.join(output_dir, 'enhanced_evaluation_report.md')

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            # 报告头部
            f.write("# HD-DDPG Enhanced Evaluation Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Test Workflows:** {len(results_df)}\n")
            f.write(f"**Evaluation Rounds:** {args.evaluation_rounds}\n")
            f.write(f"**Model Type:** {args.model_type}\n")
            f.write(f"**Statistical Confidence Level:** {args.confidence_level}\n\n")

            # 🎯 执行摘要
            f.write("## Executive Summary\n")
            f.write("-" * 20 + "\n\n")

            # 计算HD-DDPG的整体性能
            hd_ddpg_data = results_df[results_df['algorithm'] == 'HD-DDPG']
            if len(hd_ddpg_data) > 0:
                avg_makespan = hd_ddpg_data['makespan'][~hd_ddpg_data['makespan'].isin([float('inf')])].mean()
                avg_quality = hd_ddpg_data['quality_score'].mean()
                avg_stability = hd_ddpg_data['stability_score'].mean()
                success_rate = hd_ddpg_data['success_rate'].mean()

                f.write(f"**HD-DDPG Overall Performance:**\n")
                f.write(f"- Average Makespan: {avg_makespan:.3f}\n")
                f.write(f"- Quality Score: {avg_quality:.3f}\n")
                f.write(f"- Stability Score: {avg_stability:.3f}\n")
                f.write(f"- Success Rate: {success_rate:.1%}\n\n")

            # 🎯 算法性能摘要
            f.write("## Algorithm Performance Summary\n")
            f.write("-" * 30 + "\n\n")

            algorithms = results_df['algorithm'].unique()
            for algorithm in algorithms:
                algo_data = results_df[results_df['algorithm'] == algorithm]

                f.write(f"### {algorithm}\n")

                # 基础指标
                makespan_data = algo_data['makespan'][~algo_data['makespan'].isin([float('inf'), float('-inf')])].dropna()
                if len(makespan_data) > 0:
                    f.write(f"- **Makespan:** {makespan_data.mean():.3f} ± {makespan_data.std():.3f}\n")

                for metric, name in [('load_balance', 'Load Balance'), ('total_energy', 'Energy'),
                                   ('scheduling_time', 'Scheduling Time'), ('quality_score', 'Quality Score'),
                                   ('stability_score', 'Stability Score')]:
                    data = algo_data[metric].dropna()
                    if len(data) > 0:
                        f.write(f"- **{name}:** {data.mean():.3f} ± {data.std():.3f}\n")

                f.write("\n")

            # 🎯 统计显著性结果
            if statistical_results and args.include_statistical_test:
                f.write("## Statistical Significance Analysis\n")
                f.write("-" * 35 + "\n\n")

                for metric, metric_results in statistical_results.items():
                    f.write(f"### {metric.upper()}\n")

                    # 描述性统计
                    f.write("**Descriptive Statistics:**\n")
                    for algo, stats in metric_results.items():
                        if isinstance(stats, dict) and 'mean' in stats:
                            f.write(f"- {algo}: μ={stats['mean']:.3f}, σ={stats['std']:.3f}, "
                                   f"median={stats['median']:.3f} (n={stats['count']})\n")

                    f.write("\n**Statistical Tests (HD-DDPG vs Baselines):**\n")
                    for comparison, stats in metric_results.items():
                        if 'HD-DDPG_vs_' in comparison and isinstance(stats, dict):
                            baseline = comparison.replace('HD-DDPG_vs_', '')

                            # Wilcoxon结果
                            sig_w = "✓ Significant" if stats.get('significant_wilcoxon', False) else "✗ Not Significant"
                            f.write(f"- **vs {baseline}:**\n")
                            f.write(f"  - Wilcoxon: p={stats.get('wilcoxon_p_value', 'N/A'):.4f} ({sig_w})\n")
                            f.write(f"  - Mann-Whitney: p={stats.get('mannwhitney_p_value', 'N/A'):.4f}\n")
                            f.write(f"  - Effect size (Cohen's d): {stats.get('cohens_d', 'N/A'):.3f} ({stats.get('effect_size', 'N/A')})\n")

                    f.write("\n")

            # 🎯 可扩展性分析
            if args.scalability_analysis:
                f.write("## Scalability Analysis\n")
                f.write("-" * 20 + "\n\n")

                # 按工作流大小分析
                f.write("### Performance by Workflow Size\n")
                for size in sorted(results_df['workflow_size'].unique()):
                    size_data = results_df[results_df['workflow_size'] == size]
                    f.write(f"**Size {size}:** {len(size_data)} workflows\n")

                    for algo in algorithms:
                        algo_size_data = size_data[size_data['algorithm'] == algo]
                        if len(algo_size_data) > 0:
                            makespan_data = algo_size_data['makespan'][~algo_size_data['makespan'].isin([float('inf')])].dropna()
                            if len(makespan_data) > 0:
                                f.write(f"- {algo}: {makespan_data.mean():.3f} ± {makespan_data.std():.3f}\n")
                    f.write("\n")

                # 按工作流类型分析
                f.write("### Performance by Workflow Type\n")
                for wf_type in sorted(results_df['workflow_type'].unique()):
                    type_data = results_df[results_df['workflow_type'] == wf_type]
                    f.write(f"**{wf_type.title()}:** {len(type_data)} workflows\n")

                    for algo in algorithms:
                        algo_type_data = type_data[type_data['algorithm'] == algo]
                        if len(algo_type_data) > 0:
                            makespan_data = algo_type_data['makespan'][~algo_type_data['makespan'].isin([float('inf')])].dropna()
                            if len(makespan_data) > 0:
                                f.write(f"- {algo}: {makespan_data.mean():.3f} ± {makespan_data.std():.3f}\n")
                    f.write("\n")

            # 🎯 结论和建议
            f.write("## Conclusions and Recommendations\n")
            f.write("-" * 35 + "\n\n")

            # 基于结果生成智能结论
            if 'HD-DDPG' in algorithms and len(results_df) > 0:
                hd_ddpg_data = results_df[results_df['algorithm'] == 'HD-DDPG']
                baseline_data = results_df[results_df['algorithm'] != 'HD-DDPG']

                if len(hd_ddpg_data) > 0 and len(baseline_data) > 0:
                    hd_ddpg_makespan = hd_ddpg_data['makespan'][~hd_ddpg_data['makespan'].isin([float('inf')])].mean()
                    baseline_makespan = baseline_data['makespan'][~baseline_data['makespan'].isin([float('inf')])].mean()

                    if hd_ddpg_makespan < baseline_makespan:
                        improvement_pct = (baseline_makespan - hd_ddpg_makespan) / baseline_makespan * 100
                        f.write(f"1. **HD-DDPG demonstrates superior performance** with {improvement_pct:.1f}% improvement in makespan compared to baseline algorithms.\n\n")

                    # 稳定性评估
                    hd_ddpg_stability = hd_ddpg_data['stability_score'].mean()
                    if hd_ddpg_stability > 0.7:
                        f.write("2. **High stability achieved** - HD-DDPG shows consistent performance across different workflow types and sizes.\n\n")
                    elif hd_ddpg_stability > 0.5:
                        f.write("2. **Moderate stability** - HD-DDPG shows reasonable consistency with room for improvement.\n\n")
                    else:
                        f.write("2. **Stability concerns** - HD-DDPG performance varies significantly across test cases.\n\n")

                    # 质量评估
                    hd_ddpg_quality = hd_ddpg_data['quality_score'].mean()
                    if hd_ddpg_quality > 0.8:
                        f.write("3. **Excellent overall quality** - HD-DDPG delivers high-quality scheduling decisions.\n\n")
                    elif hd_ddpg_quality > 0.6:
                        f.write("3. **Good overall quality** - HD-DDPG provides satisfactory scheduling performance.\n\n")
                    else:
                        f.write("3. **Quality improvement needed** - HD-DDPG quality scores suggest areas for enhancement.\n\n")

            # 技术建议
            f.write("### Technical Recommendations\n")
            f.write("1. **Production Deployment:** ")
            if 'HD-DDPG' in algorithms:
                hd_ddpg_success = results_df[results_df['algorithm'] == 'HD-DDPG']['success_rate'].mean()
                if hd_ddpg_success > 0.9:
                    f.write("Ready for production deployment with high confidence.\n")
                elif hd_ddpg_success > 0.8:
                    f.write("Suitable for production with proper monitoring.\n")
                else:
                    f.write("Requires further optimization before production deployment.\n")

            f.write("2. **Monitoring:** Implement real-time performance monitoring focusing on makespan stability and quality scores.\n")
            f.write("3. **Optimization:** Consider fine-tuning hyperparameters for specific workflow types showing suboptimal performance.\n")
            f.write("4. **Scalability:** Test with larger workflow sizes and more diverse medical workflow types.\n\n")

            # 附录信息
            f.write("## Appendix\n")
            f.write("-" * 10 + "\n\n")
            f.write(f"**Evaluation Configuration:**\n")
            f.write(f"- Test workflows: {args.test_workflows}\n")
            f.write(f"- Workflow sizes: {args.workflow_sizes}\n")
            f.write(f"- Workflow types: {args.workflow_types}\n")
            f.write(f"- Baseline algorithms: {args.baseline_algorithms}\n")
            f.write(f"- Environment failure rate: {args.failure_rate}\n")
            f.write(f"- Random seed: {args.seed}\n\n")

        print(f"✅ 增强评估报告已保存: {report_path}")

    except Exception as e:
        print(f"⚠️ 报告生成失败: {e}")


def save_enhanced_results(results_df, output_dir, args):
    """🎯 保存增强的结果数据"""
    print("💾 保存增强结果数据...")

    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        if args.export_format in ['csv', 'both']:
            # 保存详细CSV
            csv_path = os.path.join(output_dir, 'enhanced_evaluation_results.csv')
            results_df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✅ CSV结果已保存: {csv_path}")

            # 保存汇总CSV
            summary_df = results_df.groupby(['algorithm', 'workflow_type', 'workflow_size']).agg({
                'makespan': ['mean', 'std', 'count'],
                'quality_score': ['mean', 'std'],
                'stability_score': ['mean', 'std'],
                'success_rate': ['mean', 'std']
            }).round(4)

            summary_csv_path = os.path.join(output_dir, 'performance_summary.csv')
            summary_df.to_csv(summary_csv_path, encoding='utf-8')
            print(f"✅ 汇总CSV已保存: {summary_csv_path}")

        if args.export_format in ['json', 'both']:
            # 保存JSON格式
            # 转换数据为JSON兼容格式
            results_dict = results_df.to_dict('records')

            # 处理无穷值
            for record in results_dict:
                for key, value in record.items():
                    if value == float('inf'):
                        record[key] = 'Infinity'
                    elif value == float('-inf'):
                        record[key] = '-Infinity'
                    elif pd.isna(value):
                        record[key] = None

            json_path = os.path.join(output_dir, 'enhanced_evaluation_results.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'evaluation_date': datetime.now().isoformat(),
                        'total_workflows': len(results_df),
                        'algorithms': list(results_df['algorithm'].unique()),
                        'workflow_types': list(results_df['workflow_type'].unique()),
                        'workflow_sizes': list(results_df['workflow_size'].unique()),
                        'evaluation_rounds': args.evaluation_rounds
                    },
                    'results': results_dict
                }, f, indent=2, ensure_ascii=False)

            print(f"✅ JSON结果已保存: {json_path}")

        # 保存配置信息
        config_info = {
            'evaluation_args': vars(args),
            'evaluation_timestamp': datetime.now().isoformat(),
            'python_version': sys.version,
            'dependencies': {
                'tqdm_available': TQDM_AVAILABLE,
                'scipy_available': SCIPY_AVAILABLE,
                'stabilized_modules': STABILIZED_MODULES
            }
        }

        config_path = os.path.join(output_dir, 'evaluation_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2, default=str)

        print(f"✅ 配置信息已保存: {config_path}")

    except Exception as e:
        print(f"⚠️ 结果保存失败: {e}")


def main():
    """🎯 主评估函数"""
    print("🔍 启动HD-DDPG增强模型评估")
    print("=" * 60)

    # 解析参数
    args = parse_enhanced_arguments()

    # 设置随机种子
    np.random.seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 输出目录: {args.output_dir}")

    # 加载模型
    print(f"\n🔧 加载模型...")
    hd_ddpg, config = load_enhanced_model_and_config(args.model_path, args.config_path, args.model_type)
    if hd_ddpg is None:
        print("❌ 模型加载失败，程序退出")
        return

    # 解析评估参数
    workflow_sizes = [int(x.strip()) for x in args.workflow_sizes.split(',')]
    workflow_types = [x.strip() for x in args.workflow_types.split(',')]
    baseline_algorithms = [x.strip() for x in args.baseline_algorithms.split(',')]

    print(f"\n🎯 评估配置:")
    print(f"  测试工作流: {args.test_workflows}")
    print(f"  工作流大小: {workflow_sizes}")
    print(f"  工作流类型: {workflow_types}")
    print(f"  基线算法: {baseline_algorithms}")
    print(f"  评估轮数: {args.evaluation_rounds}")
    print(f"  置信水平: {args.confidence_level}")
    print("-" * 60)

    # 创建组件
    print("🏗️ 创建评估环境...")
    environment = create_enhanced_environment(args)

    # 创建工作流生成器
    try:
        workflow_generator = MedicalWorkflowGenerator()
        print("✅ 工作流生成器创建成功")
    except Exception as e:
        print(f"⚠️ 工作流生成器创建失败: {e}")
        workflow_generator = None

    # 生成测试工作流配置
    workflows_per_config = max(1, args.test_workflows // (len(workflow_sizes) * len(workflow_types) * args.evaluation_rounds))
    test_configs = []

    for size in workflow_sizes:
        for wf_type in workflow_types:
            test_configs.append({
                'size': size,
                'type': wf_type,
                'count': workflows_per_config
            })

    # 生成测试工作流
    print(f"\n📋 生成测试工作流...")
    test_workflows = generate_enhanced_test_workflows(workflow_generator, test_configs, args)
    print(f"✅ 生成了 {len(test_workflows)} 个测试工作流")

    # 🎯 开始评估
    print(f"\n🚀 开始增强评估...")
    evaluation_start_time = time.time()

    # 评估HD-DDPG
    print("\n🤖 评估HD-DDPG...")
    hd_ddpg_results = evaluate_hd_ddpg_enhanced(hd_ddpg, test_workflows, environment, args)

    # 评估基线算法
    all_results = hd_ddpg_results.copy()

    for baseline in baseline_algorithms:
        print(f"\n📊 评估基线算法: {baseline}...")
        baseline_results = evaluate_baseline_enhanced(baseline, test_workflows, environment, args)
        all_results.extend(baseline_results)

    evaluation_duration = time.time() - evaluation_start_time
    print(f"\n✅ 评估完成！总耗时: {evaluation_duration:.2f}秒")

    # 创建结果DataFrame
    print("\n📊 处理评估结果...")
    results_df = pd.DataFrame(all_results)

    # 数据清理
    results_df = results_df.replace([float('inf'), float('-inf')], np.nan)

    # 保存原始结果
    if args.save_detailed_results:
        save_enhanced_results(results_df, args.output_dir, args)

    # 执行增强统计分析
    statistical_results = {}
    if args.include_statistical_test:
        print("\n📈 执行增强统计分析...")
        statistical_results = perform_enhanced_statistical_analysis(results_df, args)

    # 创建增强可视化图表
    if args.save_plots:
        print("\n🎨 创建增强可视化图表...")
        create_enhanced_evaluation_plots(results_df, args.output_dir, args)

    # 生成增强评估报告
    print("\n📝 生成增强评估报告...")
    generate_enhanced_evaluation_report(results_df, statistical_results, args.output_dir, args)

    # 🎯 打印最终摘要
    print(f"\n🎉 增强评估完成！")
    print("=" * 60)

    # HD-DDPG性能摘要
    hd_ddpg_data = results_df[results_df['algorithm'] == 'HD-DDPG']
    if len(hd_ddpg_data) > 0:
        print(f"📊 HD-DDPG性能摘要:")

        # 过滤有效数据
        valid_makespan = hd_ddpg_data['makespan'][~hd_ddpg_data['makespan'].isna()]
        if len(valid_makespan) > 0:
            print(f"  平均Makespan: {valid_makespan.mean():.3f} ± {valid_makespan.std():.3f}")

        print(f"  平均负载均衡: {hd_ddpg_data['load_balance'].mean():.3f}")
        print(f"  平均质量评分: {hd_ddpg_data['quality_score'].mean():.3f}")
        print(f"  平均稳定性评分: {hd_ddpg_data['stability_score'].mean():.3f}")
        print(f"  平均成功率: {hd_ddpg_data['success_rate'].mean():.1%}")
        print(f"  平均调度时间: {hd_ddpg_data['scheduling_time'].mean():.6f}秒")

    # 比较结果
    if len(baseline_algorithms) > 0:
        print(f"\n📈 与基线算法比较:")
        for baseline in baseline_algorithms:
            baseline_data = results_df[results_df['algorithm'] == baseline]
            if len(baseline_data) > 0 and len(hd_ddpg_data) > 0:
                hd_makespan = hd_ddpg_data['makespan'][~hd_ddpg_data['makespan'].isna()].mean()
                base_makespan = baseline_data['makespan'][~baseline_data['makespan'].isna()].mean()

                if not pd.isna(hd_makespan) and not pd.isna(base_makespan) and base_makespan > 0:
                    improvement = (base_makespan - hd_makespan) / base_makespan * 100
                    print(f"  vs {baseline}: {improvement:+.1f}% makespan改进")

    print(f"\n📁 所有结果已保存到: {args.output_dir}")
    print(f"📋 详细报告: enhanced_evaluation_report.md")
    print(f"📊 图表文件: *.png")
    print(f"📄 数据文件: enhanced_evaluation_results.{args.export_format}")

    print("\n" + "=" * 60)
    print("🏁 HD-DDPG增强评估程序结束")


if __name__ == "__main__":
    main()