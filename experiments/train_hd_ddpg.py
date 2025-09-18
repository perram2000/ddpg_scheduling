"""
HD-DDPG Training Script - 高效优化版本 + 错误修复 + 性能优化
HD-DDPG训练主脚本 - 配合核心算法优化

主要优化：
- 🚀 增加训练轮数和改进参数
- 🎯 更现实的makespan目标
- 📈 课程学习和学习率调度
- 🔧 增强的探索策略
- 💪 更大的批次大小和缓冲区
- 📊 改进的监控和评估
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
import copy
warnings.filterwarnings('ignore')

# 导入tqdm用于进度条显示
try:
    from tqdm import tqdm, trange
    TQDM_AVAILABLE = True
    print("INFO: tqdm module loaded successfully")
except ImportError:
    print("WARNING: tqdm not installed, using fallback progress display")
    TQDM_AVAILABLE = False
    # 创建简单的替代品
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", unit="", ncols=100, **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0
            if iterable:
                self.total = len(iterable)
            print(f"Starting {desc} (Total: {self.total})")

        def __iter__(self):
            for item in self.iterable:
                yield item
                self.update(1)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            self.n += n
            if self.total and self.n % max(1, self.total // 10) == 0:
                progress = (self.n / self.total) * 100
                print(f"  Progress: {progress:.1f}% ({self.n}/{self.total})")

        def set_postfix(self, **kwargs):
            info = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            print(f"  Status: {info}")

        def set_description(self, desc):
            self.desc = desc

        def close(self):
            pass

    def trange(*args, **kwargs):
        return tqdm(range(*args), **kwargs)

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入优化后的模块
try:
    from src.algorithms.hd_ddpg import HDDDPG
    from src.environment.medical_simulator import StabilizedMedicalSchedulingSimulator
    from src.environment.fog_cloud_env import StabilizedFogCloudEnvironment
    from src.environment.workflow_generator import MedicalWorkflowGenerator
    from src.utils.metrics import EnhancedSchedulingMetrics
    MODULES_AVAILABLE = True
    print("INFO: All optimized modules imported successfully")
except ImportError as e:
    print(f"WARNING: Optimized modules import failed: {e}")
    print("WARNING: Using simplified fallback version")
    MODULES_AVAILABLE = False


# 🚀 新增：课程学习管理器
class CurriculumLearningManager:
    """课程学习管理器 - 逐步增加训练难度"""

    def __init__(self, initial_workflows=4, max_workflows=10, initial_tasks=5, max_tasks=15):
        self.initial_workflows = initial_workflows
        self.max_workflows = max_workflows
        self.initial_tasks = initial_tasks
        self.max_tasks = max_tasks
        self.current_phase = 0
        self.episodes_per_phase = 50

    def get_current_difficulty(self, episode):
        """获取当前训练难度"""
        phase = min(episode // self.episodes_per_phase, 4)  # 最多5个阶段

        # 工作流数量
        workflow_count = min(
            self.initial_workflows + phase * 1,
            self.max_workflows
        )

        # 任务数量
        task_count = min(
            self.initial_tasks + phase * 2,
            self.max_tasks
        )

        return {
            'workflows_per_episode': workflow_count,
            'min_tasks': max(self.initial_tasks, task_count - 2),
            'max_tasks': min(self.max_tasks, task_count + 2),
            'complexity_level': ['simple', 'easy', 'medium', 'hard', 'expert'][phase],
            'phase': phase
        }


# 🚀 新增：学习率调度器
class AdaptiveLearningRateScheduler:
    """自适应学习率调度器"""

    def __init__(self, initial_lr=0.0003, min_lr=1e-6, decay_factor=0.995, patience=20):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.no_improvement_count = 0
        self.best_performance = float('inf')

    def update(self, current_performance, episode):
        """更新学习率"""
        # 基于性能的调整
        if current_performance < self.best_performance:
            self.best_performance = current_performance
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # 如果性能停滞，降低学习率
        if self.no_improvement_count >= self.patience:
            self.current_lr = max(self.current_lr * self.decay_factor, self.min_lr)
            self.no_improvement_count = 0
            return True  # 表示学习率已更新

        # 定期衰减
        if episode % 50 == 0 and episode > 0:
            self.current_lr = max(self.current_lr * 0.98, self.min_lr)
            return True

        return False

    def get_current_lr(self):
        return self.current_lr


# 🚀 新增：增强探索策略
class EnhancedExplorationStrategy:
    """增强探索策略"""

    def __init__(self, initial_noise=0.3, final_noise=0.05, decay_episodes=300):
        self.initial_noise = initial_noise
        self.final_noise = final_noise
        self.decay_episodes = decay_episodes
        self.current_noise = initial_noise

    def get_exploration_noise(self, episode, performance_trend='stable'):
        """获取当前探索噪声"""
        # 基本衰减
        progress = min(episode / self.decay_episodes, 1.0)
        base_noise = self.initial_noise - (self.initial_noise - self.final_noise) * progress

        # 根据性能趋势调整
        if performance_trend == 'improving':
            # 性能改善时减少探索
            self.current_noise = base_noise * 0.8
        elif performance_trend == 'declining':
            # 性能下降时增加探索
            self.current_noise = min(base_noise * 1.2, self.initial_noise)
        else:
            # 稳定时使用基本噪声
            self.current_noise = base_noise

        return max(self.current_noise, self.final_noise)


# 🚀 新增：FCFS训练监控器
class FCFSTrainingMonitor:
    """FCFS训练监控器 - 轻量级基线对比"""

    def __init__(self, comparison_frequency=20):
        self.comparison_frequency = comparison_frequency
        self.fcfs_history = []
        self.hd_ddpg_history = []
        self.improvement_history = []

    def should_compare(self, episode):
        """检查是否需要进行FCFS对比"""
        return episode % self.comparison_frequency == 0 and episode > 0

    # === 新增：估算任务数据量（MB） ===
    def _estimate_data_size_mb(self, task) -> float:
        # 优先使用任务类型映射；否则用内存需求作为近似
        type_map = {
            'IMAGE_PROCESSING': 120.0,
            'ML_INFERENCE': 80.0,
            'DATA_ANALYSIS': 40.0,
            'DATABASE_QUERY': 15.0,
            'REPORT_GENERATION': 20.0
        }
        task_type = getattr(task, 'task_type', None)
        if task_type in type_map:
            return float(type_map[task_type])

        mem_req = float(getattr(task, 'memory_requirement', 10.0))
        # 简单夹紧，避免极端值
        return max(5.0, min(200.0, mem_req * 1.0))

    # === 新增：获取/估算传输时间（秒） ===
    def _get_transmission_time(self, environment, from_type: str, to_type: str, data_size_mb: float) -> float:
        # 优先走环境接口（如 OptimizedFogCloudEnvironment）
        try:
            if hasattr(environment, 'get_transmission_time'):
                t = environment.get_transmission_time(from_type, to_type, data_size_mb)
                return float(max(0.0, t))
        except Exception:
            pass

        # 回退估计（与 fog_cloud_env 保持一致的简化参数）
        same_layer = (from_type == to_type)
        if same_layer:
            latency_ms = 0.1
            bandwidth_mbps = 10000.0  # 10 Gbps
        else:
            pair = {from_type, to_type}
            if pair == {'FPGA', 'FOG_GPU'}:
                latency_ms = 2.0
                bandwidth_mbps = 1000.0  # 1 Gbps
            elif pair == {'FOG_GPU', 'CLOUD'}:
                latency_ms = 20.0
                bandwidth_mbps = 500.0   # 500 Mbps
            else:  # 包括 {'FPGA', 'CLOUD'} 等默认情况
                latency_ms = 50.0
                bandwidth_mbps = 200.0   # 200 Mbps

        # 传输时间 = 基础延迟 + 数据传输时间(秒)
        transmission_time = latency_ms / 1000.0 + (data_size_mb * 8.0) / bandwidth_mbps
        return float(max(0.0, transmission_time))

    def quick_fcfs_comparison(self, workflow, environment):
        """快速FCFS对比 - 专为训练期间设计（已加入网络传输时间）"""
        try:
            start_time = time.time()

            task_completion_times = []
            total_energy = 0.0
            completed_tasks = 0
            current_time = 0.0

            # 获取所有可用节点（简化版本）
            all_nodes = self._get_all_available_nodes(environment)
            if not all_nodes:
                return {'makespan': float('inf'), 'success_rate': 0, 'time': 0}

            node_index = 0

            # 新增：上一任务所在层（数据来源层），默认从边缘开始
            last_cluster = 'FPGA'

            # FCFS调度：按顺序轮流分配到节点
            for task in workflow:
                if all_nodes:
                    selected_node = all_nodes[node_index % len(all_nodes)]
                    node_index += 1

                    # 计算执行时间（已有）
                    comp_req = float(getattr(task, 'computation_requirement', 1.0))
                    exec_time = float(self._calculate_execution_time(selected_node, comp_req))

                    # 新增：计算传输时间
                    to_cluster = getattr(selected_node, 'cluster_type', 'CLOUD')
                    data_size_mb = self._estimate_data_size_mb(task)
                    trans_time = float(self._get_transmission_time(environment, last_cluster, to_cluster, data_size_mb))

                    # 完成时间 = 当前时间 + 传输 + 执行（保持串行推进）
                    completion_time = current_time + trans_time + exec_time
                    task_completion_times.append(completion_time)

                    # 能耗（保持原先的简化计算，不计网络能耗）
                    energy = float(self._calculate_energy(selected_node, exec_time))
                    total_energy += energy

                    # 推进时间线与来源层
                    current_time = completion_time
                    last_cluster = to_cluster
                    completed_tasks += 1

            # 指标
            makespan = max(task_completion_times) if task_completion_times else float('inf')
            success_rate = completed_tasks / len(workflow) if workflow else 0
            comparison_time = time.time() - start_time

            return {
                'makespan': makespan,
                'success_rate': success_rate,
                'total_energy': total_energy,
                'completed_tasks': completed_tasks,
                'time': comparison_time
            }

        except Exception as e:
            print(f"WARNING: FCFS comparison failed: {e}")
            return {'makespan': float('inf'), 'success_rate': 0, 'time': 0}

    def _get_all_available_nodes(self, environment):
        """获取所有可用节点 - 简化版本"""
        try:
            all_nodes = []
            for cluster_type in ['FPGA', 'FOG_GPU', 'CLOUD']:
                if hasattr(environment, 'get_available_nodes'):
                    nodes = environment.get_available_nodes(cluster_type)
                    if nodes:
                        all_nodes.extend(nodes)
                else:
                    mock_nodes = self._create_simple_mock_nodes(cluster_type)
                    all_nodes.extend(mock_nodes)
            return all_nodes if all_nodes else self._create_fallback_nodes()
        except Exception:
            return self._create_fallback_nodes()

    def _create_simple_mock_nodes(self, cluster_type):
        """创建简单的模拟节点"""
        class SimpleFCFSNode:
            def __init__(self, node_id, cluster_type):
                self.node_id = node_id
                self.cluster_type = cluster_type
                self.base_time = {'FPGA': 0.8, 'FOG_GPU': 0.5, 'CLOUD': 0.3}[cluster_type]
                self.base_energy = {'FPGA': 12, 'FOG_GPU': 25, 'CLOUD': 45}[cluster_type]
        node_counts = {'FPGA': 4, 'FOG_GPU': 3, 'CLOUD': 1}
        count = node_counts.get(cluster_type, 2)
        return [SimpleFCFSNode(f"{cluster_type}_{i}", cluster_type) for i in range(count)]

    def _create_fallback_nodes(self):
        """创建回退节点"""
        fallback_nodes = []
        for i in range(8):
            node = type('FallbackNode', (), {
                'node_id': f'fallback_{i}',
                'cluster_type': 'CLOUD',
                'base_time': 0.5,
                'base_energy': 25
            })()
            fallback_nodes.append(node)
        return fallback_nodes

    def _calculate_execution_time(self, node, comp_requirement):
        """计算执行时间"""
        try:
            if hasattr(node, 'base_time'):
                return node.base_time * comp_requirement
            else:
                cluster_times = {'FPGA': 0.8, 'FOG_GPU': 0.5, 'CLOUD': 0.3}
                cluster_type = getattr(node, 'cluster_type', 'CLOUD')
                return cluster_times.get(cluster_type, 0.5) * comp_requirement
        except Exception:
            return 1.0

    def _calculate_energy(self, node, execution_time):
        """计算能耗"""
        try:
            if hasattr(node, 'base_energy'):
                return node.base_energy * execution_time
            else:
                cluster_energy = {'FPGA': 12, 'FOG_GPU': 25, 'CLOUD': 45}
                cluster_type = getattr(node, 'cluster_type', 'CLOUD')
                return cluster_energy.get(cluster_type, 25) * execution_time
        except Exception:
            return 25.0

    def analyze_improvement(self, hd_ddpg_result, fcfs_result):
        """分析HD-DDPG相对于FCFS的改进"""
        try:
            hd_ddpg_makespan = hd_ddpg_result.get('avg_makespan', float('inf'))
            fcfs_makespan = fcfs_result.get('makespan', float('inf'))

            if hd_ddpg_makespan != float('inf'):
                self.hd_ddpg_history.append(hd_ddpg_makespan)
            if fcfs_makespan != float('inf'):
                self.fcfs_history.append(fcfs_makespan)

            if fcfs_makespan != float('inf') and hd_ddpg_makespan != float('inf') and fcfs_makespan > 0:
                improvement = (fcfs_makespan - hd_ddpg_makespan) / fcfs_makespan * 100
                self.improvement_history.append(improvement)
                return {
                    'improvement_percentage': improvement,
                    'hd_ddpg_makespan': hd_ddpg_makespan,
                    'fcfs_makespan': fcfs_makespan,
                    'comparison_time': fcfs_result.get('time', 0)
                }

            return None
        except Exception as e:
            print(f"WARNING: Improvement analysis failed: {e}")
            return None

    def get_training_summary(self):
        """获取训练期间的对比摘要"""
        try:
            if not self.improvement_history:
                return "No FCFS comparisons performed"

            recent_improvements = self.improvement_history[-10:]
            avg_improvement = np.mean(recent_improvements)

            if self.hd_ddpg_history and self.fcfs_history:
                avg_hd_ddpg = np.mean(self.hd_ddpg_history[-10:])
                avg_fcfs = np.mean(self.fcfs_history[-10:])
                return {
                    'average_improvement': avg_improvement,
                    'recent_hd_ddpg_makespan': avg_hd_ddpg,
                    'recent_fcfs_makespan': avg_fcfs,
                    'total_comparisons': len(self.improvement_history),
                    'consistent_improvement': sum(1 for x in recent_improvements if x > 0) / len(recent_improvements)
                }
            return {'average_improvement': avg_improvement}
        except Exception:
            return "Summary calculation failed"


# 🚀 优化的全局损失追踪器
class OptimizedLossTracker:
    """优化的损失追踪器 - 减少内存使用"""
    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有损失记录"""
        self.training_losses = {
            'meta_critic': [],
            'meta_actor': [],
            'fpga_critic': [],
            'fpga_actor': [],
            'fog_gpu_critic': [],
            'fog_gpu_actor': [],
            'cloud_critic': [],
            'cloud_actor': [],
            'training_steps': [],
            'episodes': []
        }
        self.current_episode = 0
        self.training_step = 0

    def log_meta_losses(self, critic_loss, actor_loss, episode):
        """记录Meta Controller损失"""
        if critic_loss is not None and not (np.isnan(critic_loss) or np.isinf(critic_loss)):
            self.training_losses['meta_critic'].append(float(critic_loss))
            self.training_losses['training_steps'].append(self.training_step)
            self.training_losses['episodes'].append(episode)
            self.training_step += 1

        if actor_loss is not None and not (np.isnan(actor_loss) or np.isinf(actor_loss)):
            self.training_losses['meta_actor'].append(float(actor_loss))

    def log_sub_losses(self, cluster_name, critic_loss, actor_loss):
        """记录Sub Controller损失"""
        cluster_key = cluster_name.lower().replace('-', '_')

        if critic_loss is not None and not (np.isnan(critic_loss) or np.isinf(critic_loss)):
            if f'{cluster_key}_critic' in self.training_losses:
                self.training_losses[f'{cluster_key}_critic'].append(float(critic_loss))

        if actor_loss is not None and not (np.isnan(actor_loss) or np.isinf(actor_loss)):
            if f'{cluster_key}_actor' in self.training_losses:
                self.training_losses[f'{cluster_key}_actor'].append(float(actor_loss))

    def get_loss_data(self):
        """获取损失数据"""
        return self.training_losses.copy()

    def save_loss_data(self, save_path):
        """保存损失数据到CSV"""
        try:
            import pandas as pd

            # 构建损失数据DataFrame
            max_length = max(len(v) for v in self.training_losses.values() if isinstance(v, list) and v)

            if max_length > 0:
                loss_data = {}

                # 填充数据到相同长度
                for key, values in self.training_losses.items():
                    if isinstance(values, list):
                        # 🚀 只保留最新的1000个记录，减少内存
                        values = values[-1000:] if len(values) > 1000 else values
                        filled_values = values + [np.nan] * (min(max_length, 1000) - len(values))
                        loss_data[key] = filled_values[:min(max_length, 1000)]

                loss_df = pd.DataFrame(loss_data)
                csv_path = os.path.join(save_path, "training_losses.csv")
                loss_df.to_csv(csv_path, index=False)
                return csv_path
        except ImportError:
            print("WARNING: pandas not available, skipping loss CSV save")
        except Exception as e:
            print(f"WARNING: Loss data save error: {e}")
        return None

# 创建全局损失追踪器实例
GLOBAL_LOSS_TRACKER = OptimizedLossTracker()


def setup_optimized_environment():
    """设置优化的训练环境"""
    print("INFO: Configuring optimized training environment...")

    # 禁用GPU，强制使用CPU
    tf.config.set_visible_devices([], 'GPU')

    # 🚀 优化的CPU设置
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)

    # 内存增长设置
    physical_devices = tf.config.list_physical_devices('CPU')
    print(f"INFO: Detected {len(physical_devices)} CPU devices")

    # 设置TensorFlow日志级别
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print(f"INFO: TensorFlow version: {tf.__version__}")
    print(f"INFO: CPU training mode enabled")

    return True


def convert_to_serializable(obj):
    """简化的类型转换函数"""
    if isinstance(obj, (np.float32, np.float64, float)):
        val = float(obj)
        return val if not (np.isnan(val) or np.isinf(val)) else 0.0
    elif isinstance(obj, (np.int32, np.int64, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'numpy'):  # TensorFlow tensors
        try:
            return float(obj.numpy()) if obj.shape == () else obj.numpy().tolist()
        except:
            return str(obj)
    elif obj is None:
        return None
    return obj


def parse_optimized_arguments():
    """解析优化的命令行参数"""
    parser = argparse.ArgumentParser(
        description='HD-DDPG Medical Workflow Scheduling Training (High-Performance Optimized Version)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 🚀 核心训练参数 - 大幅优化
    training_group = parser.add_argument_group('Core Training Parameters')
    training_group.add_argument('--episodes', type=int, default=500,  # 🚀 增加训练轮数
                        help='Number of training episodes')
    training_group.add_argument('--workflows-per-episode', type=int, default=8,  # 🚀 增加工作流数量
                        help='Number of workflows per episode')
    training_group.add_argument('--batch-size', type=int, default=128,  # 🚀 更大批次
                        help='Batch size for training')
    training_group.add_argument('--learning-rate', type=float, default=0.0003,  # 🚀 优化学习率
                        help='Initial learning rate')
    training_group.add_argument('--memory-capacity', type=int, default=50000,  # 🚀 更大缓冲区
                        help='Replay buffer capacity')

    # 环境参数
    env_group = parser.add_argument_group('Environment Parameters')
    env_group.add_argument('--min-workflow-size', type=int, default=5,
                        help='Minimum workflow size')
    env_group.add_argument('--max-workflow-size', type=int, default=15,  # 🚀 增加最大任务数
                        help='Maximum workflow size')
    env_group.add_argument('--failure-rate', type=float, default=0.005,
                        help='System failure rate')
    env_group.add_argument('--makespan-target', type=float, default=2.8,  # 🚀 更现实的目标
                        help='Target makespan')
    env_group.add_argument('--makespan-weight', type=float, default=0.85,  # 🚀 调整权重
                        help='Makespan weight in reward function')

    # 🚀 新增：高级训练参数
    advanced_group = parser.add_argument_group('Advanced Training Parameters')
    advanced_group.add_argument('--enable-curriculum', action='store_true', default=True,
                        help='Enable curriculum learning')
    advanced_group.add_argument('--enable-lr-schedule', action='store_true', default=True,
                        help='Enable adaptive learning rate scheduling')
    advanced_group.add_argument('--enable-enhanced-exploration', action='store_true', default=True,
                        help='Enable enhanced exploration strategy')
    advanced_group.add_argument('--noise-decay-episodes', type=int, default=400,
                        help='Episodes for noise decay')
    advanced_group.add_argument('--warmup-episodes', type=int, default=50,
                        help='Warmup episodes before full training')

    # 🚀 简化的稳定化参数
    stable_group = parser.add_argument_group('Optimization Parameters')
    stable_group.add_argument('--enable-per', action='store_true', default=True,
                        help='Enable prioritized experience replay')
    stable_group.add_argument('--quality-threshold', type=float, default=0.08,
                        help='Experience quality threshold')
    stable_group.add_argument('--action-smoothing', action='store_true', default=True,
                        help='Enable action smoothing')
    stable_group.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping norm')

    # 🚀 FCFS监控参数
    fcfs_group = parser.add_argument_group('FCFS Comparison Parameters')
    fcfs_group.add_argument('--fcfs-comparison-frequency', type=int, default=25,
                        help='FCFS comparison frequency (every N episodes)')
    fcfs_group.add_argument('--disable-fcfs-comparison', action='store_true',
                        help='Disable FCFS comparison during training')

    # 保存和加载
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--save-interval', type=int, default=50,  # 🚀 调整保存频率
                        help='Model save interval')
    io_group.add_argument('--eval-interval', type=int, default=25,  # 🚀 调整评估频率
                        help='Evaluation interval')
    io_group.add_argument('--load-model', type=str, default=None,
                        help='Path to load pre-trained model')
    io_group.add_argument('--save-dir', type=str, default='optimized_model_result',
                        help='Model save directory')

    # 实验设置
    exp_group = parser.add_argument_group('Experiment Settings')
    exp_group.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name')
    exp_group.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    exp_group.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    # 可视化参数
    viz_group = parser.add_argument_group('Visualization Parameters')
    viz_group.add_argument('--disable-progress-bar', action='store_true',
                        help='Disable tqdm progress bar')
    viz_group.add_argument('--save-plots', action='store_true', default=True,
                        help='Save training plots')

    # 损失追踪参数
    loss_group = parser.add_argument_group('Loss Tracking Parameters')
    loss_group.add_argument('--loss-logging-frequency', type=int, default=1,
                        help='Frequency of loss logging (every N episodes)')
    loss_group.add_argument('--enable-detailed-loss-tracking', action='store_true', default=True,
                        help='Enable detailed loss tracking for all controllers')

    # CPU优化参数
    cpu_group = parser.add_argument_group('CPU Optimization')
    cpu_group.add_argument('--cpu-threads', type=int, default=0,
                        help='Number of CPU threads (0=auto)')

    parser.add_argument("--reward-scale", type=float, default=None,
                        help="Global reward scaling factor (None=use internal default)")
    parser.add_argument("--makespan-improvement-bonus", type=float, default=None,
                        help="Bonus scale for makespan improvement (None=use internal default)")

    return parser.parse_args()


def setup_optimized_experiment(args):
    """设置优化的实验环境"""
    # 设置随机种子
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # CPU线程设置
    if args.cpu_threads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(args.cpu_threads)
        tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
        print(f"INFO: CPU threads set to: {args.cpu_threads}")

    # 创建实验目录
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"enhanced_hd_ddpg_{timestamp}"

    experiment_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)

    # 重置全局损失追踪器
    GLOBAL_LOSS_TRACKER.reset()

    # 保存实验配置
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        config_dict = vars(args).copy()
        config_dict['training_mode'] = 'HIGH_PERFORMANCE_ENHANCED'
        config_dict['tensorflow_version'] = tf.__version__
        json.dump(config_dict, f, indent=2)

    print(f"INFO: Experiment setup completed: {experiment_dir}")
    return experiment_dir


def create_optimized_config(args):
    """🚀 创建高性能优化的HD-DDPG配置"""
    config = {
        # 🚀 基础参数 - 大幅优化
        'meta_state_dim': 15,
        'meta_action_dim': 3,
        'gamma': 0.99,  # 🚀 提高折扣因子
        'batch_size': args.batch_size,
        'memory_capacity': args.memory_capacity,
        'meta_lr': args.learning_rate,
        'sub_lr': args.learning_rate * 0.7,  # 🚀 子控制器稍低学习率
        'tau': 0.005,  # 🚀 软更新参数
        'update_frequency': 1,
        'save_frequency': args.save_interval,

        # 🚀 高性能探索参数
        'gradient_clip_norm': args.gradient_clip,
        'exploration_noise': 0.25,  # 🚀 增加初始探索
        'noise_decay': 0.9992,  # 🚀 更慢的噪声衰减
        'min_noise': 0.03,  # 🚀 保持最小探索
        'reward_scale': 1.2,  # 🚀 奖励缩放
        'verbose': args.verbose,

        # 🎯 关键！makespan配置
        'action_smoothing': args.action_smoothing,
        'action_smoothing_alpha': 0.2,  # 🚀 增加动作平滑
        'makespan_weight': args.makespan_weight,
        'stability_weight': 0.1,
        'quality_weight': 0.05,
        'target_update_frequency': 2,

        # 🎯 优化的makespan目标
        'makespan_target': float(args.makespan_target),
        'makespan_improvement_bonus': 10.0,  # 🚀 增加改进奖励

        # 🎯 高效学习参数
        'enable_per': args.enable_per,
        'quality_threshold': args.quality_threshold,
        'td_error_clipping': True,
        'td_error_max': 1.5,  # 🚀 增加TD误差范围
        'adaptive_learning': True,
        'learning_rate_decay': True,
        'lr_decay_factor': 0.995,  # 🚀 更慢的学习率衰减
        'lr_decay_frequency': 75,

        # 🎯 训练稳定性
        'makespan_smoothing': True,
        'makespan_smoothing_window': 8,  # 🚀 增加平滑窗口
        'multi_level_smoothing': True,  # 🚀 启用多级平滑
        'rebound_detection': True,  # 🚀 启用反弹检测

        # 🎯 收敛参数
        'convergence_patience': 80,  # 🚀 增加耐心
        'convergence_threshold': 0.03,
        'early_stopping': True,
        'plateau_detection': True,  # 🚀 启用平台检测

        # 损失监控
        'track_losses': True,
        'loss_logging_frequency': args.loss_logging_frequency,
        'detailed_loss_tracking': args.enable_detailed_loss_tracking,
        'global_loss_tracker': GLOBAL_LOSS_TRACKER,

        # 🚀 新增：高级训练特性
        'warmup_episodes': args.warmup_episodes,
        'buffer_warmup': True,
        'progressive_batch_size': True,
        'dynamic_exploration': True,
        'performance_monitoring': True,
    }
    # === 透传命令行的奖励尺度与改进奖励（仅当提供时） ===
    if args.reward_scale is not None:
        try:
            config['reward_scale'] = float(args.reward_scale)
            print(f"INFO: Override reward_scale via CLI: {config['reward_scale']}")
        except Exception:
            print("WARNING: Invalid --reward-scale, keeping default")

    if args.makespan_improvement_bonus is not None:
        try:
            config['makespan_improvement_bonus'] = float(args.makespan_improvement_bonus)
            print(f"INFO: Override makespan_improvement_bonus via CLI: {config['makespan_improvement_bonus']}")
        except Exception:
            print("WARNING: Invalid --makespan-improvement-bonus, keeping default")

    # 若在 hd_ddpg.py 中启用了 force_internal_reward_scaling，想让CLI生效则显式关闭
    if args.reward_scale is not None or args.makespan_improvement_bonus is not None:
        config['force_internal_reward_scaling'] = False

    # 🎯 添加调试输出确认配置
    print(f"INFO: 🚀 High-Performance HD-DDPG configuration created")
    print(f"  - Makespan target: {config['makespan_target']} (Enhanced target)")
    print(f"  - Makespan weight: {config['makespan_weight']}")
    print(f"  - Batch size: {config['batch_size']} (Large batch)")
    print(f"  - Buffer size: {config['memory_capacity']} (Large buffer)")
    print(f"  - Learning rate: {config['meta_lr']:.6f}")
    print(f"  - Gradient clip: {config['gradient_clip_norm']}")
    print(f"  - Exploration noise: {config['exploration_noise']}")
    print(f"  - Multi-level smoothing: {config['multi_level_smoothing']}")

    return config


def moving_average(data, window_size):
    """简化的移动平均计算"""
    if len(data) < window_size:
        return data

    result = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        window_data = data[start_idx:i + 1]
        # 过滤无效值
        valid_data = [x for x in window_data if not (np.isnan(x) or np.isinf(x))]
        if valid_data:
            result.append(np.mean(valid_data))
        else:
            result.append(data[i])
    return result


def save_optimized_training_plots(history, experiment_dir, episode, logger, config):
    """
    🚀 保存优化的训练图表 - 2x3布局，清晰可读
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        episode_rewards = history.get('episode_rewards', [])
        loss_data = GLOBAL_LOSS_TRACKER.get_loss_data()

        if len(episode_rewards) > 10:
            # 🚀 创建优化的2x3图表布局 - 更大更清晰
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            episodes = range(1, len(episode_rewards) + 1)

            # === 第一行：核心性能指标 ===

            # [1] 训练奖励曲线
            axes[0, 0].plot(episodes, episode_rewards, alpha=0.6, color='blue', linewidth=1, label='Episode Rewards')
            if len(episode_rewards) > 20:
                ma_20 = moving_average(episode_rewards, 20)
                axes[0, 0].plot(episodes, ma_20, 'r-', linewidth=2.5, label='MA-20')
            axes[0, 0].set_title('Training Rewards', fontweight='bold', fontsize=14)
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # [2] 平均Makespan曲线
            makespans = history.get('makespans', [])
            if makespans:
                # 过滤无效值
                valid_makespans = []
                valid_episodes = []
                for i, m in enumerate(makespans):
                    if m != float('inf') and not np.isnan(m) and m > 0:
                        valid_makespans.append(m)
                        valid_episodes.append(i + 1)

                if valid_makespans and len(valid_makespans) > 5:
                    axes[0, 1].plot(valid_episodes, valid_makespans,
                                   alpha=0.6, color='green', linewidth=1, label='Makespan')

                    # 添加移动平均
                    if len(valid_makespans) > 10:
                        ma_makespan = moving_average(valid_makespans, 10)
                        axes[0, 1].plot(valid_episodes[:len(ma_makespan)], ma_makespan,
                                       'r-', linewidth=2.5, label='MA-10')

                    # 🚀 修复：使用传入的config参数
                    target_makespan = config.get('makespan_target', 8.0)
                    axes[0, 1].axhline(y=target_makespan, color='orange',
                                      linestyle='--', linewidth=2, label='Target')

                    axes[0, 1].set_title('Average Makespan', fontweight='bold', fontsize=14)
                    axes[0, 1].set_xlabel('Episode')
                    axes[0, 1].set_ylabel('Makespan')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                else:
                    axes[0, 1].text(0.5, 0.5, 'Insufficient Valid Makespan Data',
                                   ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
                    axes[0, 1].set_title('Average Makespan', fontweight='bold', fontsize=14)

            # [3] Makespan稳定性评分
            stability_scores = history.get('makespan_stability', [])
            if stability_scores and len(stability_scores) > 5:
                valid_stability = [s for s in stability_scores if not np.isnan(s)]
                if valid_stability:
                    axes[0, 2].plot(episodes[:len(valid_stability)], valid_stability,
                                   alpha=0.7, color='purple', linewidth=2, label='Stability Score')

                    # 添加移动平均
                    if len(valid_stability) > 10:
                        ma_stability = moving_average(valid_stability, 10)
                        axes[0, 2].plot(episodes[:len(ma_stability)], ma_stability,
                                       'orange', linewidth=2.5, label='MA-10')

                    axes[0, 2].set_title('Makespan Stability Score', fontweight='bold', fontsize=14)
                    axes[0, 2].set_xlabel('Episode')
                    axes[0, 2].set_ylabel('Stability Score')
                    axes[0, 2].set_ylim(0, 1)
                    axes[0, 2].legend()
                    axes[0, 2].grid(True, alpha=0.3)
                else:
                    axes[0, 2].text(0.5, 0.5, 'No Valid Stability Data',
                                   ha='center', va='center', transform=axes[0, 2].transAxes, fontsize=12)
                    axes[0, 2].set_title('Makespan Stability Score', fontweight='bold', fontsize=14)

            # === 第二行：损失和分析 ===

            # [4] Training Losses - Meta Controller
            meta_critic_losses = loss_data.get('meta_critic', [])
            meta_actor_losses = loss_data.get('meta_actor', [])

            if meta_critic_losses or meta_actor_losses:
                # Meta Critic Loss
                if meta_critic_losses and len(meta_critic_losses) > 5:
                    training_episodes = range(len(meta_critic_losses))
                    axes[1, 0].plot(training_episodes, meta_critic_losses,
                                   color='red', linewidth=1.5, alpha=0.8, label='Meta Critic')

                # Meta Actor Loss
                if meta_actor_losses and len(meta_actor_losses) > 5:
                    training_episodes = range(len(meta_actor_losses))
                    axes[1, 0].plot(training_episodes, meta_actor_losses,
                                   color='purple', linewidth=1.5, alpha=0.8, label='Meta Actor')

                axes[1, 0].set_title('Meta Controller Losses', fontweight='bold', fontsize=14)
                axes[1, 0].set_xlabel('Training Steps')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_yscale('log')
            else:
                axes[1, 0].text(0.5, 0.5, 'Meta Controller Loss Data Not Available',
                               ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
                axes[1, 0].set_title('Meta Controller Losses', fontweight='bold', fontsize=14)

            # [5] 子控制器损失 - 简化版本
            cluster_colors = {'fpga': 'red', 'fog_gpu': 'green', 'cloud': 'blue'}
            has_sub_data = False

            # 绘制子控制器损失
            for cluster, color in cluster_colors.items():
                critic_key = f'{cluster}_critic'
                if critic_key in loss_data and loss_data[critic_key]:
                    valid_losses = [l for l in loss_data[critic_key] if not (np.isnan(l) or np.isinf(l))]
                    if valid_losses and len(valid_losses) > 5:
                        loss_episodes = range(len(valid_losses))
                        axes[1, 1].plot(loss_episodes, valid_losses,
                                       alpha=0.7, color=color, linewidth=1.5,
                                       label=f'{cluster.upper()} Critic')
                        has_sub_data = True

            if has_sub_data:
                axes[1, 1].set_title('Sub-Controller Losses', fontweight='bold', fontsize=14)
                axes[1, 1].set_xlabel('Training Steps')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].legend(fontsize=10)
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_yscale('log')
            else:
                axes[1, 1].text(0.5, 0.5, 'Sub-Controller Loss Data Not Available',
                               ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].set_title('Sub-Controller Losses', fontweight='bold', fontsize=14)

            # [6] 负载均衡和成功率
            load_balances = history.get('load_balances', [])
            if load_balances:
                valid_lb = [lb for lb in load_balances if not np.isnan(lb)]
                if valid_lb:
                    axes[1, 2].plot(episodes[:len(valid_lb)], valid_lb,
                                   alpha=0.7, color='orange', linewidth=2, label='Load Balance')
                    if len(valid_lb) > 10:
                        ma_lb = moving_average(valid_lb, 10)
                        axes[1, 2].plot(episodes[:len(ma_lb)], ma_lb,
                                       'darkorange', linewidth=2.5, label='MA-10')

                    axes[1, 2].set_title('Load Balance', fontweight='bold', fontsize=14)
                    axes[1, 2].set_xlabel('Episode')
                    axes[1, 2].set_ylabel('Load Balance')
                    axes[1, 2].legend()
                    axes[1, 2].grid(True, alpha=0.3)
                    axes[1, 2].set_ylim(0, 1)

            # 🚀 调整布局 - 更好的间距
            plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)

            # 保存图表
            plot_dir = os.path.join(experiment_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f'enhanced_training_analysis_ep{episode}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            if logger.verbose:
                print(f"INFO: Enhanced training analysis plot saved: {plot_path}")

    except ImportError:
        print("WARNING: matplotlib not installed, skipping plot generation")
    except Exception as e:
        print(f"WARNING: Plot generation error: {e}")

def create_optimized_simulator_config(args):
    """创建优化的仿真器配置"""
    return {
        'simulation_episodes': args.episodes,
        'workflows_per_episode': args.workflows_per_episode,
        'min_workflow_size': args.min_workflow_size,
        'max_workflow_size': args.max_workflow_size,
        'workflow_types': ['radiology', 'pathology', 'general'],
        'failure_rate': args.failure_rate,
        'save_interval': args.save_interval,
        'evaluation_interval': args.eval_interval,

        # 🚀 高性能配置
        'stability_monitoring': True,
        'reward_normalization': True,
        'performance_tracking_window': 50,  # 🚀 增加跟踪窗口
        'convergence_threshold': 0.015,  # 🚀 更严格的收敛阈值
        'makespan_target': args.makespan_target,
        'advanced_metrics': True,
        'dynamic_difficulty': args.enable_curriculum,
    }


def create_optimized_logger(experiment_dir, verbose=False):
    """创建优化的日志记录器"""
    class OptimizedLogger:
        def __init__(self, experiment_dir, verbose):
            self.experiment_dir = experiment_dir
            self.verbose = verbose
            self.log_file = os.path.join(experiment_dir, 'logs', 'training.log')
            self.metrics_file = os.path.join(experiment_dir, 'logs', 'metrics.jsonl')

            # 创建日志文件
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            self._write_header()

        def _write_header(self):
            """写入日志头部"""
            header = f"""
{'='*80}
Enhanced HD-DDPG Medical Workflow Scheduling Training Log
Training Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Experiment Directory: {self.experiment_dir}
Training Mode: HIGH_PERFORMANCE_ENHANCED with Advanced Features
{'='*80}
"""
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(header)

        def log_episode(self, episode, result, duration):
            """记录episode信息 - 优化格式"""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 🚀 修复：安全获取结果值，避免target_gap错误
            try:
                total_reward = result.get('total_reward', 0)
                avg_makespan = result.get('avg_makespan', float('inf'))
                success_rate = result.get('success_rate', 0)
                stability_score = result.get('makespan_stability', 0)
                current_lr = result.get('current_lr', 0)

                # 🚀 修复：确保所有值都是有效的
                if np.isnan(total_reward) or np.isinf(total_reward):
                    total_reward = 0
                if np.isnan(avg_makespan) and avg_makespan != float('inf'):
                    avg_makespan = float('inf')
                if np.isnan(success_rate) or np.isinf(success_rate):
                    success_rate = 0
                if np.isnan(stability_score) or np.isinf(stability_score):
                    stability_score = 0
                if np.isnan(current_lr) or np.isinf(current_lr):
                    current_lr = 0

            except Exception as e:
                # 🚀 修复：如果结果解析失败，使用默认值
                print(f"WARNING: Result parsing failed in episode {episode}: {e}")
                total_reward = 0
                avg_makespan = float('inf')
                success_rate = 0
                stability_score = 0
                current_lr = 0

            # 格式化makespan显示
            makespan_str = f"{avg_makespan:6.2f}" if avg_makespan != float('inf') else "   INF"

            # 🚀 增强的控制台输出格式
            console_msg = (
                f"Ep {episode:4d} | "
                f"Reward: {total_reward:8.2f} | "
                f"Makespan: {makespan_str} | "
                f"Success: {success_rate:.1%} | "
                f"Stab: {stability_score:.3f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {duration:.2f}s"
            )

            # 🚀 详细的日志格式
            detailed_log = (
                f"[{timestamp}] Ep {episode:4d} | "
                f"R: {total_reward:8.2f} | "
                f"MS: {makespan_str} | "
                f"SR: {success_rate:.2f} | "
                f"ST: {stability_score:.3f} | "
                f"LR: {current_lr:.6f} | "
                f"T: {duration:.2f}s\n"
            )

            # 写入日志文件
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(detailed_log)

            # 🚀 简化的JSON格式指标
            metrics_entry = {
                'timestamp': timestamp,
                'episode': episode,
                'total_reward': convert_to_serializable(total_reward),
                'avg_makespan': convert_to_serializable(avg_makespan),
                'success_rate': convert_to_serializable(success_rate),
                'stability_score': convert_to_serializable(stability_score),
                'current_lr': convert_to_serializable(current_lr),
                'duration': duration
            }

            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics_entry) + '\n')

            # 控制台输出
            if self.verbose or episode % 15 == 0:  # 🚀 减少输出频率
                print(console_msg)

            return console_msg

        def log_fcfs_comparison(self, episode, improvement_analysis):
            """🚀 记录FCFS对比信息"""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if improvement_analysis:
                fcfs_log = (
                    f"[{timestamp}] FCFS_CMP Ep {episode:4d} | "
                    f"HD-DDPG: {improvement_analysis['hd_ddpg_makespan']:.3f} | "
                    f"FCFS: {improvement_analysis['fcfs_makespan']:.3f} | "
                    f"Improve: {improvement_analysis['improvement_percentage']:+.1f}% | "
                    f"Time: {improvement_analysis['comparison_time']:.3f}s\n"
                )

                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(fcfs_log)

        def log_curriculum_update(self, episode, difficulty_info):
            """🚀 新增：记录课程学习更新"""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            curriculum_log = (
                f"[{timestamp}] CURRICULUM Ep {episode:4d} | "
                f"Phase: {difficulty_info['phase']} | "
                f"Workflows: {difficulty_info['workflows_per_episode']} | "
                f"Tasks: {difficulty_info['min_tasks']}-{difficulty_info['max_tasks']} | "
                f"Level: {difficulty_info['complexity_level']}\n"
            )

            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(curriculum_log)

        def log_lr_update(self, episode, old_lr, new_lr):
            """🚀 新增：记录学习率更新"""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lr_log = (
                f"[{timestamp}] LR_UPDATE Ep {episode:4d} | "
                f"Old: {old_lr:.6f} | "
                f"New: {new_lr:.6f} | "
                f"Change: {((new_lr - old_lr) / old_lr * 100):+.2f}%\n"
            )

            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(lr_log)

        def log_evaluation(self, episode, eval_result):
            """记录评估信息"""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            eval_log = f"[{timestamp}] === Evaluation Episode {episode} ===\n"
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(eval_log)

        def log_error(self, episode, error_msg):
            """记录错误信息"""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_log = f"[{timestamp}] ERROR Ep {episode}: {error_msg}\n"
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(error_log)
            print(error_log.strip())

        def log_convergence(self, episode, convergence_info):
            """记录收敛信息"""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conv_log = f"[{timestamp}] CONVERGENCE Ep {episode}: {convergence_info}\n"
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(conv_log)
            print(conv_log.strip())

    return OptimizedLogger(experiment_dir, verbose)


def save_optimized_training_metrics(simulator, experiment_dir, episode, logger, config):
    """保存优化的训练指标 - 修复版本"""
    try:
        # 🚀 简化的训练摘要获取
        try:
            if hasattr(simulator, 'hd_ddpg') and hasattr(simulator.hd_ddpg, 'get_training_summary'):
                training_summary = simulator.hd_ddpg.get_training_summary()
            else:
                training_summary = {
                    'training_history': getattr(simulator, 'training_history', {
                        'episode_rewards': [],
                        'makespans': [],
                        'load_balances': [],
                        'makespan_stability': []
                    })
                }
        except Exception as e:
            logger.log_error(episode, f"Failed to get training summary: {e}")
            training_summary = {'training_history': {}}

        history = training_summary.get('training_history', {})

        # 🚀 简化的CSV数据保存
        try:
            import pandas as pd

            episodes_count = len(history.get('episode_rewards', []))
            if episodes_count > 0:
                # 🚀 只保存核心指标
                metrics_data = {
                    'episode': range(1, episodes_count + 1),
                    'reward': history.get('episode_rewards', [])[:episodes_count],
                    'makespan': history.get('makespans', [])[:episodes_count],
                    'load_balance': history.get('load_balances', [])[:episodes_count],
                    'stability': history.get('makespan_stability', [0] * episodes_count)[:episodes_count]
                }

                # 🚀 只在关键节点保存详细数据
                if episode % (config.get('save_interval', 50)) == 0:
                    metrics_df = pd.DataFrame(metrics_data)
                    csv_path = os.path.join(experiment_dir, f'enhanced_metrics_ep{episode}.csv')
                    metrics_df.to_csv(csv_path, index=False)

                    if logger.verbose:
                        print(f"INFO: Training metrics saved: {csv_path}")

        except ImportError:
            pass
        except Exception as e:
            logger.log_error(episode, f"CSV save error: {e}")

        # 🚀 保存优化的训练图表 - 传递config参数
        try:
            save_optimized_training_plots(history, experiment_dir, episode, logger, config)
        except Exception as e:
            logger.log_error(episode, f"Plot save error: {e}")

    except Exception as e:
        logger.log_error(episode, f"Save training metrics error: {e}")


def evaluate_optimized_model(simulator, experiment_dir, episode, logger):
    """优化的模型评估"""
    eval_start_time = time.time()

    try:
        # 🚀 增强的评估逻辑
        recent_episodes = 30  # 🚀 增加评估窗口

        if hasattr(simulator, 'training_history'):
            history = simulator.training_history
        else:
            history = {'episode_rewards': [], 'makespans': [], 'load_balances': []}

        recent_rewards = history.get('episode_rewards', [])[-recent_episodes:]
        recent_makespans = [m for m in history.get('makespans', [])[-recent_episodes:]
                           if m != float('inf')]

        eval_result = {
            'episode': episode,
            'recent_performance': {
                'avg_reward': float(np.mean(recent_rewards)) if recent_rewards else 0,
                'avg_makespan': float(np.mean(recent_makespans)) if recent_makespans else float('inf'),
                'makespan_stability': float(np.var(recent_makespans)) if len(recent_makespans) > 5 else 0,
                'reward_trend': float(np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]) if len(recent_rewards) > 10 else 0,
            },
            'evaluation_duration': time.time() - eval_start_time,
        }

        # 记录评估结果
        logger.log_evaluation(episode, eval_result['recent_performance'])

        # 🚀 在关键节点保存评估文件
        if episode % (50) == 0:  # 调整为每50轮保存一次
            eval_path = os.path.join(experiment_dir, f'evaluation_ep{episode}.json')
            with open(eval_path, 'w') as f:
                json.dump(eval_result, f, indent=2, default=str)

        return eval_result

    except Exception as e:
        logger.log_error(episode, f"Evaluation error: {e}")
        return {'episode': episode, 'error': str(e)}


def create_fallback_simulator(config):
    """创建简化的回退仿真器"""
    class SimplifiedTask:
        def __init__(self, task_id, computation_req=1.0, memory_req=10, priority=1):
            self.task_id = task_id
            self.computation_requirement = computation_req
            self.memory_requirement = memory_req
            self.priority = priority
            self.dependencies = []

    class SimplifiedEnvironment:
        def __init__(self):
            self.current_time = 0
            self.nodes = self._create_nodes()

        def _create_nodes(self):
            from collections import defaultdict
            nodes = defaultdict(list)

            node_configs = {
                'FPGA': {'count': 4, 'memory': 100, 'base_time': 0.8, 'base_energy': 12},
                'FOG_GPU': {'count': 3, 'memory': 150, 'base_time': 0.5, 'base_energy': 25},
                'CLOUD': {'count': 1, 'memory': 200, 'base_time': 0.3, 'base_energy': 45}
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
                        'efficiency_score': np.random.uniform(0.8, 1.0),
                        'can_accommodate': lambda self, req: self.memory_capacity - self.current_load >= req,
                        'get_execution_time': lambda self, comp_req, cfg=config: cfg['base_time'] * comp_req / self.efficiency_score,
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
            return np.random.random(15).astype(np.float32)

        def get_cluster_state(self, cluster_type):
            dims = {'FPGA': 6, 'FOG_GPU': 8, 'CLOUD': 6}
            return np.random.random(dims.get(cluster_type, 6)).astype(np.float32)

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

    class SimplifiedSimulator:
        def __init__(self, config):
            self.config = config
            self.environment = SimplifiedEnvironment()
            self.hd_ddpg = None
            self.training_history = {
                'episode_rewards': [],
                'makespans': [],
                'load_balances': [],
                'makespan_stability': []
            }

        def _generate_episode_workflows(self, difficulty_info=None):
            workflows = []

            # 🚀 支持课程学习的工作流生成
            if difficulty_info:
                num_workflows = difficulty_info['workflows_per_episode']
                min_size = difficulty_info['min_tasks']
                max_size = difficulty_info['max_tasks']
            else:
                num_workflows = self.config.get('workflows_per_episode', 8)
                min_size = self.config.get('min_workflow_size', 5)
                max_size = self.config.get('max_workflow_size', 15)

            for w_id in range(num_workflows):
                workflow_size = np.random.randint(min_size, max_size + 1)
                workflow = []

                for t_id in range(workflow_size):
                    # 🚀 根据难度调整任务复杂度
                    if difficulty_info and difficulty_info['complexity_level'] == 'expert':
                        comp_req = np.random.uniform(1.5, 3.0)
                        memory_req = np.random.randint(15, 40)
                    elif difficulty_info and difficulty_info['complexity_level'] == 'hard':
                        comp_req = np.random.uniform(1.0, 2.5)
                        memory_req = np.random.randint(10, 30)
                    else:
                        comp_req = np.random.uniform(0.5, 2.0)
                        memory_req = np.random.randint(5, 25)

                    task = SimplifiedTask(
                        task_id=f"w{w_id}_t{t_id}",
                        computation_req=comp_req,
                        memory_req=memory_req,
                        priority=np.random.randint(1, 4)
                    )
                    workflow.append(task)

                workflows.append(workflow)

            return workflows

        def get_training_summary(self):
            """获取训练摘要"""
            if hasattr(self.hd_ddpg, 'get_training_summary'):
                return self.hd_ddpg.get_training_summary()
            else:
                return {
                    'training_history': self.training_history,
                    'total_episodes': len(self.training_history['episode_rewards']),
                    'trend_direction': 'unknown'
                }

    return SimplifiedSimulator


def print_optimized_training_summary(final_summary):
    """打印优化的训练总结"""
    print("\n" + "="*60)
    print("ENHANCED HD-DDPG TRAINING SUMMARY")
    print("="*60)

    # 实验信息
    exp_info = final_summary['experiment_info']
    print(f"\nExperiment ID: {exp_info['experiment_id']}")
    print(f"Total Episodes: {exp_info['total_episodes']}")
    print(f"Training Time: {exp_info['total_training_time']:.2f}s")
    print(f"Avg Episode Time: {exp_info['average_episode_time']:.2f}s")

    # 性能指标
    perf = final_summary['performance_metrics']
    print(f"\nPerformance Metrics:")
    print(f"  Final Avg Reward: {perf['final_avg_reward']:.2f}")
    print(f"  Best Reward: {perf['best_reward']:.2f}")
    makespan_str = f"{perf['final_avg_makespan']:.2f}" if perf['final_avg_makespan'] != float('inf') else "INF"
    print(f"  Final Avg Makespan: {makespan_str}")
    print(f"  Success Rate: {perf['success_rate']:.1%}")
    print(f"  Success Rate Improvement: {perf.get('success_rate_improvement', 0):.1%}")

    # 配置信息
    config = final_summary['training_config']
    print(f"\nConfiguration:")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['meta_lr']:.6f}")
    print(f"  Makespan Target: {config['makespan_target']}")
    print(f"  Makespan Weight: {config['makespan_weight']}")
    print(f"  Buffer Size: {config['memory_capacity']}")

    print("\n" + "="*60)


def main():
    """🚀 主训练函数 - 高性能增强版本"""
    print("Enhanced HD-DDPG Medical Workflow Scheduling Training")
    print("🚀 High-Performance Version with Advanced Features")
    print("=" * 80)

    # 设置优化环境
    setup_optimized_environment()

    # 解析参数
    global args
    args = parse_optimized_arguments()

    # 设置实验
    experiment_dir = setup_optimized_experiment(args)

    # 创建优化日志记录器
    logger = create_optimized_logger(experiment_dir, args.verbose)

    # 🚀 初始化高级训练组件
    curriculum_manager = None
    lr_scheduler = None
    exploration_strategy = None
    fcfs_monitor = None

    if args.enable_curriculum:
        curriculum_manager = CurriculumLearningManager(
            initial_workflows=4, max_workflows=args.workflows_per_episode,
            initial_tasks=args.min_workflow_size, max_tasks=args.max_workflow_size
        )
        print(f"INFO: Curriculum learning enabled")

    if args.enable_lr_schedule:
        lr_scheduler = AdaptiveLearningRateScheduler(
            initial_lr=args.learning_rate,
            min_lr=1e-6,
            decay_factor=0.995,
            patience=25
        )
        print(f"INFO: Adaptive learning rate scheduling enabled")

    if args.enable_enhanced_exploration:
        exploration_strategy = EnhancedExplorationStrategy(
            initial_noise=0.25,
            final_noise=0.03,
            decay_episodes=args.noise_decay_episodes
        )
        print(f"INFO: Enhanced exploration strategy enabled")

    if not args.disable_fcfs_comparison:
        fcfs_monitor = FCFSTrainingMonitor(comparison_frequency=args.fcfs_comparison_frequency)
        print(f"INFO: FCFS training monitor initialized (comparison every {args.fcfs_comparison_frequency} episodes)")
    else:
        print("INFO: FCFS comparison disabled")

    # 🎯 创建高性能配置
    hd_ddpg_config = create_optimized_config(args)
    hd_ddpg_config['comm_weight'] = 0.5
    simulator_config = create_optimized_simulator_config(args)

    # 🎯 验证配置传递
    print(f"🔧 Enhanced Training Configuration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Workflows per episode: {args.workflows_per_episode}")
    print(f"  Batch size: {hd_ddpg_config['batch_size']}")
    print(f"  Buffer size: {hd_ddpg_config['memory_capacity']}")
    print(f"  Learning rate: {hd_ddpg_config['meta_lr']:.6f}")
    print(f"  🎯 Makespan target: {hd_ddpg_config['makespan_target']}")
    print(f"  🎯 Makespan weight: {hd_ddpg_config['makespan_weight']}")
    print(f"  🚀 Curriculum learning: {args.enable_curriculum}")
    print(f"  🚀 LR scheduling: {args.enable_lr_schedule}")
    print(f"  🚀 Enhanced exploration: {args.enable_enhanced_exploration}")
    print(f"  Experiment name: {args.experiment_name}")
    print("-" * 80)

    # 创建仿真器
    try:
        if MODULES_AVAILABLE:
            simulator = StabilizedMedicalSchedulingSimulator(simulator_config)
            print("INFO: Using optimized simulator")
        else:
            FallbackSimulator = create_fallback_simulator(simulator_config)
            simulator = FallbackSimulator(simulator_config)
            print("WARNING: Using fallback simulator")
    except Exception as e:
        logger.log_error(0, f"Simulator creation failed: {e}")
        FallbackSimulator = create_fallback_simulator(simulator_config)
        simulator = FallbackSimulator(simulator_config)

    # 🎯 创建HD-DDPG实例
    try:
        print("INFO: Creating Enhanced HD-DDPG...")
        simulator.hd_ddpg = HDDDPG(hd_ddpg_config)
        print("INFO: ✅ Enhanced HD-DDPG created successfully")

        # 🎯 验证配置传递成功
        actual_target = simulator.hd_ddpg.config.get('makespan_target', 'NOT_SET')
        actual_weight = simulator.hd_ddpg.config.get('makespan_weight', 'NOT_SET')
        print(f"INFO: 🎯 HD-DDPG actual makespan target: {actual_target}")
        print(f"INFO: 🎯 HD-DDPG actual makespan weight: {actual_weight}")

    except Exception as e:
        logger.log_error(0, f"HD-DDPG creation failed: {e}")
        print("ERROR: Cannot create HD-DDPG instance, exiting")
        return

    # 加载预训练模型
    if args.load_model:
        try:
            simulator.hd_ddpg.load_models(args.load_model)
            print(f"INFO: Pre-trained model loaded: {args.load_model}")
        except Exception as e:
            logger.log_error(0, f"Cannot load model {args.load_model}: {e}")

    # 开始训练
    training_start_time = time.time()
    episode_rewards = []
    episode_results = []
    performance_history = []  # 🚀 性能历史用于LR调度

    print(f"\nINFO: 🚀 Starting Enhanced HD-DDPG training - {args.episodes} episodes")
    print(f"INFO: 🎯 Target makespan: {hd_ddpg_config['makespan_target']}")
    print(f"INFO: 📊 Expected improvement target: 50-80% success rate")

    # 🚀 创建增强的主进度条
    use_tqdm = TQDM_AVAILABLE and not args.disable_progress_bar

    if use_tqdm:
        main_pbar = trange(
            1, args.episodes + 1,
            desc="Enhanced HD-DDPG Training",
            unit="ep",
            ncols=130,
            colour='blue'
        )
    else:
        main_pbar = range(1, args.episodes + 1)

    # 收敛标志
    convergence_achieved = False

    try:
        for episode in main_pbar:
            episode_start_time = time.time()

            try:
                # 🚀 课程学习：动态调整难度
                difficulty_info = None
                if curriculum_manager:
                    difficulty_info = curriculum_manager.get_current_difficulty(episode)
                    if episode % 50 == 0:  # 每50轮记录一次课程更新
                        logger.log_curriculum_update(episode, difficulty_info)
                        print(f"📚 Curriculum Update - Phase: {difficulty_info['phase']}, "
                              f"Workflows: {difficulty_info['workflows_per_episode']}, "
                              f"Tasks: {difficulty_info['min_tasks']}-{difficulty_info['max_tasks']}")

                # 生成工作流（兼容现有接口）
                if hasattr(simulator, '_generate_episode_workflows'):
                    # 检查函数签名以确保兼容性
                    import inspect
                    try:
                        sig = inspect.signature(simulator._generate_episode_workflows)
                        if len(sig.parameters) > 1:  # 支持额外参数
                            workflows = simulator._generate_episode_workflows(difficulty_info)
                        else:  # 不支持额外参数，使用默认方式
                            workflows = simulator._generate_episode_workflows()
                            # 手动调整工作流数量以支持课程学习
                            if difficulty_info and len(workflows) > difficulty_info['workflows_per_episode']:
                                workflows = workflows[:difficulty_info['workflows_per_episode']]
                    except Exception as e:
                        # 如果检查失败，使用默认方式
                        workflows = simulator._generate_episode_workflows()
                        if difficulty_info and len(workflows) > difficulty_info['workflows_per_episode']:
                            workflows = workflows[:difficulty_info['workflows_per_episode']]
                else:
                    # 回退到简化版本
                    workflows = []
                    num_workflows = difficulty_info['workflows_per_episode'] if difficulty_info else 8
                    for i in range(num_workflows):
                        # 创建简单的模拟工作流
                        workflow = []
                        for j in range(np.random.randint(5, 12)):
                            task = type('SimpleTask', (), {
                                'computation_requirement': np.random.uniform(0.5, 2.0),
                                'memory_requirement': np.random.randint(5, 25),
                                'priority': 1
                            })()
                            workflow.append(task)
                        workflows.append(workflow)

                # 更新全局损失追踪器的当前episode
                GLOBAL_LOSS_TRACKER.current_episode = episode

                # 🚀 增强探索策略
                if exploration_strategy and hasattr(simulator.hd_ddpg, 'set_exploration_noise'):
                    # 分析最近性能趋势
                    if len(performance_history) > 10:
                        recent_trend = np.polyfit(range(10), performance_history[-10:], 1)[0]
                        if recent_trend > 0.1:
                            trend = 'improving'
                        elif recent_trend < -0.1:
                            trend = 'declining'
                        else:
                            trend = 'stable'
                    else:
                        trend = 'stable'

                    current_noise = exploration_strategy.get_exploration_noise(episode, trend)
                    simulator.hd_ddpg.set_exploration_noise(current_noise)

                # 🎯 训练一个episode
                try:
                    result = simulator.hd_ddpg.train_episode(workflows, simulator.environment)
                except Exception as train_error:
                    # 🚀 修复：如果训练失败，创建默认结果
                    logger.log_error(episode, f"Episode training error: {train_error}")
                    result = {
                        'total_reward': -15.0,
                        'avg_makespan': float('inf'),
                        'success_rate': 0.0,
                        'makespan_stability': 0.0,
                        'avg_load_balance': 0.0,
                        'current_lr': hd_ddpg_config.get('meta_lr', 0.0003),
                        'makespan_target': hd_ddpg_config.get('makespan_target', 2.8),
                        'target_achievement_rate': 0.0
                    }

                episode_duration = time.time() - episode_start_time

                # 🚀 学习率调度
                if lr_scheduler:
                    current_performance = result.get('avg_makespan', float('inf'))
                    if current_performance != float('inf'):
                        performance_history.append(current_performance)
                        old_lr = lr_scheduler.get_current_lr()
                        lr_updated = lr_scheduler.update(current_performance, episode)

                        if lr_updated:
                            new_lr = lr_scheduler.get_current_lr()
                            # 更新HD-DDPG的学习率
                            if hasattr(simulator.hd_ddpg, 'update_learning_rate'):
                                simulator.hd_ddpg.update_learning_rate(new_lr)

                            logger.log_lr_update(episode, old_lr, new_lr)
                            print(f"📉 Learning rate updated: {old_lr:.6f} → {new_lr:.6f}")

                            # 更新结果中的学习率
                            result['current_lr'] = new_lr

                # 🚀 修复：安全地更新训练历史
                try:
                    if hasattr(simulator, 'training_history'):
                        simulator.training_history['episode_rewards'].append(result.get('total_reward', -15.0))
                        simulator.training_history['makespans'].append(result.get('avg_makespan', float('inf')))
                        simulator.training_history['load_balances'].append(result.get('avg_load_balance', 0))
                        simulator.training_history['makespan_stability'].append(result.get('makespan_stability', 0))
                except Exception as history_error:
                    logger.log_error(episode, f"History update error: {history_error}")

                # 🚀 FCFS对比监控
                if fcfs_monitor and fcfs_monitor.should_compare(episode):
                    try:
                        print(f"\n📊 Episode {episode} - Running FCFS comparison...")

                        # 选择第一个工作流进行快速对比
                        if workflows:
                            test_workflow = workflows[0]
                            fcfs_result = fcfs_monitor.quick_fcfs_comparison(test_workflow, simulator.environment)

                            # 分析改进
                            improvement_analysis = fcfs_monitor.analyze_improvement(result, fcfs_result)

                            if improvement_analysis:
                                print(f"📈 HD-DDPG vs FCFS Comparison:")
                                print(f"   HD-DDPG Makespan: {improvement_analysis['hd_ddpg_makespan']:.3f}")
                                print(f"   FCFS Makespan:    {improvement_analysis['fcfs_makespan']:.3f}")
                                print(f"   Improvement:      {improvement_analysis['improvement_percentage']:+.1f}%")
                                print(f"   Comparison Time:  {improvement_analysis['comparison_time']:.3f}s")

                                # 记录到日志
                                logger.log_fcfs_comparison(episode, improvement_analysis)

                                # 根据改进程度给出反馈
                                if improvement_analysis['improvement_percentage'] > 25:
                                    print(f"   🚀 Outstanding performance!")
                                elif improvement_analysis['improvement_percentage'] > 15:
                                    print(f"   ✅ Excellent improvement!")
                                elif improvement_analysis['improvement_percentage'] > 5:
                                    print(f"   👍 Good improvement!")
                                elif improvement_analysis['improvement_percentage'] > 0:
                                    print(f"   📈 Modest improvement")
                                else:
                                    print(f"   ⚠️  Need more training")

                    except Exception as e:
                        print(f"WARNING: FCFS comparison failed: {e}")

                # 🚀 损失记录
                if episode % args.loss_logging_frequency == 0 and args.enable_detailed_loss_tracking:
                    try:
                        # 从HD-DDPG获取最新损失
                        if hasattr(simulator.hd_ddpg, 'meta_controller'):
                            meta_controller = simulator.hd_ddpg.meta_controller
                            if hasattr(meta_controller, 'last_critic_loss') and hasattr(meta_controller,
                                                                                        'last_actor_loss'):
                                GLOBAL_LOSS_TRACKER.log_meta_losses(
                                    meta_controller.last_critic_loss,
                                    meta_controller.last_actor_loss,
                                    episode
                                )

                        # 记录子控制器损失
                        if hasattr(simulator.hd_ddpg, 'sub_controller_manager'):
                            for cluster_name in ['FPGA', 'FOG_GPU', 'CLOUD']:
                                controller = simulator.hd_ddpg.sub_controller_manager.get_sub_controller(cluster_name)
                                if controller and hasattr(controller, 'last_critic_loss') and hasattr(controller,
                                                                                                      'last_actor_loss'):
                                    GLOBAL_LOSS_TRACKER.log_sub_losses(
                                        cluster_name,
                                        controller.last_critic_loss,
                                        controller.last_actor_loss
                                    )
                    except Exception as e:
                        if args.verbose:
                            print(f"WARNING: Loss logging error in episode {episode}: {e}")

                # 🚀 收敛检测
                if hasattr(result, 'should_stop') and result.get('should_stop', False):
                    logger.log_convergence(episode, "Early convergence detected")
                    convergence_achieved = True
                    if use_tqdm:
                        main_pbar.close()
                    break

                # 🎯 增强的进度条更新
                if use_tqdm:
                    makespan_display = f"{result.get('avg_makespan', float('inf')):.2f}" if result.get('avg_makespan',
                                                                                                       float(
                                                                                                           'inf')) != float(
                        'inf') else "INF"
                    success_rate = result.get('success_rate', 0)
                    target_achievement = result.get('target_achievement_rate', 0)
                    current_lr = result.get('current_lr', args.learning_rate)

                    # 🚀 显示更多信息
                    postfix = {
                        'R': f"{result.get('total_reward', 0):.1f}",
                        'MS': makespan_display,
                        'SR': f"{success_rate:.1%}",
                        'LR': f"{current_lr:.5f}",
                        'Phase': difficulty_info['phase'] if difficulty_info else 0
                    }

                    main_pbar.set_postfix(postfix)

                # 记录结果
                episode_rewards.append(result.get('total_reward', -15.0))
                episode_results.append(result)

                # 记录进度
                console_msg = logger.log_episode(episode, result, episode_duration)

                # 🎯 显示阶段性改进信息
                if episode % 50 == 0 and episode > 0:
                    recent_success_rates = [r.get('success_rate', 0) for r in episode_results[-20:]]
                    recent_makespans = [r.get('avg_makespan', float('inf')) for r in episode_results[-20:]
                                        if r.get('avg_makespan', float('inf')) != float('inf')]

                    if recent_success_rates and recent_makespans:
                        avg_recent_success = np.mean(recent_success_rates)
                        avg_recent_makespan = np.mean(recent_makespans)
                        target = hd_ddpg_config['makespan_target']

                        print(f"\n🎯 Episode {episode} Progress Report:")
                        print(f"   Recent Success Rate: {avg_recent_success:.1%}")
                        print(f"   Recent Avg Makespan: {avg_recent_makespan:.3f} (Target: {target})")
                        print(f"   Gap to Target: {avg_recent_makespan - target:.3f}")

                        # 🚀 性能趋势分析
                        if len(recent_makespans) > 10:
                            trend = np.polyfit(range(len(recent_makespans)), recent_makespans, 1)[0]
                            if trend < -0.05:
                                print(f"   📈 Trend: Improving (slope: {trend:.4f})")
                            elif trend > 0.05:
                                print(f"   📉 Trend: Declining (slope: {trend:.4f})")
                            else:
                                print(f"   📊 Trend: Stable (slope: {trend:.4f})")

                # 🚀 评估频率
                if episode % args.eval_interval == 0:
                    eval_result = evaluate_optimized_model(simulator, experiment_dir, episode, logger)

                # 🚀 保存频率
                if episode % args.save_interval == 0:
                    try:
                        # 保存模型
                        model_path = os.path.join(experiment_dir, 'checkpoints', f'enhanced_hd_ddpg_ep{episode}')
                        simulator.hd_ddpg.save_models(model_path)

                        # 🎯 保存训练指标
                        save_optimized_training_metrics(simulator, experiment_dir, episode, logger, hd_ddpg_config)

                        print(f"INFO: Episode {episode} checkpoint saved")
                    except Exception as e:
                        logger.log_error(episode, f"Checkpoint save error: {e}")

            except Exception as e:
                logger.log_error(episode, f"Episode error: {e}")
                # 记录错误的episode结果
                episode_rewards.append(-15.0)
                episode_results.append({
                    'total_reward': -15.0,
                    'avg_makespan': float('inf'),
                    'success_rate': 0.0,
                    'makespan_stability': 0.0,
                    'target_achievement_rate': 0.0
                })

    except KeyboardInterrupt:
        print("\nWARNING: User interrupted training")
        if use_tqdm:
            main_pbar.close()

    except Exception as e:
        logger.log_error(0, f"Training error: {e}")
        if use_tqdm:
            main_pbar.close()

    finally:
        # 训练完成处理
        if use_tqdm and 'main_pbar' in locals():
            main_pbar.close()

        training_duration = time.time() - training_start_time

        print(f"\n🎉 Enhanced Training completed!")
        print(f"Total training time: {training_duration:.2f} seconds")
        print(f"Average episode time: {training_duration / max(len(episode_results), 1):.2f} seconds")

        # 🎯 显示训练效果总结
        if episode_results:
            final_success_rates = [r.get('success_rate', 0) for r in episode_results[-30:]]
            final_makespans = [r.get('avg_makespan', float('inf')) for r in episode_results[-30:]
                               if r.get('avg_makespan', float('inf')) != float('inf')]

            if final_success_rates and final_makespans:
                final_avg_success = np.mean(final_success_rates)
                final_avg_makespan = np.mean(final_makespans)
                target = hd_ddpg_config['makespan_target']

                print(f"\n🎯 Final Training Results:")
                print(f"   Final Success Rate: {final_avg_success:.1%}")
                print(f"   Final Avg Makespan: {final_avg_makespan:.3f}")
                print(f"   Target Achievement: {'✅ ACHIEVED' if final_avg_makespan <= target else '❌ NOT YET'}")
                print(f"   Improvement vs Target: {((target - final_avg_makespan) / target * 100):+.1f}%")

        # 🚀 FCFS对比总结
        if fcfs_monitor:
            print("\n" + "=" * 60)
            print("ENHANCED HD-DDPG vs FCFS TRAINING SUMMARY")
            print("=" * 60)

            fcfs_summary = fcfs_monitor.get_training_summary()
            if isinstance(fcfs_summary, dict):
                avg_improvement = fcfs_summary.get('average_improvement', 0)
                consistent_rate = fcfs_summary.get('consistent_improvement', 0)

                print(f"📈 Average Improvement vs FCFS: {avg_improvement:+.1f}%")
                print(f"📊 Consistent Improvement Rate: {consistent_rate:.1%}")
                print(f"🔢 Total Comparisons: {fcfs_summary.get('total_comparisons', 0)}")

                if avg_improvement > 25:
                    print("🚀 Outstanding performance vs FCFS!")
                elif avg_improvement > 15:
                    print("✅ Excellent performance vs FCFS!")
                elif avg_improvement > 8:
                    print("👍 Good performance vs FCFS")
                elif avg_improvement > 0:
                    print("⚠️  Modest improvement vs FCFS")
                else:
                    print("❌ Performance needs improvement vs FCFS")
            else:
                print(fcfs_summary)

        # 保存最终模型
        final_model_path = os.path.join(experiment_dir, 'final_enhanced_model')
        try:
            simulator.hd_ddpg.save_models(final_model_path)
            print(f"INFO: Final enhanced model saved: {final_model_path}")

        except Exception as e:
            logger.log_error(0, f"Cannot save final model: {e}")

        # 🎯 保存最终指标
        try:
            save_optimized_training_metrics(simulator, experiment_dir, len(episode_results), logger, hd_ddpg_config)
            print(f"INFO: Training metrics saved: {experiment_dir}/enhanced_metrics_ep{len(episode_results)}.csv")
        except Exception as e:
            logger.log_error(0, f"Cannot save training metrics: {e}")

        # 🚀 生成增强的最终报告
        try:
            # 获取训练总结
            if hasattr(simulator, 'hd_ddpg') and hasattr(simulator.hd_ddpg, 'get_training_summary'):
                training_summary = simulator.hd_ddpg.get_training_summary()
            else:
                training_summary = {'training_history': getattr(simulator, 'training_history', {})}

            # 计算最终统计
            valid_rewards = [r for r in episode_rewards if not np.isnan(r) and not np.isinf(r)]
            valid_results = [r for r in episode_results if r.get('avg_makespan', float('inf')) != float('inf')]

            # 🎯 计算成功率改进
            initial_success_rates = [r.get('success_rate', 0) for r in episode_results[:30]] if len(
                episode_results) >= 30 else []
            final_success_rates = [r.get('success_rate', 0) for r in episode_results[-30:]] if len(
                episode_results) >= 30 else []

            success_rate_improvement = 0
            if initial_success_rates and final_success_rates:
                initial_avg = np.mean(initial_success_rates)
                final_avg = np.mean(final_success_rates)
                success_rate_improvement = final_avg - initial_avg

            final_summary = {
                'experiment_info': {
                    'experiment_id': args.experiment_name,
                    'total_episodes': len(episode_results),
                    'total_training_time': training_duration,
                    'average_episode_time': training_duration / max(len(episode_results), 1),
                    'training_mode': 'HIGH_PERFORMANCE_ENHANCED_WITH_ADVANCED_FEATURES',
                    'tensorflow_version': tf.__version__,
                    'fcfs_comparison_enabled': not args.disable_fcfs_comparison,
                    'curriculum_learning_enabled': args.enable_curriculum,
                    'lr_scheduling_enabled': args.enable_lr_schedule,
                    'enhanced_exploration_enabled': args.enable_enhanced_exploration
                },
                'performance_metrics': {
                    'final_avg_reward': float(np.mean(valid_rewards[-40:])) if len(valid_rewards) >= 40 else float(
                        np.mean(valid_rewards)) if valid_rewards else 0,
                    'best_reward': float(np.max(valid_rewards)) if valid_rewards else 0,
                    'final_avg_makespan': float(
                        np.mean([r.get('avg_makespan', float('inf')) for r in valid_results[-30:]])) if len(
                        valid_results) >= 30 else float(np.mean(
                        [r.get('avg_makespan', float('inf')) for r in valid_results])) if valid_results else float(
                        'inf'),
                    'success_rate': len(valid_results) / len(episode_results) if episode_results else 0,
                    'success_rate_improvement': float(success_rate_improvement),
                    'makespan_target_achievement': float(final_avg_makespan <= hd_ddpg_config[
                        'makespan_target']) if 'final_avg_makespan' in locals() else 0
                },
                'training_config': hd_ddpg_config,
                'episode_rewards': [float(x) for x in episode_rewards],
                'completion_timestamp': datetime.now().isoformat()
            }

            # 🚀 FCFS对比摘要
            if fcfs_monitor:
                fcfs_summary = fcfs_monitor.get_training_summary()
                if isinstance(fcfs_summary, dict):
                    final_summary['fcfs_comparison_summary'] = fcfs_summary

            # 🚀 高级特性摘要
            if curriculum_manager:
                final_summary['curriculum_summary'] = {
                    'phases_completed': min(len(episode_results) // 50, 4),
                    'final_difficulty': curriculum_manager.get_current_difficulty(len(episode_results))
                }

            if lr_scheduler:
                internal_updates = 0
                internal_final_lr = None
                try:
                    if hasattr(simulator, 'hd_ddpg') and hasattr(simulator.hd_ddpg, 'lr_scheduler'):
                        internal_updates = getattr(simulator.hd_ddpg.lr_scheduler, 'total_updates', 0)
                        internal_final_lr = getattr(simulator.hd_ddpg.lr_scheduler, 'current_lr', None)
                except Exception:
                    pass

                final_summary['lr_schedule_summary'] = {
                    'initial_lr': args.learning_rate,
                    'final_lr': internal_final_lr if internal_final_lr is not None else lr_scheduler.get_current_lr(),
                    'total_updates': internal_updates if internal_updates else getattr(lr_scheduler, 'update_count', 0)
                }

            # 转换为可序列化格式
            final_summary = convert_to_serializable(final_summary)

            # 保存最终总结
            summary_path = os.path.join(experiment_dir, 'final_enhanced_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(final_summary, f, indent=2, default=str)

            # 打印优化的总结
            print_optimized_training_summary(final_summary)

            print(f"\nINFO: Final results saved to: {experiment_dir}")

        except Exception as e:
            logger.log_error(0, f"Cannot save training summary: {e}")

    print("\n" + "=" * 80)
    print("🎉 Enhanced HD-DDPG Training Program Completed Successfully!")
    print("=" * 80)

    


if __name__ == "__main__":
    main()
