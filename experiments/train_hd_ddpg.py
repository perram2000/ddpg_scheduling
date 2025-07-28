"""
HD-DDPG Training Script - 超优化版本 with Training Losses Tracking
HD-DDPG训练主脚本 - 解决图表布局和损失跟踪问题

主要优化：
- 添加完整的Training Losses追踪和可视化
- 修复图表布局 (删除buffer quality，优化为2x3布局)
- 修复子控制器损失数据记录问题
- 进一步优化Makespan稳定性
- 完全学术化输出格式 (移除emoji)
- 表格化结果展示
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
    print("INFO: All stabilized modules imported successfully")
except ImportError as e:
    print(f"WARNING: Stabilized modules import failed: {e}")
    print("WARNING: Using simplified fallback version")
    MODULES_AVAILABLE = False

# ===============================
# 新增：全局损失追踪变量
# ===============================
class GlobalLossTracker:
    """全局损失追踪器"""
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
                        # 填充缺失值
                        filled_values = values + [np.nan] * (max_length - len(values))
                        loss_data[key] = filled_values[:max_length]

                loss_df = pd.DataFrame(loss_data)
                csv_path = os.path.join(save_path, "training_losses_detailed.csv")
                loss_df.to_csv(csv_path, index=False)
                return csv_path
        except ImportError:
            print("WARNING: pandas not available, skipping loss CSV save")
        except Exception as e:
            print(f"WARNING: Loss data save error: {e}")
        return None

# 创建全局损失追踪器实例
GLOBAL_LOSS_TRACKER = GlobalLossTracker()


def setup_advanced_environment():
    """设置高级训练环境"""
    print("INFO: Configuring advanced training environment...")

    # 禁用GPU，强制使用CPU
    tf.config.set_visible_devices([], 'GPU')

    # CPU优化设置
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
    print(f"INFO: Available CPU threads: {tf.config.threading.get_inter_op_parallelism_threads()}")

    return True


def convert_to_serializable(obj):
    """增强的类型转换函数"""
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


def parse_enhanced_arguments():
    """解析增强的命令行参数"""
    parser = argparse.ArgumentParser(
        description='HD-DDPG Medical Workflow Scheduling Training (Ultra-Optimized Version with Loss Tracking)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 核心训练参数
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    training_group.add_argument('--workflows-per-episode', type=int, default=8,
                        help='Number of workflows per episode')
    training_group.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    training_group.add_argument('--learning-rate', type=float, default=0.00005,
                        help='Learning rate')
    training_group.add_argument('--memory-capacity', type=int, default=10000,
                        help='Replay buffer capacity')

    # 环境参数
    env_group = parser.add_argument_group('Environment Parameters')
    env_group.add_argument('--min-workflow-size', type=int, default=6,
                        help='Minimum workflow size')
    env_group.add_argument('--max-workflow-size', type=int, default=12,
                        help='Maximum workflow size')
    env_group.add_argument('--failure-rate', type=float, default=0.008,
                        help='System failure rate')
    env_group.add_argument('--makespan-target', type=float, default=1.3,
                        help='Target makespan')

    # 稳定化参数
    stable_group = parser.add_argument_group('Stabilization Parameters')
    stable_group.add_argument('--enable-per', action='store_true', default=True,
                        help='Enable prioritized experience replay')
    stable_group.add_argument('--quality-threshold', type=float, default=0.08,
                        help='Experience quality threshold')
    stable_group.add_argument('--action-smoothing', action='store_true', default=True,
                        help='Enable action smoothing')
    stable_group.add_argument('--state-smoothing', action='store_true', default=True,
                        help='Enable state smoothing')
    stable_group.add_argument('--disable-per', action='store_true',
                        help='Disable prioritized experience replay')

    # 保存和加载
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument('--save-interval', type=int, default=50,
                        help='Model save interval')
    io_group.add_argument('--eval-interval', type=int, default=20,
                        help='Evaluation interval')
    io_group.add_argument('--load-model', type=str, default=None,
                        help='Path to load pre-trained model')
    io_group.add_argument('--save-dir', type=str, default='stabilized_model_result',
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
    viz_group.add_argument('--update-interval', type=int, default=1,
                        help='Progress bar update interval')
    viz_group.add_argument('--save-plots', action='store_true', default=True,
                        help='Save training plots')

    # 新增：损失追踪参数
    loss_group = parser.add_argument_group('Loss Tracking Parameters')
    loss_group.add_argument('--loss-logging-frequency', type=int, default=1,
                        help='Frequency of loss logging (every N episodes)')
    loss_group.add_argument('--enable-detailed-loss-tracking', action='store_true', default=True,
                        help='Enable detailed loss tracking for all controllers')
    loss_group.add_argument('--save-loss-plots', action='store_true', default=True,
                        help='Save detailed loss plots')

    # CPU优化参数
    cpu_group = parser.add_argument_group('CPU Optimization')
    cpu_group.add_argument('--cpu-threads', type=int, default=0,
                        help='Number of CPU threads (0=auto)')
    cpu_group.add_argument('--memory-limit', type=int, default=4096,
                        help='Memory limit in MB')

    return parser.parse_args()


def setup_enhanced_experiment(args):
    """设置增强的实验环境"""
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
        args.experiment_name = f"ultra_optimized_hd_ddpg_{timestamp}"

    experiment_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 重置全局损失追踪器
    GLOBAL_LOSS_TRACKER.reset()

    # 保存实验配置
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        config_dict = vars(args).copy()
        config_dict['training_mode'] = 'ULTRA_OPTIMIZED_CPU'
        config_dict['tensorflow_version'] = tf.__version__
        config_dict['tqdm_available'] = TQDM_AVAILABLE
        config_dict['modules_available'] = MODULES_AVAILABLE
        config_dict['loss_tracking_enabled'] = True
        json.dump(config_dict, f, indent=2)

    print(f"INFO: Experiment setup completed: {experiment_dir}")
    return experiment_dir


def create_ultra_optimized_config(args):
    """创建超优化HD-DDPG配置 - 解决Makespan反弹"""
    config = {
        # 基础参数 - 进一步优化
        'meta_state_dim': 15,
        'meta_action_dim': 3,
        'gamma': 0.985,  # 进一步提升长期稳定性
        'batch_size': args.batch_size,
        'memory_capacity': args.memory_capacity,
        'meta_lr': args.learning_rate * 0.8,  # 进一步降低学习率
        'sub_lr': args.learning_rate * 0.8,
        'tau': 0.001,  # 进一步降低目标网络更新率
        'update_frequency': 1,
        'save_frequency': args.save_interval,

        # 超稳定性参数 - 防止Makespan反弹
        'gradient_clip_norm': 0.2,  # 进一步限制梯度
        'exploration_noise': 0.05,  # 大幅降低初始噪声
        'noise_decay': 0.9995,  # 极慢的噪声衰减
        'min_noise': 0.003,  # 更小的最终噪声
        'reward_scale': 1.0,
        'verbose': args.verbose,

        # Makespan超稳定化参数
        'action_smoothing': args.action_smoothing,
        'action_smoothing_alpha': 0.7,  # 大幅增强动作平滑
        'state_smoothing': args.state_smoothing,
        'state_smoothing_alpha': 0.5,  # 增强状态平滑
        'makespan_weight': 0.95,  # 进一步强化makespan重要性
        'stability_weight': 0.4,   # 大幅提升稳定性权重
        'variance_penalty': 0.25,  # 增强方差惩罚
        'rebound_penalty': 0.3,   # 新增：反弹惩罚
        'target_update_frequency': 4,  # 进一步减缓目标网络更新

        # 收敛性增强参数
        'enable_per': args.enable_per and not args.disable_per,
        'quality_threshold': args.quality_threshold,
        'td_error_clipping': True,
        'td_error_max': 0.5,  # 进一步限制TD误差
        'adaptive_learning': True,
        'learning_rate_decay': True,
        'lr_decay_factor': 0.98,  # 更温和的衰减
        'lr_decay_frequency': 150,  # 更低频率的衰减

        # 超稳定化参数
        'makespan_smoothing': True,
        'makespan_smoothing_window': 10,  # 增大平滑窗口
        'multi_level_smoothing': True,
        'rebound_detection': True,
        'plateau_avoidance': True,  # 新增：平台期避免
        'performance_tracking_depth': 50,

        # 收敛监控参数
        'convergence_patience': 120,  # 增加收敛耐心
        'convergence_threshold': 0.01,  # 更严格的收敛阈值
        'early_stopping': True,
        'plateau_detection': True,
        'plateau_patience': 80,

        # 损失监控参数 - 增强版
        'track_losses': True,
        'loss_logging_frequency': args.loss_logging_frequency,
        'save_loss_plots': args.save_loss_plots,
        'detailed_loss_tracking': args.enable_detailed_loss_tracking,
        'global_loss_tracker': GLOBAL_LOSS_TRACKER,  # 传递全局追踪器
    }

    print(f"INFO: Ultra-optimized HD-DDPG configuration created")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Buffer capacity: {config['memory_capacity']}")
    print(f"  - Learning rate: {config['meta_lr']:.6f}")
    print(f"  - Makespan weight: {config['makespan_weight']}")
    print(f"  - Stability weight: {config['stability_weight']}")
    print(f"  - Action smoothing alpha: {config['action_smoothing_alpha']}")
    print(f"  - Early stopping: {'Enabled' if config['early_stopping'] else 'Disabled'}")
    print(f"  - Loss tracking: {'Enabled' if config['detailed_loss_tracking'] else 'Disabled'}")

    return config


def moving_average(data, window_size):
    """改进的移动平均计算"""
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


def save_optimized_training_plots(history, experiment_dir, episode, logger):
    """
    保存优化的训练图表 - 修复版本 + Training Losses
    主要改进：
    1. 添加Training Losses图表 (类似您提供的图片)
    2. 修复子控制器损失数据问题
    3. 优化为2x3布局，显示6个重要图表
    4. 增强数据验证和错误处理
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        episode_rewards = history.get('episode_rewards', [])
        loss_data = GLOBAL_LOSS_TRACKER.get_loss_data()

        if len(episode_rewards) > 10:
            # 创建优化的2x3图表布局
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            episodes = range(1, len(episode_rewards) + 1)

            # === 第一行：核心性能指标 ===

            # [1] 训练奖励曲线
            axes[0, 0].plot(episodes, episode_rewards, alpha=0.6, color='blue', linewidth=1, label='Episode Rewards')
            if len(episode_rewards) > 20:
                ma_20 = moving_average(episode_rewards, 20)
                axes[0, 0].plot(episodes, ma_20, 'r-', linewidth=2, label='20-episode MA')
            if len(episode_rewards) > 50:
                ma_50 = moving_average(episode_rewards, 50)
                axes[0, 0].plot(episodes, ma_50, 'g-', linewidth=2, label='50-episode MA')
            axes[0, 0].set_title('Training Rewards', fontweight='bold', fontsize=12)
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
                        ma_makespan = moving_average(valid_makespans, min(10, len(valid_makespans)//3))
                        axes[0, 1].plot(valid_episodes[:len(ma_makespan)], ma_makespan,
                                       'r-', linewidth=2, label='Moving Average')

                    # 添加目标线
                    target_makespan = 1.3  # 目标makespan
                    axes[0, 1].axhline(y=target_makespan, color='orange',
                                      linestyle='--', linewidth=2, label='Target')

                    axes[0, 1].set_title('Average Makespan', fontweight='bold', fontsize=12)
                    axes[0, 1].set_xlabel('Episode')
                    axes[0, 1].set_ylabel('Makespan')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                else:
                    axes[0, 1].text(0.5, 0.5, 'Insufficient Valid Makespan Data',
                                   ha='center', va='center', transform=axes[0, 1].transAxes)
                    axes[0, 1].set_title('Average Makespan', fontweight='bold', fontsize=12)

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
                                       'orange', linewidth=2, label='Smoothed')

                    axes[0, 2].set_title('Makespan Stability Score', fontweight='bold', fontsize=12)
                    axes[0, 2].set_xlabel('Episode')
                    axes[0, 2].set_ylabel('Stability Score')
                    axes[0, 2].set_ylim(0, 1)
                    axes[0, 2].legend()
                    axes[0, 2].grid(True, alpha=0.3)
                else:
                    axes[0, 2].text(0.5, 0.5, 'No Valid Stability Data',
                                   ha='center', va='center', transform=axes[0, 2].transAxes)
                    axes[0, 2].set_title('Makespan Stability Score', fontweight='bold', fontsize=12)

            # === 第二行：损失和性能指标 ===

            # [4] Training Losses - 新增重点图表
            meta_critic_losses = loss_data.get('meta_critic', [])
            meta_actor_losses = loss_data.get('meta_actor', [])

            if meta_critic_losses or meta_actor_losses:
                # Meta Critic Loss
                if meta_critic_losses and len(meta_critic_losses) > 5:
                    training_episodes = range(len(meta_critic_losses))
                    axes[1, 0].plot(training_episodes, meta_critic_losses,
                                   'r-', color='red', linewidth=1.5, label='Meta Critic Loss')

                # Meta Actor Loss
                if meta_actor_losses and len(meta_actor_losses) > 5:
                    training_episodes = range(len(meta_actor_losses))
                    axes[1, 0].plot(training_episodes, meta_actor_losses,
                                   color='purple', linewidth=1.5, label='Meta Actor Loss')

                axes[1, 0].set_title('Training Losses', fontweight='bold', fontsize=12)
                axes[1, 0].set_xlabel('Training Steps')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

                # 设置Y轴范围，避免过大的loss值影响可视化
                if meta_critic_losses:
                    y_max = min(np.max(meta_critic_losses), np.percentile(meta_critic_losses, 95) * 2)
                    axes[1, 0].set_ylim(bottom=0, top=y_max)
            else:
                axes[1, 0].text(0.5, 0.5, 'Training Loss data not available',
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Training Losses', fontweight='bold', fontsize=12)

            # [5] 子控制器损失 - 修复数据记录问题
            cluster_colors = {'fpga': 'red', 'fog_gpu': 'green', 'cloud': 'blue'}
            legend_items = []

            # 绘制子控制器损失
            for cluster, color in cluster_colors.items():
                critic_key = f'{cluster}_critic'
                actor_key = f'{cluster}_actor'

                if critic_key in loss_data and loss_data[critic_key]:
                    valid_losses = [l for l in loss_data[critic_key] if not (np.isnan(l) or np.isinf(l))]
                    if valid_losses and len(valid_losses) > 5:
                        loss_episodes = range(len(valid_losses))
                        axes[1, 1].plot(loss_episodes, valid_losses,
                                       alpha=0.7, color=color, linewidth=1,
                                       label=f'{cluster.upper()} Critic')
                        legend_items.append(f'{cluster.upper()} Critic')

                if actor_key in loss_data and loss_data[actor_key]:
                    valid_losses = [l for l in loss_data[actor_key] if not (np.isnan(l) or np.isinf(l))]
                    if valid_losses and len(valid_losses) > 5:
                        loss_episodes = range(len(valid_losses))
                        axes[1, 1].plot(loss_episodes, valid_losses,
                                       alpha=0.5, color=color, linewidth=1, linestyle='--',
                                       label=f'{cluster.upper()} Actor')
                        legend_items.append(f'{cluster.upper()} Actor')

            if legend_items:
                axes[1, 1].set_title('Sub-Controller Losses', fontweight='bold', fontsize=12)
                axes[1, 1].set_xlabel('Training Steps')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].legend(fontsize=8)
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_yscale('log')
            else:
                axes[1, 1].text(0.5, 0.5, 'Sub-Controller Loss Data Not Available',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Sub-Controller Losses', fontweight='bold', fontsize=12)

            # [6] 负载均衡和TD误差
            load_balances = history.get('load_balances', [])
            td_errors = history.get('td_errors', [])

            if load_balances:
                valid_lb = [lb for lb in load_balances if not np.isnan(lb)]
                if valid_lb:
                    ax1 = axes[1, 2]
                    ax1.plot(episodes[:len(valid_lb)], valid_lb,
                            alpha=0.7, color='orange', linewidth=1, label='Load Balance')
                    if len(valid_lb) > 10:
                        ma_lb = moving_average(valid_lb, 10)
                        ax1.plot(episodes[:len(ma_lb)], ma_lb,
                                'darkorange', linewidth=2, label='LB MA')
                    ax1.set_xlabel('Episode')
                    ax1.set_ylabel('Load Balance', color='orange')
                    ax1.tick_params(axis='y', labelcolor='orange')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend(loc='upper left')

            if td_errors:
                valid_td = [td for td in td_errors if not (np.isnan(td) or np.isinf(td))]
                if valid_td:
                    if load_balances and valid_lb:
                        ax2 = ax1.twinx()
                    else:
                        ax2 = axes[1, 2]

                    ax2.plot(episodes[:len(valid_td)], valid_td,
                            alpha=0.6, color='red', linewidth=1, label='TD Error')
                    if len(valid_td) > 10:
                        ma_td = moving_average(valid_td, 10)
                        ax2.plot(episodes[:len(ma_td)], ma_td,
                                'darkred', linewidth=2, label='TD MA')
                    ax2.set_xlabel('Episode')
                    ax2.set_ylabel('TD Error', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    ax2.legend(loc='upper right')

            axes[1, 2].set_title('Load Balance & TD Error', fontweight='bold', fontsize=12)

            # 调整布局
            plt.tight_layout(pad=3.0)

            # 保存图表
            plot_dir = os.path.join(experiment_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f'optimized_training_analysis_ep{episode}.png')
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close()

            if logger.verbose:
                print(f"INFO: Optimized training analysis plot saved: {plot_path}")

    except ImportError:
        print("WARNING: matplotlib not installed, skipping plot generation")
    except Exception as e:
        print(f"WARNING: Plot generation error: {e}")
        import traceback
        traceback.print_exc()


def create_stabilized_simulator_config(args):
    """创建稳定化仿真器配置"""
    return {
        'simulation_episodes': args.episodes,
        'workflows_per_episode': args.workflows_per_episode,
        'min_workflow_size': args.min_workflow_size,
        'max_workflow_size': args.max_workflow_size,
        'workflow_types': ['radiology', 'pathology', 'general'],
        'failure_rate': args.failure_rate,
        'save_interval': args.save_interval,
        'evaluation_interval': args.eval_interval,

        # 稳定化配置
        'stability_monitoring': True,
        'state_smoothing': args.state_smoothing,
        'reward_normalization': True,
        'performance_tracking_window': 50,
        'convergence_threshold': 0.05,
        'makespan_target': args.makespan_target,
    }


def create_enhanced_logger(experiment_dir, verbose=False):
    """创建增强的日志记录器 - 学术化版本"""
    class AcademicLogger:
        def __init__(self, experiment_dir, verbose):
            self.experiment_dir = experiment_dir
            self.verbose = verbose
            self.log_file = os.path.join(experiment_dir, 'logs', 'training.log')
            self.metrics_file = os.path.join(experiment_dir, 'logs', 'metrics.jsonl')

            # 创建日志文件
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            self._write_header()

        def _write_header(self):
            """写入学术化日志头部"""
            header = f"""
{'='*80}
Ultra-Optimized HD-DDPG Medical Workflow Scheduling Training Log
Training Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Experiment Directory: {self.experiment_dir}
Training Mode: ULTRA_OPTIMIZED_CPU
Loss Tracking: ENABLED
{'='*80}
"""
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(header)

        def log_episode(self, episode, result, duration):
            """记录episode信息 - 学术化格式"""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 安全获取结果值
            total_reward = result.get('total_reward', 0)
            avg_makespan = result.get('avg_makespan', float('inf'))
            avg_load_balance = result.get('avg_load_balance', 0)
            success_rate = result.get('success_rate', 0)
            stability_score = result.get('makespan_stability', 0)
            quality_score = result.get('quality_score', 0)
            current_lr = result.get('current_lr', 0)

            # 格式化makespan显示
            makespan_str = f"{avg_makespan:6.2f}" if avg_makespan != float('inf') else "   INF"

            # 学术化控制台输出格式
            console_msg = (
                f"Episode {episode:4d} | "
                f"Reward: {total_reward:8.2f} | "
                f"Makespan: {makespan_str} | "
                f"Success: {success_rate:.1%} | "
                f"Stability: {stability_score:.3f} | "
                f"LR: {current_lr:.6f}"
            )

            # 详细学术化日志格式
            detailed_log = (
                f"[{timestamp}] Episode {episode:4d} | "
                f"Reward: {total_reward:8.2f} | "
                f"Makespan: {makespan_str} | "
                f"LoadBalance: {avg_load_balance:.3f} | "
                f"Success: {success_rate:.2f} | "
                f"Stability: {stability_score:.3f} | "
                f"Quality: {quality_score:.3f} | "
                f"LR: {current_lr:.6f} | "
                f"Duration: {duration:.2f}s\n"
            )

            # 写入日志文件
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(detailed_log)

            # JSON格式指标
            metrics_entry = {
                'timestamp': timestamp,
                'episode': episode,
                'total_reward': convert_to_serializable(total_reward),
                'avg_makespan': convert_to_serializable(avg_makespan),
                'avg_load_balance': convert_to_serializable(avg_load_balance),
                'success_rate': convert_to_serializable(success_rate),
                'stability_score': convert_to_serializable(stability_score),
                'quality_score': convert_to_serializable(quality_score),
                'current_lr': convert_to_serializable(current_lr),
                'duration': duration
            }

            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics_entry) + '\n')

            # 控制台输出（学术化）
            if self.verbose or episode % 10 == 0:
                print(console_msg)

            return console_msg

        def log_evaluation(self, episode, eval_result):
            """记录评估信息 - 学术化格式"""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            eval_log = f"[{timestamp}] === Episode {episode} Performance Evaluation ===\n"
            for key, value in eval_result.items():
                eval_log += f"  {key}: {value}\n"
            eval_log += "=== Evaluation Completed ===\n"

            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(eval_log)

            if self.verbose:
                print(eval_log.strip())

        def log_error(self, episode, error_msg):
            """记录错误信息 - 学术化格式"""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_log = f"[{timestamp}] ERROR Episode {episode}: {error_msg}\n"

            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(error_log)

            print(error_log.strip())

        def log_convergence(self, episode, convergence_info):
            """记录收敛信息 - 学术化格式"""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conv_log = f"[{timestamp}] CONVERGENCE Episode {episode}: {convergence_info}\n"

            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(conv_log)

            print(conv_log.strip())

    return AcademicLogger(experiment_dir, verbose)


def save_enhanced_training_metrics(simulator, experiment_dir, episode, logger):
    """保存增强的训练指标 - 修复损失跟踪版本"""
    try:
        # 安全获取训练摘要
        try:
            if hasattr(simulator, 'hd_ddpg') and hasattr(simulator.hd_ddpg, 'get_training_summary'):
                training_summary = simulator.hd_ddpg.get_training_summary()
            elif hasattr(simulator, 'get_training_summary'):
                training_summary = simulator.get_training_summary()
            else:
                # 创建默认训练摘要
                training_summary = {
                    'training_history': getattr(simulator, 'training_history', {
                        'episode_rewards': [],
                        'makespans': [],
                        'load_balances': [],
                        'makespan_stability': [],
                        'td_errors': []
                    }),
                    'total_episodes': episode,
                    'average_reward_recent_100': 0,
                    'average_makespan_recent_100': 0,
                    'trend_direction': 'unknown'
                }
        except Exception as e:
            logger.log_error(episode, f"Failed to get training summary: {e}")
            training_summary = {'training_history': {}}

        history = training_summary.get('training_history', {})

        # 保存详细CSV数据 - 包含损失数据
        try:
            import pandas as pd

            episodes_count = len(history.get('episode_rewards', []))
            if episodes_count > 0:
                metrics_data = {
                    'episode': range(1, episodes_count + 1),
                    'reward': history.get('episode_rewards', [])[:episodes_count],
                    'makespan': history.get('makespans', [])[:episodes_count],
                    'load_balance': history.get('load_balances', [])[:episodes_count],
                    'stability': history.get('makespan_stability', [0] * episodes_count)[:episodes_count],
                    'td_error': history.get('td_errors', [0] * episodes_count)[:episodes_count]
                }

                # 添加损失数据 - 从全局追踪器获取
                loss_data = GLOBAL_LOSS_TRACKER.get_loss_data()

                # Meta控制器损失
                if loss_data['meta_critic']:
                    metrics_data['meta_critic_loss'] = loss_data['meta_critic'][:episodes_count] + [0] * max(0, episodes_count - len(loss_data['meta_critic']))
                else:
                    metrics_data['meta_critic_loss'] = [0] * episodes_count

                if loss_data['meta_actor']:
                    metrics_data['meta_actor_loss'] = loss_data['meta_actor'][:episodes_count] + [0] * max(0, episodes_count - len(loss_data['meta_actor']))
                else:
                    metrics_data['meta_actor_loss'] = [0] * episodes_count

                # 子控制器损失
                for cluster in ['fpga', 'fog_gpu', 'cloud']:
                    for loss_type in ['critic', 'actor']:
                        key = f'{cluster}_{loss_type}'
                        if loss_data[key]:
                            metrics_data[f'{cluster}_{loss_type}_loss'] = loss_data[key][:episodes_count] + [0] * max(0, episodes_count - len(loss_data[key]))
                        else:
                            metrics_data[f'{cluster}_{loss_type}_loss'] = [0] * episodes_count

                # 确保所有列长度一致
                min_length = min(len(v) for v in metrics_data.values() if v)
                if min_length > 0:
                    for key in metrics_data:
                        if len(metrics_data[key]) > min_length:
                            metrics_data[key] = metrics_data[key][:min_length]
                        elif len(metrics_data[key]) < min_length:
                            # 填充缺失数据
                            metrics_data[key].extend([0] * (min_length - len(metrics_data[key])))

                    metrics_df = pd.DataFrame(metrics_data)
                    csv_path = os.path.join(experiment_dir, f'ultra_optimized_metrics_ep{episode}.csv')
                    metrics_df.to_csv(csv_path, index=False)

                    if logger.verbose:
                        print(f"INFO: Ultra-optimized training metrics saved: {csv_path}")

        except ImportError:
            print("WARNING: pandas not installed, skipping CSV save")
        except Exception as e:
            logger.log_error(episode, f"CSV save error: {e}")

        # 保存损失数据CSV
        try:
            loss_csv_path = GLOBAL_LOSS_TRACKER.save_loss_data(experiment_dir)
            if loss_csv_path and logger.verbose:
                print(f"INFO: Training losses saved: {loss_csv_path}")
        except Exception as e:
            logger.log_error(episode, f"Loss data save error: {e}")

        # 保存优化的训练图表
        try:
            save_optimized_training_plots(history, experiment_dir, episode, logger)
        except Exception as e:
            logger.log_error(episode, f"Plot save error: {e}")

        # 保存缓冲区状态
        try:
            buffer_status = None
            if hasattr(simulator, 'hd_ddpg') and hasattr(simulator.hd_ddpg, 'replay_buffer'):
                if hasattr(simulator.hd_ddpg.replay_buffer, 'get_buffer_status'):
                    buffer_status = simulator.hd_ddpg.replay_buffer.get_buffer_status()
            elif hasattr(simulator, 'replay_buffer') and hasattr(simulator.replay_buffer, 'get_buffer_status'):
                buffer_status = simulator.replay_buffer.get_buffer_status()

            if buffer_status:
                buffer_path = os.path.join(experiment_dir, f'buffer_status_ep{episode}.json')
                with open(buffer_path, 'w') as f:
                    json.dump(convert_to_serializable(buffer_status), f, indent=2)
        except Exception as e:
            logger.log_error(episode, f"Buffer status save error: {e}")

    except Exception as e:
        logger.log_error(episode, f"Save training metrics error: {e}")


def evaluate_enhanced_model(simulator, experiment_dir, episode, logger):
    """增强的模型评估 - 修复版本"""
    eval_start_time = time.time()

    try:
        # 安全获取训练摘要
        try:
            if hasattr(simulator, 'hd_ddpg') and hasattr(simulator.hd_ddpg, 'get_training_summary'):
                training_summary = simulator.hd_ddpg.get_training_summary()
            elif hasattr(simulator, 'get_training_summary'):
                training_summary = simulator.get_training_summary()
            else:
                training_summary = {
                    'training_history': getattr(simulator, 'training_history', {
                        'episode_rewards': [],
                        'makespans': [],
                        'load_balances': []
                    })
                }
        except Exception as e:
            logger.log_error(episode, f"Failed to get training summary: {e}")
            training_summary = {'training_history': {}}

        # 获取优化状态
        try:
            if hasattr(simulator, 'hd_ddpg') and hasattr(simulator.hd_ddpg, 'get_optimization_status'):
                optimization_status = simulator.hd_ddpg.get_optimization_status()
            elif hasattr(simulator, 'get_optimization_status'):
                optimization_status = simulator.get_optimization_status()
            else:
                optimization_status = {
                    'episode': episode,
                    'makespan_trend': 0,
                    'makespan_variance': 0,
                    'stability_score': 0,
                    'optimization_progress': {
                        'improving': False,
                        'stable': False,
                        'converged': False
                    }
                }
        except Exception as e:
            logger.log_error(episode, f"Failed to get optimization status: {e}")
            optimization_status = {'episode': episode, 'optimization_progress': {'improving': False}}

        # 计算最近性能
        recent_episodes = 30
        history = training_summary.get('training_history', {})
        recent_rewards = history.get('episode_rewards', [])[-recent_episodes:]
        recent_makespans = [m for m in history.get('makespans', [])[-recent_episodes:]
                           if m != float('inf')]
        recent_load_balances = history.get('load_balances', [])[-recent_episodes:]

        # 计算趋势
        if len(recent_rewards) > 10:
            reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
        else:
            reward_trend = 0.0

        # 稳定性分析
        if len(recent_makespans) > 5:
            makespan_variance = np.var(recent_makespans)
            makespan_stability = 1.0 / (1.0 + makespan_variance)
        else:
            makespan_stability = 0.0

        eval_result = {
            'episode': episode,
            'recent_performance': {
                'avg_reward': float(np.mean(recent_rewards)) if recent_rewards else 0,
                'std_reward': float(np.std(recent_rewards)) if recent_rewards else 0,
                'avg_makespan': float(np.mean(recent_makespans)) if recent_makespans else float('inf'),
                'std_makespan': float(np.std(recent_makespans)) if recent_makespans else 0,
                'avg_load_balance': float(np.mean(recent_load_balances)) if recent_load_balances else 0,
                'reward_trend': float(reward_trend),
                'makespan_stability': float(makespan_stability)
            },
            'optimization_status': convert_to_serializable(optimization_status),
            'buffer_status': {},
            'evaluation_duration': time.time() - eval_start_time,
            'timestamp': datetime.now().isoformat()
        }

        # 安全获取缓冲区状态
        try:
            if hasattr(simulator, 'hd_ddpg') and hasattr(simulator.hd_ddpg, 'replay_buffer'):
                if hasattr(simulator.hd_ddpg.replay_buffer, 'get_buffer_status'):
                    buffer_status = simulator.hd_ddpg.replay_buffer.get_buffer_status()
                    eval_result['buffer_status'] = convert_to_serializable(buffer_status)
            elif hasattr(simulator, 'replay_buffer') and hasattr(simulator.replay_buffer, 'get_buffer_status'):
                buffer_status = simulator.replay_buffer.get_buffer_status()
                eval_result['buffer_status'] = convert_to_serializable(buffer_status)
        except Exception as e:
            logger.log_error(episode, f"Failed to get buffer status: {e}")

        # 记录评估结果
        logger.log_evaluation(episode, eval_result['recent_performance'])

        # 保存评估结果
        eval_path = os.path.join(experiment_dir, f'ultra_optimized_evaluation_ep{episode}.json')
        with open(eval_path, 'w') as f:
            json.dump(eval_result, f, indent=2, default=str)

        return eval_result

    except Exception as e:
        logger.log_error(episode, f"Enhanced evaluation error: {e}")
        return {'episode': episode, 'error': str(e)}


def create_fallback_simulator(config):
    """创建修复的回退仿真器"""
    class FixedTask:
        def __init__(self, task_id, computation_req=1.0, memory_req=10, priority=1):
            self.task_id = task_id
            self.computation_requirement = computation_req
            self.memory_requirement = memory_req
            self.priority = priority
            self.dependencies = []

    class FixedEnvironment:
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
                    node = type('EnhancedNode', (), {
                        'node_id': node_id,
                        'cluster_type': cluster_type,
                        'memory_capacity': config['memory'],
                        'current_load': 0,
                        'availability': True,
                        'efficiency_score': np.random.uniform(0.8, 1.0),
                        'stability_score': np.random.uniform(0.7, 1.0),
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

        def get_stabilized_system_state(self):
            return self.get_system_state()

        def get_cluster_state(self, cluster_type):
            dims = {'FPGA': 6, 'FOG_GPU': 8, 'CLOUD': 6}
            return np.random.random(dims.get(cluster_type, 6)).astype(np.float32)

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

        def get_system_health(self):
            return np.random.uniform(0.7, 1.0)

    class FixedSimulator:
        def __init__(self, config):
            self.config = config
            self.environment = FixedEnvironment()
            self.hd_ddpg = None
            # 添加完整的训练历史记录
            self.training_history = {
                'episode_rewards': [],
                'makespans': [],
                'load_balances': [],
                'makespan_stability': [],
                'td_errors': []
            }

        def _generate_episode_workflows(self):
            workflows = []
            num_workflows = self.config.get('workflows_per_episode', 8)
            min_size = self.config.get('min_workflow_size', 6)
            max_size = self.config.get('max_workflow_size', 12)

            for w_id in range(num_workflows):
                workflow_size = np.random.randint(min_size, max_size + 1)
                workflow = []

                for t_id in range(workflow_size):
                    task = FixedTask(
                        task_id=f"w{w_id}_t{t_id}",
                        computation_req=np.random.uniform(0.5, 2.0),
                        memory_req=np.random.randint(5, 25),
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
                    'average_reward_recent_100': np.mean(self.training_history['episode_rewards'][-100:]) if self.training_history['episode_rewards'] else 0,
                    'average_makespan_recent_100': np.mean([m for m in self.training_history['makespans'][-100:] if m != float('inf')]) if self.training_history['makespans'] else 0,
                    'trend_direction': 'unknown'
                }

        def get_optimization_status(self):
            """获取优化状态"""
            if hasattr(self.hd_ddpg, 'get_optimization_status'):
                return self.hd_ddpg.get_optimization_status()
            else:
                return {
                    'episode': len(self.training_history['episode_rewards']),
                    'makespan_trend': 0,
                    'makespan_variance': np.var([m for m in self.training_history['makespans'] if m != float('inf')]) if self.training_history['makespans'] else 0,
                    'stability_score': np.mean(self.training_history['makespan_stability']) if self.training_history['makespan_stability'] else 0,
                    'optimization_progress': {
                        'improving': False,
                        'stable': False,
                        'converged': False
                    }
                }

    return FixedSimulator


def print_training_summary_table(final_summary):
    """打印学术化表格格式的训练总结"""
    print("\n" + "="*80)
    print("TRAINING SUMMARY - ACADEMIC REPORT")
    print("="*80)

    # 实验信息表格
    print("\n1. EXPERIMENT INFORMATION")
    print("-" * 40)
    exp_info = final_summary['experiment_info']
    print(f"{'Experiment ID':<25}: {exp_info['experiment_id']}")
    print(f"{'Total Episodes':<25}: {exp_info['total_episodes']}")
    print(f"{'Training Time':<25}: {exp_info['total_training_time']:.2f} seconds")
    print(f"{'Avg Episode Time':<25}: {exp_info['average_episode_time']:.2f} seconds")
    print(f"{'Training Mode':<25}: {exp_info['training_mode']}")
    print(f"{'TensorFlow Version':<25}: {exp_info['tensorflow_version']}")

    # 性能指标表格
    print("\n2. PERFORMANCE METRICS")
    print("-" * 40)
    perf_metrics = final_summary['performance_metrics']
    print(f"{'Final Avg Reward':<25}: {perf_metrics['final_avg_reward']:.2f}")
    print(f"{'Best Reward':<25}: {perf_metrics['best_reward']:.2f}")
    makespan_str = f"{perf_metrics['final_avg_makespan']:.2f}" if perf_metrics['final_avg_makespan'] != float('inf') else "INF"
    print(f"{'Final Avg Makespan':<25}: {makespan_str}")
    print(f"{'Final Avg Load Balance':<25}: {perf_metrics['final_avg_load_balance']:.3f}")
    print(f"{'Final Stability Score':<25}: {perf_metrics['final_stability_score']:.3f}")
    print(f"{'Success Rate':<25}: {perf_metrics['success_rate']:.1%}")

    # 稳定化指标表格
    print("\n3. STABILIZATION METRICS")
    print("-" * 40)
    stab_metrics = final_summary['stabilization_metrics']
    print(f"{'Avg Stability Score':<25}: {stab_metrics['avg_stability_score']:.3f}")
    conv_status = "CONVERGED" if stab_metrics['convergence_achieved'] else "NOT CONVERGED"
    print(f"{'Convergence Status':<25}: {conv_status}")
    print(f"{'Trend Direction':<25}: {stab_metrics['trend_direction'].upper()}")
    print(f"{'Buffer Quality':<25}: {stab_metrics.get('buffer_quality', 0):.3f}")

    # 配置总结表格
    print("\n4. TRAINING CONFIGURATION SUMMARY")
    print("-" * 40)
    config = final_summary['training_config']
    print(f"{'Batch Size':<25}: {config['batch_size']}")
    print(f"{'Memory Capacity':<25}: {config['memory_capacity']}")
    print(f"{'Learning Rate':<25}: {config['meta_lr']:.6f}")
    print(f"{'Gamma':<25}: {config['gamma']}")
    print(f"{'Makespan Weight':<25}: {config['makespan_weight']}")
    print(f"{'Stability Weight':<25}: {config['stability_weight']}")

    print("\n" + "="*80)


def main():
    """主训练函数 - 超优化版本 with Training Losses"""
    print("HD-DDPG Medical Workflow Scheduling Training (Ultra-Optimized Version with Loss Tracking)")
    print("="*80)

    # 设置高级环境
    setup_advanced_environment()

    # 解析参数
    args = parse_enhanced_arguments()

    # 设置实验
    experiment_dir = setup_enhanced_experiment(args)

    # 创建学术化日志记录器
    logger = create_enhanced_logger(experiment_dir, args.verbose)

    # 创建超优化配置
    hd_ddpg_config = create_ultra_optimized_config(args)
    simulator_config = create_stabilized_simulator_config(args)

    print(f"Training Configuration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Workflows per episode: {args.workflows_per_episode}")
    print(f"  Batch size: {hd_ddpg_config['batch_size']}")
    print(f"  Learning rate: {hd_ddpg_config['meta_lr']:.6f}")
    print(f"  Experiment name: {args.experiment_name}")
    print(f"  Progress bar: {'Enabled' if TQDM_AVAILABLE and not args.disable_progress_bar else 'Disabled'}")
    print(f"  Loss tracking: {'Enabled' if args.enable_detailed_loss_tracking else 'Disabled'}")
    print("-" * 80)

    # 创建稳定化仿真器
    try:
        if MODULES_AVAILABLE:
            simulator = StabilizedMedicalSchedulingSimulator(simulator_config)
            print("INFO: Using StabilizedMedicalSchedulingSimulator")
        else:
            FallbackSimulator = create_fallback_simulator(simulator_config)
            simulator = FallbackSimulator(simulator_config)
            print("WARNING: Using fallback simulator")
    except Exception as e:
        logger.log_error(0, f"Simulator creation failed: {e}")
        FallbackSimulator = create_fallback_simulator(simulator_config)
        simulator = FallbackSimulator(simulator_config)
        print("WARNING: Using fallback simulator")

    # 创建HD-DDPG实例
    try:
        simulator.hd_ddpg = HDDDPG(hd_ddpg_config)
        print("INFO: Ultra-optimized HD-DDPG created successfully")
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

    print(f"\nStarting ultra-optimized training with loss tracking...")

    # 创建主进度条
    use_tqdm = TQDM_AVAILABLE and not args.disable_progress_bar

    if use_tqdm:
        main_pbar = trange(
            1, args.episodes + 1,
            desc="HD-DDPG Training Progress",
            unit="episode",
            ncols=120,
            colour='green'
        )
    else:
        main_pbar = range(1, args.episodes + 1)

    # 收敛标志
    convergence_achieved = False

    try:
        for episode in main_pbar:
            episode_start_time = time.time()

            try:
                # 生成工作流
                workflows = simulator._generate_episode_workflows()

                # 更新全局损失追踪器的当前episode
                GLOBAL_LOSS_TRACKER.current_episode = episode

                # 创建工作流处理进度条
                if use_tqdm and args.verbose:
                    workflow_desc = f"Episode {episode} - Processing Workflows"
                    workflow_pbar = tqdm(
                        workflows,
                        desc=workflow_desc,
                        leave=False,
                        ncols=100,
                        colour='blue'
                    )
                else:
                    workflow_pbar = workflows

                # 训练一个episode
                result = simulator.hd_ddpg.train_episode(workflow_pbar, simulator.environment)

                episode_duration = time.time() - episode_start_time

                # 更新训练历史
                if hasattr(simulator, 'training_history'):
                    simulator.training_history['episode_rewards'].append(result.get('total_reward', 0))
                    simulator.training_history['makespans'].append(result.get('avg_makespan', float('inf')))
                    simulator.training_history['load_balances'].append(result.get('avg_load_balance', 0))
                    simulator.training_history['makespan_stability'].append(result.get('makespan_stability', 0))
                    simulator.training_history['td_errors'].append(result.get('td_error', 0))

                # 记录损失数据（每episode记录一次）
                if episode % args.loss_logging_frequency == 0 and args.enable_detailed_loss_tracking:
                    try:
                        # 从HD-DDPG获取最新损失
                        if hasattr(simulator.hd_ddpg, 'meta_controller'):
                            meta_controller = simulator.hd_ddpg.meta_controller
                            if hasattr(meta_controller, 'last_critic_loss') and hasattr(meta_controller, 'last_actor_loss'):
                                GLOBAL_LOSS_TRACKER.log_meta_losses(
                                    meta_controller.last_critic_loss,
                                    meta_controller.last_actor_loss,
                                    episode
                                )

                        # 记录子控制器损失
                        if hasattr(simulator.hd_ddpg, 'sub_controllers'):
                            for cluster_name, controller in simulator.hd_ddpg.sub_controllers.items():
                                if hasattr(controller, 'last_critic_loss') and hasattr(controller, 'last_actor_loss'):
                                    GLOBAL_LOSS_TRACKER.log_sub_losses(
                                        cluster_name,
                                        controller.last_critic_loss,
                                        controller.last_actor_loss
                                    )
                    except Exception as e:
                        if args.verbose:
                            print(f"WARNING: Loss logging error in episode {episode}: {e}")

                # 检查收敛状态
                if hasattr(simulator.hd_ddpg, 'convergence_monitor'):
                    convergence_status = simulator.hd_ddpg.convergence_monitor.get_status()

                    # 早停检测
                    if convergence_status['early_stop']:
                        logger.log_convergence(episode, f"Convergence detected, early stopping training")
                        logger.log_convergence(episode, f"Best performance: {convergence_status['best_score']:.4f}")
                        convergence_achieved = True
                        if use_tqdm:
                            main_pbar.close()
                        break

                    # 更新进度条显示收敛信息
                    if use_tqdm:
                        main_pbar.set_postfix({
                            'Reward': f"{result.get('total_reward', 0):.1f}",
                            'Makespan': f"{result.get('avg_makespan', float('inf')):.2f}" if result.get('avg_makespan', float('inf')) != float('inf') else "INF",
                            'LR': f"{result.get('current_lr', 0):.6f}",
                            'Conv': f"{convergence_status['patience_counter']}/{simulator.hd_ddpg.convergence_monitor.patience}",
                            'Duration': f"{episode_duration:.1f}s"
                        })

                # 记录结果
                episode_rewards.append(result.get('total_reward', 0))
                episode_results.append(result)

                # 记录进度
                console_msg = logger.log_episode(episode, result, episode_duration)

                # 定期评估
                if episode % args.eval_interval == 0:
                    if use_tqdm:
                        main_pbar.set_description("Evaluating...")

                    eval_result = evaluate_enhanced_model(simulator, experiment_dir, episode, logger)

                    if use_tqdm:
                        main_pbar.set_description("HD-DDPG Training Progress")

                # 定期保存
                if episode % args.save_interval == 0:
                    if use_tqdm:
                        main_pbar.set_description("Saving...")

                    try:
                        # 保存模型
                        model_path = os.path.join(experiment_dir, 'checkpoints', f'ultra_optimized_hd_ddpg_ep{episode}')
                        simulator.hd_ddpg.save_models(model_path)

                        # 保存训练指标
                        save_enhanced_training_metrics(simulator, experiment_dir, episode, logger)

                        print(f"INFO: Episode {episode} ultra-optimized checkpoint saved")
                    except Exception as e:
                        logger.log_error(episode, f"Checkpoint save error: {e}")

                    if use_tqdm:
                        main_pbar.set_description("HD-DDPG Training Progress")

            except Exception as e:
                logger.log_error(episode, f"Episode training error: {e}")
                # 记录错误的episode结果
                episode_rewards.append(-10.0)
                episode_results.append({
                    'total_reward': -10.0,
                    'avg_makespan': float('inf'),
                    'avg_load_balance': 0.0,
                    'episode_duration': time.time() - episode_start_time,
                    'success_rate': 0.0,
                    'makespan_stability': 0.0
                })

    except KeyboardInterrupt:
        print("\nWARNING: User interrupted training")
        if use_tqdm:
            main_pbar.close()

    except Exception as e:
        logger.log_error(0, f"Training severe error: {e}")
        import traceback
        traceback.print_exc()
        if use_tqdm:
            main_pbar.close()

    finally:
        # 训练完成处理
        if use_tqdm and 'main_pbar' in locals():
            main_pbar.close()

        training_duration = time.time() - training_start_time

        print(f"\nTraining completed!")
        print(f"Total training time: {training_duration:.2f} seconds")
        print(f"Average episode time: {training_duration/max(len(episode_results), 1):.2f} seconds")

        # 保存最终模型
        final_model_path = os.path.join(experiment_dir, 'final_ultra_optimized_model')
        try:
            simulator.hd_ddpg.save_models(final_model_path)
            print(f"INFO: Final ultra-optimized model saved: {final_model_path}")
        except Exception as e:
            logger.log_error(0, f"Cannot save final model: {e}")

        # 保存最终指标
        try:
            save_enhanced_training_metrics(simulator, experiment_dir, len(episode_results), logger)
        except Exception as e:
            logger.log_error(0, f"Cannot save training metrics: {e}")

        # 生成最终报告
        try:
            # 获取训练总结
            if hasattr(simulator, 'hd_ddpg') and hasattr(simulator.hd_ddpg, 'get_training_summary'):
                training_summary = simulator.hd_ddpg.get_training_summary()
                optimization_status = simulator.hd_ddpg.get_optimization_status()
            else:
                training_summary = {'training_history': simulator.training_history}
                optimization_status = {'optimization_progress': {'converged': convergence_achieved}}

            # 计算最终统计
            valid_rewards = [r for r in episode_rewards if not np.isnan(r) and not np.isinf(r)]
            valid_results = [r for r in episode_results if r.get('avg_makespan', float('inf')) != float('inf')]

            final_summary = {
                'experiment_info': {
                    'experiment_id': args.experiment_name,
                    'total_episodes': len(episode_results),
                    'total_training_time': training_duration,
                    'average_episode_time': training_duration / max(len(episode_results), 1),
                    'training_mode': 'ULTRA_OPTIMIZED_CPU',
                    'tensorflow_version': tf.__version__,
                    'tqdm_available': TQDM_AVAILABLE,
                    'modules_available': MODULES_AVAILABLE,
                    'loss_tracking_enabled': args.enable_detailed_loss_tracking
                },
                'performance_metrics': {
                    'final_avg_reward': float(np.mean(valid_rewards[-50:])) if len(valid_rewards) >= 50 else float(np.mean(valid_rewards)) if valid_rewards else 0,
                    'best_reward': float(np.max(valid_rewards)) if valid_rewards else 0,
                    'final_avg_makespan': float(np.mean([r.get('avg_makespan', float('inf')) for r in valid_results[-20:]])) if len(valid_results) >= 20 else float(np.mean([r.get('avg_makespan', float('inf')) for r in valid_results])) if valid_results else float('inf'),
                    'final_avg_load_balance': float(np.mean([r.get('avg_load_balance', 0) for r in episode_results[-20:]])) if len(episode_results) >= 20 else float(np.mean([r.get('avg_load_balance', 0) for r in episode_results])) if episode_results else 0,
                    'final_stability_score': float(np.mean([r.get('makespan_stability', 0) for r in episode_results[-20:]])) if len(episode_results) >= 20 else float(np.mean([r.get('makespan_stability', 0) for r in episode_results])) if episode_results else 0,
                    'success_rate': len(valid_results) / len(episode_results) if episode_results else 0
                },
                'stabilization_metrics': {
                    'avg_stability_score': float(np.mean([r.get('makespan_stability', 0) for r in episode_results])) if episode_results else 0,
                    'convergence_achieved': convergence_achieved,
                    'trend_direction': training_summary.get('trend_direction', 'unknown'),
                    'buffer_quality': training_summary.get('buffer_quality', 0)
                },
                'training_config': hd_ddpg_config,
                'simulator_config': simulator_config,
                'episode_rewards': [float(x) for x in episode_rewards],
                'training_summary': convert_to_serializable(training_summary),
                'optimization_status': convert_to_serializable(optimization_status),
                'completion_timestamp': datetime.now().isoformat(),
                'loss_tracking_summary': {
                    'total_loss_records': len(GLOBAL_LOSS_TRACKER.training_losses['meta_critic']),
                    'loss_data_available': any(GLOBAL_LOSS_TRACKER.training_losses.values())
                }
            }

            # 转换为可序列化格式
            final_summary = convert_to_serializable(final_summary)

            # 保存最终总结
            summary_path = os.path.join(experiment_dir, 'final_ultra_optimized_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(final_summary, f, indent=2, default=str)

            # 打印学术化表格总结
            print_training_summary_table(final_summary)

            print(f"\nINFO: Final results saved to: {experiment_dir}")

        except Exception as e:
            logger.log_error(0, f"Cannot save training summary: {e}")
            print(f"INFO: Training completed successfully, but summary save error occurred")

    print("\n" + "="*80)
    print("HD-DDPG Ultra-Optimized Training Program Completed")


if __name__ == "__main__":
    main()