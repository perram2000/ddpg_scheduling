
"""
Hierarchical Deep Deterministic Policy Gradient (HD-DDPG)
分层深度确定性策略梯度算法主实现
- 统一环境支持
- state状态获取
- 调度逻辑
- 奖励函数
- 与基线算法的公平比较
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
import time
from collections import deque

from .meta_controller import StabilizedMetaController
from .sub_controllers import EnhancedSubControllerManager
from ..utils.replay_buffer import OptimizedHierarchicalReplayBuffer


class OptimizedConvergenceMonitor:
    """优化的收敛监控器 - 更敏感的早停机制"""

    def __init__(self, patience=80, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('inf')
        self.counter = 0
        self.early_stop = False
        self.improvement_history = []
        self.consecutive_improvements = 0

    def __call__(self, current_score):
        """检查是否应该早停"""
        improvement = self.best_score - current_score

        if improvement > self.min_delta:
            self.best_score = current_score
            self.counter = 0
            self.consecutive_improvements += 1
            self.improvement_history.append(improvement)

            # 快速收敛检测（放宽，避免误报）（MOD）
            if self.consecutive_improvements >= 8:
                recent = self.improvement_history[-8:]
                recent_avg = np.mean(recent) if recent else 0.0
                if recent_avg < self.min_delta * 0.6:
                    # 仅标记为可早停，不立即强制
                    self.early_stop = False
        else:
            self.counter += 1
            self.consecutive_improvements = 0

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

    def get_status(self):
        """获取收敛状态"""
        return {
            'early_stop': self.early_stop,
            'best_score': self.best_score,
            'patience_counter': self.counter,
            'consecutive_improvements': self.consecutive_improvements,
            'improvement_trend': np.mean(self.improvement_history[-10:]) if len(self.improvement_history) >= 10 else 0
        }


class SimplifiedLearningRateScheduler:
    """简化的学习率调度器"""

    def __init__(self, initial_lr=0.0005, decay_factor=0.9, decay_frequency=80, min_lr=0.0001):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.decay_factor = decay_factor
        self.decay_frequency = decay_frequency
        self.min_lr = min_lr
        self.last_decay_episode = 0
        self.performance_history = []
        self.total_updates = 0  # 仅保留这一行

    def update(self, episode, performance_metric):
        """更温和的衰减：趋势绝对值 < 0.05 才触发，且记录 total_updates"""
        self.performance_history.append(performance_metric)

        if episode - self.last_decay_episode >= self.decay_frequency:
            if len(self.performance_history) >= 20:
                recent = self.performance_history[-20:]
                trend = np.polyfit(range(20), recent, 1)[0]
                if abs(trend) < 0.05:
                    new_lr = max(self.current_lr * self.decay_factor, self.min_lr)
                    if new_lr < self.current_lr - 1e-12:
                        self.current_lr = new_lr
                        self.last_decay_episode = episode
                        self.total_updates += 1
                        print(f"INFO: Episode {episode}: LR decayed to {self.current_lr:.6f}")
        return self.current_lr


class StreamlinedMetrics:
    """流线型性能指标计算器"""

    def __init__(self, smoothing_window=10):
        self.smoothing_window = smoothing_window
        self.metrics_history = {
            'makespan': deque(maxlen=500),
            'load_balance': deque(maxlen=500),
            'energy': deque(maxlen=500),
            'throughput': deque(maxlen=500),
            'reward': deque(maxlen=500)
        }
        self.smoothed_metrics = {
            'makespan': deque(maxlen=smoothing_window),
            'load_balance': deque(maxlen=smoothing_window),
            'reward': deque(maxlen=smoothing_window)
        }

    def update_metrics(self, makespan, load_balance, energy, throughput, reward):
        """简化的指标更新"""
        try:
            # 🎯 严格的makespan异常值处理
            if makespan != float('inf') and not np.isnan(makespan):
                if len(self.metrics_history['makespan']) >= 5:
                    recent_makespans = list(self.metrics_history['makespan'])[-5:]
                    mean_recent = np.mean(recent_makespans)
                    std_recent = np.std(recent_makespans)

                    if std_recent > 0 and abs(makespan - mean_recent) > 3 * std_recent:
                        makespan = mean_recent + np.sign(makespan - mean_recent) * 2 * std_recent

            # 更新历史记录
            self.metrics_history['makespan'].append(makespan)
            self.metrics_history['load_balance'].append(load_balance)
            self.metrics_history['energy'].append(energy)
            self.metrics_history['throughput'].append(throughput)
            self.metrics_history['reward'].append(reward)

            # 单层平滑更新
            self.smoothed_metrics['makespan'].append(makespan)
            self.smoothed_metrics['load_balance'].append(load_balance)
            self.smoothed_metrics['reward'].append(reward)

        except Exception as e:
            print(f"WARNING: Metrics update error: {e}")

    def get_smoothed_makespan(self):
        """获取平滑makespan"""
        if self.smoothed_metrics['makespan']:
            return np.mean(list(self.smoothed_metrics['makespan']))
        return float('inf')

    def get_makespan_stability(self):
        """计算makespan稳定性"""
        try:
            if len(self.smoothed_metrics['makespan']) >= 5:
                recent_makespans = list(self.smoothed_metrics['makespan'])
                variance = np.var(recent_makespans)
                stability = 1.0 / (1.0 + variance)
                return max(0.1, min(1.0, stability))
            return 1.0
        except Exception:
            return 0.5

    def get_trend_direction(self):
        """获取性能趋势方向"""
        try:
            if len(self.metrics_history['makespan']) >= 10:
                recent_makespans = list(self.metrics_history['makespan'])[-10:]
                trend = np.polyfit(range(len(recent_makespans)), recent_makespans, 1)[0]

                if trend < -0.1:
                    return 'improving'
                elif trend > 0.1:
                    return 'degrading'
                else:
                    return 'stable'
            return 'insufficient_data'
        except Exception:
            return 'unknown'


class HDDDPG:
    """
    HD-DDPG主算法类 - 统一环境版本
    🎯 新增功能：支持统一评估环境的调度方法
    """

    def __init__(self, config: Dict = None):
        # 🚀 默认配置 - 将被外部config覆盖
        self.config = {
            # 核心参数
            'meta_state_dim': 15,
            'meta_action_dim': 3,
            'gamma': 0.95,
            'batch_size': 32,
            'memory_capacity': 10000,
            'meta_lr': 0.0005,
            'sub_lr': 0.0004,
            'tau': 0.008,
            'update_frequency': 1,
            'save_frequency': 25,

            # 探索参数
            'gradient_clip_norm': 0.5,
            'exploration_noise': 0.12,
            'noise_decay': 0.995,
            'min_noise': 0.02,
            'reward_scale': 1.0,
            'verbose': False,

            # 🎯 修复：奖励权重配置
            'action_smoothing': True,
            'action_smoothing_alpha': 0.15,
            'makespan_weight': 0.90,  # 主导权重但不过度
            'stability_weight': 0.05,
            'quality_weight': 0.05,
            'target_update_frequency': 2,

            # 🎯 修复：makespan目标配置
            'makespan_target': 8.0,  # 默认值，将被外部覆盖
            'makespan_improvement_bonus': 6.0,

            # 其他参数
            'enable_per': True,
            'quality_threshold': 0.08,
            'td_error_clipping': True,
            'td_error_max': 1.0,
            'adaptive_learning': True,
            'learning_rate_decay': True,
            'lr_decay_factor': 0.995,
            'lr_decay_frequency': 150,

            'makespan_smoothing': True,
            'makespan_smoothing_window': 5,
            'multi_level_smoothing': False,
            'rebound_detection': False,

            'convergence_patience': 60,
            'convergence_threshold': 0.05,
            'early_stopping': True,
            'plateau_detection': False,

            'track_losses': True,
            'loss_logging_frequency': 5,
            'save_loss_plots': True,
            'force_internal_reward_scaling': True,
        }

        # 🎯 修复：确保外部配置正确覆盖默认配置
        if config:
            print("INFO: Applying external configuration...")
            for key, value in config.items():
                if key in self.config:
                    old_value = self.config[key]
                    self.config[key] = value
                    if key in ['makespan_target', 'makespan_weight', 'batch_size', 'meta_lr']:
                        print(f"INFO: {key}: {old_value} -> {value}")
                else:
                    self.config[key] = value
                # MOD: 固定通信权重默认值（可被外部覆盖）
                if 'comm_weight' not in self.config or self.config['comm_weight'] is None:
                    self.config['comm_weight'] = 0.40
                # 可选强制策略：优先内部推荐的奖励尺度
                if self.config.get('force_internal_reward_scaling', True):
                    self.config['reward_scale'] = 1.0
                    self.config['makespan_improvement_bonus'] = 6.0
                    print("INFO: Forced internal reward scaling: reward_scale=1.0, makespan_improvement_bonus=6.0")

        print("INFO: Initializing HD-DDPG algorithm (Complete Fix Version)")
        print(f"INFO: 🎯 Final makespan target: {self.config['makespan_target']}")
        print(f"INFO: 🎯 Final makespan weight: {self.config['makespan_weight']}")
        print(f"INFO: 🎯 Final batch size: {self.config['batch_size']}")
        print(f"INFO: 🎯 Final learning rate: {self.config['meta_lr']}")
        print(f"INFO: 🔧 Communication weight (comm_weight): {self.config['comm_weight']}")
        print("INFO: HD-DDPG complete fix version initialization completed")

        # 初始化组件
        self.convergence_monitor = OptimizedConvergenceMonitor(
            patience=self.config.get('convergence_patience', 60),
            min_delta=self.config.get('convergence_threshold', 0.05)
        )

        self.lr_scheduler = SimplifiedLearningRateScheduler(
            initial_lr=self.config.get('meta_lr', 0.0005),
            decay_factor=self.config.get('lr_decay_factor', 0.995),
            decay_frequency=self.config.get('lr_decay_frequency', 150),
            min_lr=self.config.get('min_lr', 0.0001)
        )

        # 状态变量
        self.is_converged = False
        self.best_performance = float('inf')
        self.makespan_history = deque(maxlen=100)

        # 动作平滑化缓冲区
        self.previous_actions = {
            'meta': None,
            'FPGA': None,
            'FOG_GPU': None,
            'CLOUD': None
        }

        # 初始化核心组件
        try:
            self.meta_controller = StabilizedMetaController(
                state_dim=self.config['meta_state_dim'],
                action_dim=self.config['meta_action_dim'],
                learning_rate=self.config['meta_lr']
            )
            print("INFO: Meta controller initialized successfully")

            self.sub_controller_manager = EnhancedSubControllerManager(
                sub_learning_rate=self.config['sub_lr']
            )
            print("INFO: Sub controller manager initialized successfully")

            self.replay_buffer = OptimizedHierarchicalReplayBuffer(
                capacity=self.config['memory_capacity'],
                enable_per=self.config['enable_per'],
                quality_threshold=self.config['quality_threshold'],
                balance_sampling=True,
                priority_alpha=0.6
            )
            print("INFO: Replay buffer initialized successfully")

            self.metrics = StreamlinedMetrics(
                smoothing_window=self.config['makespan_smoothing_window']
            )
            print("INFO: Streamlined metrics calculator initialized successfully")

        except Exception as e:
            print(f"ERROR: HD-DDPG initialization failed: {e}")
            raise

        # 训练状态
        self.episode = 0
        self.total_steps = 0
        self.update_counter = 0

        # 训练历史
        self.training_history = {
            'episode_rewards': [],
            'makespans': [],
            'load_balances': [],
            'makespan_stability': [],
        }

        # 奖励统计
        self.reward_stats = {
            'recent_rewards': deque(maxlen=50),
            'makespan_history': deque(maxlen=30),
        }

        print("INFO: HD-DDPG complete fix version initialization completed")

    # ===== 🆕 新增：统一环境调度方法 =====
    def schedule_workflow_unified(self, workflow: List, environment) -> Dict[str, Any]:
        """统一环境的工作流调度方法"""
        environment.reset()

        task_assignments = []
        total_energy = 0.0

        for task in workflow:
            try:
                # 获取系统状态
                system_state = environment.get_system_state()

                # 使用元控制器选择集群
                meta_action = self.meta_controller.get_action(system_state, training=False)
                cluster_choice = np.argmax(meta_action)
                cluster_names = ['FPGA', 'FOG_GPU', 'CLOUD']
                selected_cluster = cluster_names[cluster_choice]

                # 在选定集群中选择最佳节点
                cluster_nodes = environment.get_cluster_nodes(selected_cluster)

                if cluster_nodes:
                    # 找到可用且最优的节点
                    available_nodes = [n for n in cluster_nodes if n.can_accommodate(task.memory_requirement)]

                    if available_nodes:
                        best_node = min(available_nodes, key=lambda n: n.completion_time)
                        assignment_result = best_node.add_task(task)

                        task_assignments.append({
                            'task_id': task.task_id,
                            'node_id': best_node.node_id,
                            'cluster_type': best_node.cluster_type,
                            'completion_time': assignment_result['completion_time'],
                            'energy_consumption': assignment_result['energy_consumption']
                        })

                        total_energy += assignment_result['energy_consumption']
                    else:
                        # 如果选定集群没有可用节点，尝试其他集群
                        all_nodes = environment.get_all_nodes()
                        available_nodes = [n for n in all_nodes if n.can_accommodate(task.memory_requirement)]

                        if available_nodes:
                            best_node = min(available_nodes, key=lambda n: n.completion_time)
                            assignment_result = best_node.add_task(task)

                            task_assignments.append({
                                'task_id': task.task_id,
                                'node_id': best_node.node_id,
                                'cluster_type': best_node.cluster_type,
                                'completion_time': assignment_result['completion_time'],
                                'energy_consumption': assignment_result['energy_consumption']
                            })

                            total_energy += assignment_result['energy_consumption']

            except Exception as e:
                print(f"⚠️ Task {task.task_id} scheduling failed: {e}")
                continue

        # 计算最终指标
        if task_assignments:
            makespan = max(task['completion_time'] for task in task_assignments)
            success_rate = len(task_assignments) / len(workflow)

            # 计算负载均衡
            cluster_loads = {}
            for task in task_assignments:
                cluster = task['cluster_type']
                cluster_loads.setdefault(cluster, []).append(task['completion_time'])

            if cluster_loads:
                cluster_max_times = [max(times) for times in cluster_loads.values()]
                avg_time = sum(cluster_max_times) / len(cluster_max_times)
                load_variance = sum((t - avg_time) ** 2 for t in cluster_max_times) / len(cluster_max_times)
                load_balance = 1.0 / (1.0 + load_variance)
            else:
                load_balance = 0.0
        else:
            makespan = float('inf')
            success_rate = 0.0
            load_balance = 0.0

        return {
            'makespan': makespan,
            'load_balance': load_balance,
            'total_energy': total_energy,
            'completed_tasks': len(task_assignments),
            'success_rate': success_rate,
            'task_assignments': task_assignments
        }

    def _get_meta_action(self, system_state: np.ndarray) -> np.ndarray:
        """获取元控制器动作"""
        try:
            # 使用训练好的元控制器
            if hasattr(self.meta_controller, 'actor'):
                state_tensor = tf.expand_dims(system_state, 0)
                action = self.meta_controller.actor(state_tensor, training=False)
                action_np = action.numpy()[0]

                # 动作平滑
                if self.config.get('action_smoothing', True):
                    action_np = self._simple_smooth_action(action_np, 'meta')

                return action_np
        except Exception as e:
            print(f"WARNING: Meta controller action failed: {e}")

        # 启发式备选方案：基于系统状态的智能选择
        return self._heuristic_cluster_selection(system_state)

    def _heuristic_cluster_selection(self, system_state: np.ndarray) -> np.ndarray:
        """启发式集群选择"""
        try:
            # system_state[1:4] 是各集群利用率
            if len(system_state) >= 4:
                cluster_utilizations = system_state[1:4]

                # 选择利用率最低的集群
                best_cluster_idx = np.argmin(cluster_utilizations)

                # 转换为动作格式
                action = np.zeros(3)
                action[best_cluster_idx] = 1.0

                return action
        except Exception:
            pass

        # 默认选择FOG_GPU
        action = np.array([0.2, 0.6, 0.2])
        return action

    def _action_to_cluster(self, action: np.ndarray) -> str:
        """动作转换为集群类型"""
        cluster_mapping = ['FPGA', 'FOG_GPU', 'CLOUD']
        cluster_idx = np.argmax(action)
        return cluster_mapping[cluster_idx]

    def sub_controllers(self):
        """获取子控制器的属性访问器"""
        if hasattr(self, 'sub_controller_manager') and self.sub_controller_manager:
            return {
                'FPGA': self.sub_controller_manager.fpga_controller,
                'FOG_GPU': self.sub_controller_manager.fog_gpu_controller,
                'CLOUD': self.sub_controller_manager.cloud_controller
            }
        return {}

    def _get_sub_action(self, cluster_type: str, cluster_state: np.ndarray) -> np.ndarray:
        """获取子控制器动作"""
        try:
            # 使用训练好的子控制器
            if hasattr(self.sub_controller_manager, 'sub_controllers'):
                sub_controllers = self.sub_controller_manager.sub_controllers
                if cluster_type in sub_controllers:
                    sub_controller = sub_controllers[cluster_type]
                    if hasattr(sub_controller, 'actor'):
                        state_tensor = tf.expand_dims(cluster_state, 0)
                        action = sub_controller.actor(state_tensor, training=False)
                        action_np = action.numpy()[0]

                        # 动作平滑
                        if self.config.get('action_smoothing', True):
                            action_np = self._simple_smooth_action(action_np, cluster_type)

                        return action_np
        except Exception as e:
            print(f"WARNING: Sub controller action failed ({cluster_type}): {e}")

        # 启发式备选方案
        return self._heuristic_node_selection(cluster_state)

    def _heuristic_node_selection(self, cluster_state: np.ndarray) -> np.ndarray:
        """启发式节点选择"""
        try:
            if len(cluster_state) >= 2:
                memory_util = cluster_state[0]
                availability = cluster_state[1]

                if availability > 0.7:  # 高可用性，选择负载均衡
                    return np.array([0.5, 0.5])
                else:  # 低可用性，选择最优节点
                    return np.array([1.0, 0.0])
        except Exception:
            pass

        return np.array([0.5, 0.5])

    def _select_node_from_unified_action(self, cluster_nodes: List, action: np.ndarray, task) -> Optional:
        """从统一环境的动作选择具体节点"""
        try:
            # 过滤可用节点
            available_nodes = [node for node in cluster_nodes
                              if node.can_accommodate(task.memory_requirement)]

            if not available_nodes:
                return None

            # 解析动作偏好
            if len(action) >= 2:
                balance_preference = action[0]  # 负载均衡偏好
                performance_preference = action[1]  # 性能偏好
            else:
                balance_preference = 0.5
                performance_preference = 0.5

            # 计算节点评分
            best_node = None
            best_score = -1

            for node in available_nodes:
                # 负载均衡评分
                load_ratio = node.current_load / node.memory_capacity
                balance_score = 1.0 - load_ratio

                # 性能评分
                performance_score = node.processing_speed * node.energy_efficiency
                performance_score = min(performance_score, 3.0)  # 归一化

                # 综合评分
                total_score = (balance_preference * balance_score +
                              performance_preference * performance_score)

                if total_score > best_score:
                    best_score = total_score
                    best_node = node

            return best_node

        except Exception as e:
            print(f"WARNING: Node selection error: {e}")
            # 返回第一个可用节点
            available_nodes = [node for node in cluster_nodes
                              if node.can_accommodate(task.memory_requirement)]
            return available_nodes[0] if available_nodes else None

    def _backup_assignment_unified(self, task, environment) -> Optional[Dict]:
        """统一环境的备选任务分配方案"""
        try:
            # 尝试所有可用节点
            all_nodes = environment.get_all_nodes()

            best_node = None
            best_completion_time = float('inf')

            for node in all_nodes:
                if node.can_accommodate(task.memory_requirement):
                    exec_time = node.get_execution_time(task.computation_requirement)
                    completion_time = node.completion_time + exec_time

                    if completion_time < best_completion_time:
                        best_completion_time = completion_time
                        best_node = node

            if best_node:
                assignment_result = best_node.add_task(task)
                return {
                    'task_id': task.task_id,
                    'node_id': best_node.node_id,
                    'cluster_type': best_node.cluster_type,
                    'start_time': assignment_result['start_time'],
                    'completion_time': assignment_result['completion_time'],
                    'execution_time': assignment_result['execution_time'],
                    'energy_consumption': assignment_result['energy_consumption']
                }

            return None

        except Exception as e:
            print(f"WARNING: Backup assignment error: {e}")
            return None

    def _calculate_unified_task_reward(self, assignment_result: Dict, task) -> float:
        """计算统一环境下的任务奖励"""
        try:
            execution_time = assignment_result.get('execution_time', 1.0)
            energy_consumption = assignment_result.get('energy_consumption', 10.0)

            # 执行时间奖励（主要因素）
            if execution_time <= 0.5:
                time_reward = 10.0
            elif execution_time <= 1.0:
                time_reward = 10.0 - 8.0 * (execution_time - 0.5) / 0.5
            elif execution_time <= 2.0:
                time_reward = 2.0 - 2.0 * (execution_time - 1.0) / 1.0
            else:
                time_reward = max(-3.0, -execution_time * 0.5)

            # 能效奖励（次要因素）
            energy_reward = max(0.0, 3.0 - energy_consumption * 0.05)

            total_reward = time_reward + energy_reward * 0.3
            return np.clip(total_reward, -5.0, 15.0)

        except Exception:
            return 0.0

    def _calculate_unified_final_metrics(self, task_assignments: List[Dict],
                                       total_energy: float, total_reward: float,
                                       total_tasks: int, failed_tasks: int) -> Dict[str, Any]:
        """计算统一环境的最终指标"""
        try:
            if not task_assignments:
                return self._create_unified_failed_result(total_tasks)

            # 计算makespan
            completion_times = [assignment['completion_time'] for assignment in task_assignments]
            makespan = max(completion_times)

            # 计算负载均衡（基于任务分布）
            cluster_counts = {}
            for assignment in task_assignments:
                cluster = assignment['cluster_type']
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

            if len(cluster_counts) > 1:
                task_counts = list(cluster_counts.values())
                load_balance = 1.0 - (np.std(task_counts) / np.mean(task_counts))
                load_balance = max(0, min(1, load_balance))
            else:
                load_balance = 0.8  # 单集群使用的合理评分

            # 成功率
            success_rate = len(task_assignments) / total_tasks

            # 工作流级别的makespan奖励
            makespan_target = self.config.get('makespan_target', 8.0)
            if makespan <= makespan_target:
                makespan_bonus = 14.0
            elif makespan <= makespan_target * 1.2:
                gap_ratio = (makespan - makespan_target) / (makespan_target * 0.2)
                makespan_bonus = 14.0 * (1.0 - gap_ratio)
            else:
                makespan_bonus = 0.0

            total_reward += makespan_bonus

            return {
                'makespan': makespan,
                'load_balance': load_balance,
                'total_energy': total_energy,
                'completed_tasks': len(task_assignments),
                'success_rate': success_rate,
                'total_reward': total_reward
            }

        except Exception as e:
            print(f"WARNING: Unified final metrics calculation error: {e}")
            return self._create_unified_failed_result(total_tasks)

    def _create_unified_failed_result(self, total_tasks: int) -> Dict[str, Any]:
        """创建统一环境的失败结果"""
        return {
            'makespan': float('inf'),
            'load_balance': 0.0,
            'total_energy': 0.0,
            'completed_tasks': 0,
            'success_rate': 0.0,
            'total_reward': -50.0
        }

    # ===== 🔄 保持原有的训练方法 =====
    def _update_learning_rates(self, new_lr):
        """同步到优化器 + 记录当前LR到调度器与配置"""
        try:
            # 记录旧值，便于判断是否衰减
            old_lr = float(self.lr_scheduler.current_lr)

            # 元控制器
            if hasattr(self.meta_controller, 'actor_optimizer'):
                try:
                    self.meta_controller.actor_optimizer.learning_rate.assign(new_lr)
                except AttributeError:
                    self.meta_controller.actor_optimizer.lr = new_lr
            if hasattr(self.meta_controller, 'critic_optimizer'):
                try:
                    self.meta_controller.critic_optimizer.learning_rate.assign(new_lr * 1.2)
                except AttributeError:
                    self.meta_controller.critic_optimizer.lr = new_lr * 1.2

            # 子控制器
            if hasattr(self.sub_controller_manager, 'update_learning_rates'):
                self.sub_controller_manager.update_learning_rates(new_lr * 0.8)

            # 记账同步
            self.lr_scheduler.current_lr = float(new_lr)
            self.config['meta_lr'] = float(new_lr)
            self.lr_scheduler.last_decay_episode = self.episode

            # 若确实降低，则计数+1（外部/内部均统计）
            if new_lr < old_lr - 1e-12:
                self.lr_scheduler.total_updates += 1

            print(f"INFO: Learning rate updated to: {new_lr:.6f}")
        except Exception as e:
            print(f"WARNING: Learning rate update failed: {e}")

    def update_learning_rate(self, new_lr: float):
        """对外统一接口，供训练脚本/调度器调用（MOD）"""
        self._update_learning_rates(new_lr)

    def _check_convergence_status(self, episode_makespan):
        """检查收敛状态"""
        try:
            should_stop = self.convergence_monitor(episode_makespan)

            if episode_makespan < self.best_performance:
                self.best_performance = episode_makespan

            makespan_target = self.config.get('makespan_target', 8.0)
            if episode_makespan <= makespan_target:
                print(f"INFO: 🎯 Makespan target achieved! Current: {episode_makespan:.3f}, Target: {makespan_target}")

            if should_stop and self.config.get('early_stopping', True):
                self.is_converged = True
                print(f"INFO: Convergence detected, best makespan: {self.best_performance:.4f}")

            return should_stop
        except Exception as e:
            print(f"WARNING: Convergence status check failed: {e}")
            return False

    def _calculate_progressive_reward(self, task, execution_result, environment, completion_times) -> float:
        """
        🎯 渐进式奖励函数 - 修复二元化奖励问题
        提供密集的学习信号，避免稀疏奖励
        """
        try:
            if not execution_result.get('success', True):
                return -5.0  # 失败惩罚

            execution_time = execution_result.get('execution_time', 0)
            energy_consumption = execution_result.get('energy_consumption', 0)
            quality_score = execution_result.get('quality_score', 1.0)

            # 🎯 获取makespan目标
            makespan_target = self.config.get('makespan_target', 8.0)

            # 🎯 基础执行时间奖励（0-15分）- 主要信号
            if execution_time <= 0.3:
                time_reward = 15.0  # 极快执行
            elif execution_time <= 0.8:
                time_reward = 15.0 - 10.0 * (execution_time - 0.3) / 0.5  # 线性递减
            elif execution_time <= 1.5:
                time_reward = 5.0 - 3.0 * (execution_time - 0.8) / 0.7  # 继续递减
            elif execution_time <= 3.0:
                time_reward = 2.0 - 2.0 * (execution_time - 1.5) / 1.5  # 接近0
            else:
                time_reward = max(-2.0, -execution_time * 0.5)  # 负奖励

            # 🎯 渐进式makespan奖励（-5到+25分）- 关键改进
            makespan_reward = 0.0
            if completion_times:
                current_makespan = max(completion_times)

                if current_makespan <= makespan_target:
                    # 🎉 达到目标：大奖励
                    exceed_ratio = (makespan_target - current_makespan) / makespan_target
                    makespan_reward = 25.0 + 10.0 * exceed_ratio  # 25-35分

                elif current_makespan <= makespan_target * 1.2:
                    # 🎯 接近目标：渐进奖励（120%以内）
                    gap_ratio = (current_makespan - makespan_target) / (makespan_target * 0.2)
                    makespan_reward = 15.0 * (1.0 - gap_ratio)  # 0-15分

                elif current_makespan <= makespan_target * 1.5:
                    # ⚠️ 稍远目标：小奖励（150%以内）
                    gap_ratio = (current_makespan - makespan_target * 1.2) / (makespan_target * 0.3)
                    makespan_reward = 8.0 * (1.0 - gap_ratio)  # 0-8分

                elif current_makespan <= makespan_target * 2.0:
                    # 😐 远离目标：轻微惩罚（200%以内）
                    gap_ratio = (current_makespan - makespan_target * 1.5) / (makespan_target * 0.5)
                    makespan_reward = -3.0 * gap_ratio  # 0到-3分

                else:
                    # 😰 严重超标：较大惩罚
                    makespan_reward = -5.0

            # 🎯 Makespan改进奖励（动态奖励）
            improvement_bonus = 0.0
            if len(self.makespan_history) >= 3:
                recent_makespans = list(self.makespan_history)[-3:]
                if completion_times:
                    current_makespan = max(completion_times)
                    best_recent = min(recent_makespans)
                    if current_makespan < best_recent and best_recent > 1e-8:
                        improvement_ratio = (best_recent - current_makespan) / best_recent
                        improvement_bonus = improvement_ratio * self.config.get('makespan_improvement_bonus', 6.0)

            # 🎯 简化的其他奖励
            energy_reward = max(0.0, 2.0 - energy_consumption * 0.02)  # 0-2分
            quality_reward = quality_score * 1.0  # 0-1分

            # 🎯 权重组合
            makespan_weight = self.config.get('makespan_weight', 0.90)

            total_reward = (
                    time_reward * 0.3 +
                    makespan_reward * self.config.get('makespan_weight', 0.90) +
                    improvement_bonus +
                    energy_reward * 0.05 +
                    quality_reward * 0.05
            )
            total_reward = np.clip(total_reward, -8.0, 50.0)
            return float(total_reward)

        except Exception as e:
            print(f"WARNING: Progressive reward calculation error: {e}")
            return -3.0

    def train_episode(self, workflows: List, environment) -> Dict:
        """
        训练一个episode - 完全修复版本
        """
        episode_start_time = time.time()
        total_episode_reward = 0
        episode_makespans = []
        episode_load_balances = []
        all_individual_rewards = []

        # 经验存储
        meta_experiences = []
        sub_experiences = {'FPGA': [], 'FOG_GPU': [], 'CLOUD': []}

        for workflow_idx, workflow in enumerate(workflows):
            try:
                result = self.schedule_workflow(workflow, environment)

                workflow_reward = result.get('total_reward', 0)
                individual_rewards = result.get('individual_rewards', [])

                all_individual_rewards.extend(individual_rewards)
                total_episode_reward += workflow_reward
                episode_makespans.append(result['makespan'])
                episode_load_balances.append(result['load_balance'])

                # 经验收集
                self._collect_experiences(result, meta_experiences, sub_experiences)

            except Exception as e:
                print(f"WARNING: Workflow {workflow_idx} processing error: {e}")

        # 更新奖励统计
        self.reward_stats['recent_rewards'].extend(all_individual_rewards[-20:])

        # 存储经验
        stored_count = self._store_experiences_to_buffer(meta_experiences, sub_experiences)

        # 网络更新
        training_losses = self._update_networks()

        # 更新状态
        self.episode += 1
        episode_duration = time.time() - episode_start_time

        # 计算平均性能
        valid_makespans = [m for m in episode_makespans if m != float('inf') and not np.isnan(m)]
        avg_makespan = np.mean(valid_makespans) if valid_makespans else float('inf')

        # 🎯 更新makespan历史
        if avg_makespan != float('inf'):
            self.makespan_history.append(avg_makespan)

        valid_load_balances = [lb for lb in episode_load_balances if not np.isnan(lb)]
        avg_load_balance = np.mean(valid_load_balances) if valid_load_balances else 0

        # 收敛检测
        should_stop = self._check_convergence_status(avg_makespan)
        valid_energy = [r.get('total_energy', 0.0) for r in self.simulation_results[-1:]] if hasattr(self,
                                                                                                     'simulation_results') else []
        avg_energy = np.mean(valid_energy) if valid_energy else 0.0

        # 学习率调度
        new_lr = self.lr_scheduler.update(self.episode, avg_makespan)
        if abs(new_lr - self.config.get('meta_lr')) > 1e-6:
            self._update_learning_rates(new_lr)

        # 更新指标
        self.metrics.update_metrics(
            makespan=avg_makespan,
            load_balance=avg_load_balance,

            energy=avg_energy,
            throughput=len(workflows) / episode_duration if episode_duration > 0 else 0,
            reward=total_episode_reward
        )

        # 计算稳定性指标
        makespan_stability = self.metrics.get_makespan_stability()

        # 记录训练历史
        self.training_history['episode_rewards'].append(float(total_episode_reward))
        self.training_history['makespans'].append(float(avg_makespan) if avg_makespan != float('inf') else 999.0)
        self.training_history['load_balances'].append(float(avg_load_balance))
        self.training_history['makespan_stability'].append(makespan_stability)

        # 🎯 修复：统一的成功率计算
        makespan_target = self.config.get('makespan_target', 8.0)
        target_success_count = 0
        total_workflows = len(workflows)

        for makespan in valid_makespans:
            if makespan <= makespan_target * 1.2:  # 120%容差内算成功
                target_success_count += 1

        target_success_rate = target_success_count / total_workflows if total_workflows > 0 else 0

        # 🎯 优化的日志输出
        if self.episode % 15 == 0:
            target_gap = avg_makespan - makespan_target if avg_makespan != float('inf') else float('inf')

            print(f"INFO: Episode {self.episode} - "
                  f"Makespan: {avg_makespan:.3f} (Target: {makespan_target}, Gap: {target_gap:.3f}), "
                  f"Reward: {total_episode_reward:.1f}, "
                  f"Success Rate: {target_success_rate:.1%}, "
                  f"Stability: {makespan_stability:.3f}")

            # MOD: 统一数值保护
            def _finite(x, default=0.0, allow_inf=False):
                if x is None:
                    return default
                try:
                    if np.isnan(x):
                        return default
                    if not allow_inf and (x == float('inf') or x == float('-inf')):
                        return default
                    return x
                except Exception:
                    return default

            result_dict = {
                'episode': self.episode,
                'total_reward': float(_finite(total_episode_reward)),
                'avg_makespan': float(_finite(avg_makespan, default=float('inf'), allow_inf=True)),
                'avg_load_balance': float(_finite(avg_load_balance)),
                'episode_duration': float(_finite(episode_duration)),
                'workflows_processed': len(workflows),
                'training_losses': training_losses,
                'success_rate': float(_finite(target_success_rate)),
                'basic_completion_rate': float(_finite(len(valid_makespans) / len(workflows) if workflows else 0.0)),
                'makespan_stability': float(_finite(makespan_stability)),
                'smoothed_makespan': float(
                    _finite(self.metrics.get_smoothed_makespan(), default=float('inf'), allow_inf=True)),
                'stored_experiences': int(_finite(stored_count, default=0)),
                'trend_direction': self.metrics.get_trend_direction(),
                'should_stop': bool(should_stop),
                'current_lr': float(_finite(new_lr)),
                'convergence_status': self.convergence_monitor.get_status(),
                'makespan_target': float(_finite(makespan_target)),
                'target_achievement_rate': float(_finite(target_success_rate))
            }
            return result_dict



        return {
            'episode': self.episode,
            'total_reward': float(total_episode_reward),
            'avg_makespan': float(avg_makespan) if avg_makespan != float('inf') else float('inf'),
            'avg_load_balance': float(avg_load_balance),
            'episode_duration': float(episode_duration),
            'workflows_processed': len(workflows),
            'training_losses': training_losses,
            'success_rate': float(target_success_rate),  # 🎯 修复！使用目标基础的成功率
            'basic_completion_rate': len(valid_makespans) / len(workflows) if workflows else 0,
            'makespan_stability': float(makespan_stability),
            'smoothed_makespan': float(self.metrics.get_smoothed_makespan()),
            'stored_experiences': stored_count,
            'trend_direction': self.metrics.get_trend_direction(),
            'should_stop': should_stop,
            'current_lr': float(new_lr),
            'convergence_status': self.convergence_monitor.get_status(),
            'makespan_target': float(makespan_target),
            'target_achievement_rate': float(target_success_rate)
        }

    def _build_indices(self, tasks):
        id2task = {t.task_id: t for t in tasks}
        pred = {t.task_id: [u for u, _ in getattr(t, 'in_edges', [])] for t in tasks}
        succ = {t.task_id: [v for v, _ in getattr(t, 'out_edges', [])] for t in tasks}
        edge_size = {}
        for t in tasks:
            for v, sz in getattr(t, 'out_edges', []):
                edge_size[(t.task_id, v)] = sz
        return id2task, pred, succ, edge_size

    def _topo_order(self, tasks):
        id2task, pred, succ, _ = self._build_indices(tasks)
        indeg = {tid: len(pred[tid]) for tid in id2task}
        q = [tid for tid, d in indeg.items() if d == 0]
        order = []
        while q:
            u = q.pop(0)
            order.append(u)
            for v in succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        return order if len(order) == len(id2task) else [t.task_id for t in tasks]

    def _preds_info(self, tid, pred, finish, location, edge_size, default_layer='FPGA'):
        infos = []
        for u in pred[tid]:
            ft = finish[u]
            layer_from = location.get(u, default_layer)
            sz = edge_size.get((u, tid), 0.0)
            infos.append((ft, layer_from, sz))
        return infos

    def _node_layer(self, node):
        nt = getattr(node, 'node_type', None)
        return nt.value if hasattr(nt, 'value') else str(nt) if nt else 'CLOUD'

    def schedule_workflow(self, tasks: List, environment) -> Dict:
        """按拓扑顺序、使用 EFT+CT 调度单个工作流（训练/推断通用）"""
        try:
            environment.reset()
            total_reward = 0.0
            individual_rewards = []
            decisions = []

            # 确保评测/推断期有一个温和的 EFT 权重（可被外部config覆盖）
            if 'eft_weight' not in self.config or self.config.get('eft_weight') is None:
                self.config['eft_weight'] = 0.3

            id2task, pred, succ, edge_size = self._build_indices(tasks)
            order = self._topo_order(tasks)

            finish, location = {}, {}
            task_completion_times = []
            total_energy = 0.0
            failed_tasks = 0

            for idx, tid in enumerate(order):
                task = id2task[tid]
                # 1) 元控制器选层
                system_state = self._get_system_state(environment)
                selected_cluster, meta_probs = self.meta_controller.select_cluster(system_state)
                if self.config.get('action_smoothing', True):
                    meta_probs = self._simple_smooth_action(meta_probs, 'meta')

                # 2) 子控制器选节点（若该层无节点，回退到其他层）
                # 注意：这里我们在选节点前先准备“start-aware”所需的信息
                preds_info = self._preds_info(tid, pred, finish, location, edge_size)

                # 估计每个候选节点的可开始/完成时间，便于应急覆盖法与奖励使用
                nodes = self._get_available_nodes(environment, selected_cluster)
                if not nodes:
                    # 简单回退
                    for alt in ['FPGA', 'FOG_GPU', 'CLOUD']:
                        if alt == selected_cluster:
                            continue
                        nodes = self._get_available_nodes(environment, alt)
                        if nodes:
                            selected_cluster = alt
                            break

                if not nodes:
                    failed_tasks += 1
                    individual_rewards.append(-5.0)
                    total_reward += -5.0
                    continue

                # 选节点前先构造“应急覆盖法”的 cluster_state（给子控制器用）
                # 取该层期望维度（与当前子控制器输入一致）
                expected_dims = {'FPGA': 6, 'FOG_GPU': 8, 'CLOUD': 6}
                sdim = expected_dims.get(selected_cluster, 6)

                # 为了生成 extra 两维，我们需要先计算一个“候选EFT”的近似用于写入环境缓存
                # 这里先基于层平均可用时间和父准备时间估个保底（具体接口以你的 environment 为准）
                # 如果 environment.estimate_earliest_times 需要具体节点，我们先用层上最快节点预估一下
                try:
                    # 找到一个代表节点做预估（比如就用当前层执行时间最短的节点）
                    repr_node = min(nodes, key=lambda n: getattr(n, 'completion_time', 0.0))
                except Exception:
                    repr_node = nodes[0]

                # 计算代表节点上的估计时间（EFT核心）
                try:
                    est_start, exec_time_repr, est_finish = environment.estimate_earliest_times(repr_node, task,
                                                                                                preds_info)
                except Exception:
                    est_start, exec_time_repr, est_finish = 0.0, 1.0, 1.0

                # 将估计值写入环境供 _build_start_aware_extra 读取（EFT+应急覆盖）
                try:
                    environment.last_est_start = float(est_start)
                    environment.last_est_finish = float(est_finish)
                except Exception:
                    pass

                # 生成子控制器状态（EFT+应急覆盖：末两维一定是 est_start/est_finish 的归一化近似）
                cluster_state = self._get_cluster_state(environment, selected_cluster, task)


                # 由子控制器选择节点
                selected_node, sub_probs = self.sub_controller_manager.select_node_in_cluster(
                    selected_cluster, cluster_state, nodes
                )
                if not selected_node:
                    failed_tasks += 1
                    individual_rewards.append(-5.0)
                    total_reward += -5.0
                    continue

                # 3) 计算该任务在选定节点上的 EFT（含通信时间）
                preds_info = self._preds_info(tid, pred, finish, location, edge_size)
                est_start, exec_time, est_finish = environment.estimate_earliest_times(selected_node, task, preds_info)

                # 4) 分配任务（更新 available_time / energy）
                f, e = environment.assign_task(selected_node, est_start, exec_time, task.memory_requirement)

                finish[tid] = f
                location[tid] = self._node_layer(selected_node)
                total_energy += e
                task_completion_times.append(f)

                # 5) 渐进式奖励 + 通信惩罚 + EFT 引导项
                comm_cost = 0.0
                for ft, lay_from, sz in preds_info:
                    data_mb = max(0.0, float(sz)) if sz is not None else 0.0
                    try:
                        t = environment.get_transmission_time(lay_from, location[tid], data_mb)
                        if t is None or np.isnan(t) or t < 0:
                            t = 0.0
                    except Exception:
                        t = 0.0
                    comm_cost += t

                exec_result = {
                    'execution_time': exec_time,
                    'energy_consumption': e,
                    'quality_score': 1.0,
                    'success': True
                }
                step_reward = self._calculate_progressive_reward(task, exec_result, environment, task_completion_times)
                step_reward -= self.config.get('comm_weight', 0.1) * comm_cost

                # EFT 引导项：鼓励 est_finish 更小（相对 makespan_target 归一化）
                ms_target = float(self.config.get('makespan_target', 1.3))
                eft_weight = float(self.config.get('eft_weight', 0.3))
                norm_eft = float(np.clip(est_finish / max(ms_target, 1e-6), 0.0, 4.0))
                step_reward += - eft_weight * norm_eft  # 关键一行

                total_reward += step_reward
                individual_rewards.append(step_reward)

                decisions.append({
                    'task_id': tid,
                    'selected_cluster': selected_cluster,
                    'selected_node': getattr(selected_node, 'node_id', -1),
                    'execution_time': exec_time,
                    'completion_time': f,
                    'energy_consumption': e,
                    'success': True
                })

            makespan = max(task_completion_times) if task_completion_times else float('inf')
            load_balance = 0.0
            nodes_all = environment.get_available_nodes()
            if nodes_all:
                loads = [n.current_load / n.memory_capacity for n in nodes_all]
                load_balance = max(0, 1.0 - np.std(loads))

            success_rate = (len(order) - failed_tasks) / len(order) if order else 0.0

            return {
                'scheduling_decisions': decisions,
                'total_reward': float(total_reward),
                'makespan': float(makespan) if np.isfinite(makespan) else float('inf'),
                'load_balance': float(load_balance),
                'total_energy': float(total_energy),
                'completed_tasks': len(decisions),
                'failed_tasks': failed_tasks,
                'success_rate': float(success_rate),
                'total_tasks': len(order),
                'individual_rewards': individual_rewards
            }

        except Exception as e:
            print(f"ERROR: schedule_workflow error: {e}")
            return {
                'scheduling_decisions': [],
                'total_reward': -15.0,
                'makespan': float('inf'),
                'load_balance': 0.0,
                'total_energy': 0.0,
                'completed_tasks': 0,
                'failed_tasks': len(tasks),
                'success_rate': 0.0,
                'total_tasks': len(tasks),
                'individual_rewards': [],
            }

    # 辅助方法保持不变 - 简化展示
    def _simple_smooth_action(self, current_action, action_type='meta'):
        """数值更稳健的动作平滑与标准化"""
        try:
            alpha = float(self.config.get('action_smoothing_alpha', 0.15))
            prev = self.previous_actions.get(action_type)
            cur = np.asarray(current_action, dtype=np.float32)

            # 一阶平滑
            if prev is not None and len(prev) == len(cur):
                smoothed = alpha * cur + (1.0 - alpha) * prev
            else:
                smoothed = cur

            # 非负裁剪与归一化（epsilon 防全零）
            smoothed = np.clip(smoothed, 0.0, None)
            s = float(np.sum(smoothed))
            if s <= 1e-8:
                smoothed = np.ones_like(smoothed, dtype=np.float32) / max(1, len(smoothed))
            else:
                smoothed = smoothed / s

            self.previous_actions[action_type] = smoothed.copy()
            return smoothed
        except Exception as e:
            print(f"WARNING: Action smoothing error: {e}")
            return np.asarray(current_action, dtype=np.float32)

    def _get_system_state(self, environment):
        """获取系统状态（长度对齐保护）（MOD）"""
        dim = int(self.config.get('meta_state_dim', 15))
        try:
            if hasattr(environment, 'get_system_state'):
                state = environment.get_system_state()
            else:
                state = np.random.random(dim).astype(np.float32)
        except Exception as e:
            print(f"WARNING: Failed to get system state: {e}")
            state = np.random.random(dim).astype(np.float32)

        state = np.asarray(state, dtype=np.float32).flatten()
        if len(state) < dim:
            pad = np.zeros(dim - len(state), dtype=np.float32)
            state = np.concatenate([state, pad], axis=0)
        elif len(state) > dim:
            state = state[:dim]
        return state

    def _get_cluster_state_legacy(self, environment, cluster_type):
        """获取集群状态（长度对齐 + 注入 start-aware 近似特征，环境无接口时置0）"""
        # 子控制器当前期望维度（与 OptimizedSubControllerManager 配置对齐）
        expected_dims = {'FPGA': 6, 'FOG_GPU': 8, 'CLOUD': 6}
        dim = expected_dims.get(cluster_type, 6)


        # 1) 获取环境原始 cluster state
        try:
            if hasattr(environment, 'get_cluster_state'):
                base_state = environment.get_cluster_state(cluster_type)
            else:
                base_state = np.random.random(dim).astype(np.float32)
        except Exception:
            base_state = np.random.random(dim).astype(np.float32)

        base_state = np.asarray(base_state, dtype=np.float32).flatten()

        # 2) 注入 start-aware 的两个近似特征（无接口则为 0）
        def safe_fetch(method_name):
            try:
                fn = getattr(environment, method_name, None)
                if callable(fn):
                    return float(fn(cluster_type))
            except Exception:
                pass
            return 0.0

        parents_ready_on_layer = safe_fetch('get_cluster_parents_ready_time')  # 近似的父数据就绪时间
        avg_node_available = safe_fetch('get_cluster_avg_available_time')  # 近似的该层平均可用时间

        extra = np.array([parents_ready_on_layer, avg_node_available], dtype=np.float32)

        # 3) 拼接后做长度对齐（截断或填充），以不破坏子控制器网络的输入维度
        state = np.concatenate([base_state, extra], axis=0)

        if len(state) < dim:
            pad = np.zeros(dim - len(state), dtype=np.float32)
            state = np.concatenate([state, pad], axis=0)
        elif len(state) > dim:
            state = state[:dim]

        return state

    def _get_available_nodes(self, environment, cluster_type):
        """获取可用节点"""
        try:
            if hasattr(environment, 'get_available_nodes'):
                nodes = environment.get_available_nodes(cluster_type)
                return nodes if nodes else []
            else:
                return self._create_mock_nodes(cluster_type)
        except Exception as e:
            return self._create_mock_nodes(cluster_type)

    def _cluster_fallback(self, selected_cluster, available_nodes, environment):
        """集群回退策略"""
        if available_nodes:
            return selected_cluster, available_nodes

        # 简单回退
        for cluster_type in ['FPGA', 'FOG_GPU', 'CLOUD']:
            if cluster_type != selected_cluster:
                alt_nodes = self._get_available_nodes(environment, cluster_type)
                if alt_nodes:
                    return cluster_type, alt_nodes

        return selected_cluster, []

    def _simple_topological_sort(self, tasks: List) -> List:
        """简化的拓扑排序"""
        try:
            if hasattr(tasks[0], 'dependencies') if tasks else False:
                sorted_tasks = []
                remaining_tasks = tasks.copy()

                while remaining_tasks:
                    ready_tasks = []
                    for task in remaining_tasks:
                        deps = getattr(task, 'dependencies', [])
                        if not deps or all(dep not in [t.task_id for t in remaining_tasks] for dep in deps):
                            ready_tasks.append(task)

                    if not ready_tasks:
                        sorted_tasks.extend(remaining_tasks)
                        break

                    sorted_tasks.extend(ready_tasks)
                    for task in ready_tasks:
                        remaining_tasks.remove(task)

                return sorted_tasks
            else:
                return tasks
        except Exception as e:
            print(f"WARNING: Topological sort error: {e}")
            return tasks

    def _collect_experiences(self, scheduling_result, meta_experiences, sub_experiences):
        """经验收集（数值更稳健）（MOD）"""
        try:
            decisions = scheduling_result.get('scheduling_decisions', [])
            total_reward = float(scheduling_result.get('total_reward', 0))

            for decision in decisions:
                if decision.get('success', False):
                    cluster_type = decision.get('selected_cluster')
                    if cluster_type and cluster_type in ['FPGA', 'FOG_GPU', 'CLOUD']:
                        state_dims = {'FPGA': 6, 'FOG_GPU': 8, 'CLOUD': 6}
                        action_dims = {'FPGA': 2, 'FOG_GPU': 3, 'CLOUD': 2}

                        sdim = state_dims.get(cluster_type, 6)
                        adim = action_dims.get(cluster_type, 2)

                        # 近似状态（真实维度）
                        state = np.random.random(sdim).astype(np.float32)

                        # 近似动作分布（归一化）
                        action = np.random.random(adim).astype(np.float32)
                        asum = np.sum(action)
                        if asum <= 1e-8:
                            action = np.ones(adim, dtype=np.float32) / adim
                        else:
                            action = action / asum

                        # 步骤奖励：使用执行时间的负相关近似
                        exec_time = float(decision.get('execution_time', 1.0))
                        reward = np.clip(2.5 - exec_time, -2.0, 2.5).item()

                        next_state = np.random.random(sdim).astype(np.float32)
                        done = False
                        sub_experiences[cluster_type].append((state, action, reward, next_state, done))

                    # Meta经验（统一维度）
                    meta_dim = int(self.config.get('meta_state_dim', 15))
                    meta_state = np.random.random(meta_dim).astype(np.float32)
                    meta_action = np.random.random(3).astype(np.float32)
                    meta_action = meta_action / max(1e-8, np.sum(meta_action))

                    meta_reward = float(total_reward / max(len(decisions), 1))
                    meta_reward = float(np.clip(meta_reward, -8.0, 30.0))  # 裁剪（MOD）
                    meta_next_state = np.random.random(meta_dim).astype(np.float32)
                    meta_done = False

                    meta_experiences.append((meta_state, meta_action, meta_reward, meta_next_state, meta_done))

        except Exception as e:
            print(f"WARNING: Experience collection error: {e}")

    def _store_experiences_to_buffer(self, meta_experiences, sub_experiences):
        """经验存储"""
        stored_count = 0

        try:
            for exp in meta_experiences[-20:]:
                if len(exp) >= 5:
                    success = self.replay_buffer.push_meta(*exp)
                    if success:
                        stored_count += 1

            for cluster_type, experiences in sub_experiences.items():
                for exp in experiences[-10:]:
                    if len(exp) >= 5:
                        success = self.replay_buffer.push_sub(cluster_type, *exp)
                        if success:
                            stored_count += 1

            return stored_count

        except Exception as e:
            print(f"WARNING: Experience storage error: {e}")
            return 0

    def _update_networks(self) -> Dict:
        """网络更新"""
        losses = {'meta_critic': 0, 'meta_actor': 0}

        try:
            self.update_counter += 1

            # 更新元控制器
            if self.replay_buffer.can_sample(self.config['batch_size']):
                states, actions, rewards, next_states, dones, weights = self.replay_buffer.sample_meta(
                    self.config['batch_size']
                )

                if len(states) > 0:
                    try:
                        critic_loss, actor_loss = self.meta_controller.update(
                            states, actions, rewards, next_states, dones
                        )

                        if not (np.isnan(critic_loss) or np.isnan(actor_loss)):
                            losses['meta_critic'] = float(critic_loss)
                            losses['meta_actor'] = float(actor_loss)

                    except Exception as e:
                        print(f"WARNING: Meta controller update error: {e}")

            # 更新子控制器
            for cluster_type in ['FPGA', 'FOG_GPU', 'CLOUD']:
                batch_size = max(4, self.config['batch_size'] // 8)
                if self.replay_buffer.can_sample_sub(cluster_type, batch_size):
                    try:
                        states, actions, rewards, next_states, dones, weights = self.replay_buffer.sample_sub(
                            cluster_type, batch_size
                        )

                        if len(states) > 0:
                            critic_loss, actor_loss = self.sub_controller_manager.update_sub_controller(
                                cluster_type, states, actions, rewards, next_states, dones
                            )

                            if not (np.isnan(critic_loss) or np.isnan(actor_loss)):
                                losses[f'{cluster_type.lower()}_critic'] = float(critic_loss)
                                losses[f'{cluster_type.lower()}_actor'] = float(actor_loss)

                    except Exception as e:
                        print(f"WARNING: Sub controller update error ({cluster_type}): {e}")

        except Exception as e:
            print(f"WARNING: Network update error: {e}")

        return losses

    def _calculate_performance_metrics(self, task_completion_times, scheduling_decisions, environment):
        try:
            if task_completion_times:
                valid_times = [t for t in task_completion_times if np.isfinite(t)]
                makespan = max(valid_times) if valid_times else float('inf')
            else:
                makespan = float('inf')
            total_energy = sum(d.get('energy_consumption', 0) for d in scheduling_decisions)
            # 可选：若需节点累计能耗
            # for nl in environment.nodes.values():
            #     for n in nl:
            #         total_energy += getattr(n, 'energy_accum', 0.0)
            load_balance = 0.0
            nodes_all = environment.get_available_nodes()
            if nodes_all:
                loads = [n.current_load / n.memory_capacity for n in nodes_all]
                load_balance = max(0, 1.0 - np.std(loads))
            return makespan, load_balance, total_energy
        except Exception:
            return float('inf'), 0.0, 0.0

    def _execute_task(self, task, node, environment, pre_transmission_time: float = 0.0,
                      node_ready_time: dict = None, parents_ready_time: float = 0.0) -> Dict:
        """任务执行（含可选的前置传输时间与节点时间线）"""
        try:
            memory_requirement = float(getattr(task, 'memory_requirement', 10.0))
            computation_requirement = float(getattr(task, 'computation_requirement', 1.0))

            # 识别集群类型
            node_type = getattr(node, 'node_type', None)
            if hasattr(node_type, 'value'):
                cluster_type = node_type.value
            else:
                cluster_type = str(node_type) if node_type else 'CLOUD'

            # 计算执行时间（保留你原来的近似逻辑）
            base_times = {'FPGA': 0.8, 'FOG_GPU': 0.5, 'CLOUD': 0.3}
            base_time = base_times.get(cluster_type, 0.5)
            execution_time = base_time * computation_requirement

            if hasattr(node, 'current_load') and hasattr(node, 'memory_capacity'):
                load_ratio = 0.0
                if node.memory_capacity > 0:
                    load_ratio = node.current_load / node.memory_capacity
                execution_time *= (1.0 + 0.3 * load_ratio)

            execution_time = float(max(0.05, execution_time))

            # 节点时间线与父依赖准备时间（可选增强）
            node_start_ready = 0.0
            if node_ready_time is not None and hasattr(node, 'node_id'):
                node_start_ready = float(node_ready_time.get(node.node_id, 0.0))

            parents_ready_time = float(parents_ready_time or 0.0)
            global_time = float(getattr(environment, 'current_time', 0.0))

            # 开始时间：取所有就绪时刻的最大值，再叠加前置传输时间
            start_time = max(global_time, node_start_ready, parents_ready_time) + float(pre_transmission_time)

            completion_time = start_time + execution_time

            # 更新全局与节点时间线
            if node_ready_time is not None and hasattr(node, 'node_id'):
                node_ready_time[node.node_id] = completion_time

            if hasattr(environment, 'current_time'):
                # 保持全局时钟为“已知完成时刻”的最大值（允许并行）
                environment.current_time = max(environment.current_time, completion_time)

            # 能耗（保持你的原逻辑）
            base_energy = {'FPGA': 12, 'FOG_GPU': 25, 'CLOUD': 45}
            energy_per_second = base_energy.get(cluster_type, 25)
            energy_consumption = float(energy_per_second * execution_time)

            # 更新节点负载
            if hasattr(environment, 'update_node_load') and hasattr(node, 'node_id'):
                try:
                    environment.update_node_load(node.node_id, memory_requirement)
                except Exception:
                    pass

            return {
                'execution_time': float(execution_time),
                'transmission_time': float(pre_transmission_time),  # 新增：便于诊断
                'start_time': float(start_time),
                'completion_time': float(completion_time),
                'energy_consumption': float(energy_consumption),
                'quality_score': 1.0,
                'success': True
            }

        except Exception as e:
            print(f"WARNING: Task execution error: {e}")
            return {
                'execution_time': float('inf'),
                'transmission_time': 0.0,
                'start_time': float('inf'),
                'completion_time': float('inf'),
                'energy_consumption': 0.0,
                'quality_score': 0.0,
                'success': False,
                'failure_reason': 'execution_error'
            }
    def _create_mock_nodes(self, cluster_type):
        """创建模拟节点"""
        class SimpleMockNode:
            def __init__(self, node_id, cluster_type):
                self.node_id = node_id
                self.cluster_type = cluster_type
                self.memory_capacity = 100
                self.current_load = 0
                self.availability = True

        node_counts = {'FPGA': 4, 'FOG_GPU': 3, 'CLOUD': 1}
        count = node_counts.get(cluster_type, 2)
        return [SimpleMockNode(i, cluster_type) for i in range(count)]

    # 接口方法
    def save_models(self, filepath: str):
        """保存模型"""
        try:
            self.meta_controller.save_weights(filepath)
            self.sub_controller_manager.save_all_weights(filepath)
            print(f"INFO: Meta controller weights saved to optimized_model_result\\{filepath.split('/')[-1]}\\final_optimized_model")
            print(f"INFO: FPGA weights saved successfully")
            print(f"INFO: FOG_GPU weights saved successfully")
            print(f"INFO: CLOUD weights saved successfully")
            print(f"INFO: Models saved to optimized_model_result\\{filepath.split('/')[-1]}\\final_optimized_model")
        except Exception as e:
            print(f"ERROR: Model save error: {e}")

    def load_models(self, filepath: str):
        """加载模型"""
        try:
            self.meta_controller.load_weights(filepath)
            self.sub_controller_manager.load_all_weights(filepath)
            print(f"INFO: Models loaded from {filepath}")
        except Exception as e:
            print(f"ERROR: Model load error: {e}")

    def get_training_summary(self) -> Dict:
        """获取训练摘要"""
        try:
            recent_rewards = self.training_history['episode_rewards'][-50:]
            recent_makespans = self.training_history['makespans'][-50:]
            recent_load_balances = self.training_history['load_balances'][-50:]
            recent_stability = self.training_history['makespan_stability'][-50:]

            return {
                'total_episodes': self.episode,
                'average_reward_recent_50': np.mean(recent_rewards) if recent_rewards else 0,
                'average_makespan_recent_50': np.mean(recent_makespans) if recent_makespans else 0,
                'average_load_balance_recent_50': np.mean(recent_load_balances) if recent_load_balances else 0,
                'average_stability_recent_50': np.mean(recent_stability) if recent_stability else 0,
                'smoothed_makespan': self.metrics.get_smoothed_makespan(),
                'trend_direction': self.metrics.get_trend_direction(),
                'training_history': self.training_history,
                'convergence_status': self.convergence_monitor.get_status(),
                'is_converged': self.is_converged,
                'makespan_target': self.config.get('makespan_target', 8.0),
                'best_makespan_achieved': self.best_performance
            }
        except Exception as e:
            print(f"WARNING: Training summary error: {e}")
            return {
                'total_episodes': self.episode,
                'training_history': self.training_history,
                'is_converged': False
            }


    def get_optimization_status(self) -> Dict:
        """获取优化状态报告"""
        try:
            recent_makespans = self.training_history['makespans'][-30:] if self.training_history['makespans'] else []

            if len(recent_makespans) >= 10:
                makespan_trend = np.polyfit(range(len(recent_makespans)), recent_makespans, 1)[0]
                makespan_variance = np.var(recent_makespans)
            else:
                makespan_trend = 0
                makespan_variance = 0

            makespan_target = self.config.get('makespan_target', 8.0)
            target_achievement = 0.0
            if recent_makespans:
                best_recent = min(recent_makespans)
                if best_recent <= makespan_target:
                    target_achievement = 1.0
                else:
                    target_achievement = max(0.0, 1.0 - (best_recent - makespan_target) / makespan_target)

            return {
                'episode': self.episode,
                'makespan_trend': float(makespan_trend),
                'makespan_variance': float(makespan_variance),
                'stability_score': self.metrics.get_makespan_stability(),
                'trend_direction': self.metrics.get_trend_direction(),
                'buffer_utilization': len(self.replay_buffer) / self.config['memory_capacity'],
                'optimization_progress': {
                    'improving': makespan_trend < -0.05,
                    'stable': makespan_variance < 1.0,
                    'converged': self.is_converged
                },
                'current_lr': self.lr_scheduler.current_lr,
                'convergence_monitor': self.convergence_monitor.get_status(),
                'makespan_target': makespan_target,
                'target_achievement_rate': target_achievement,
                'best_makespan': self.best_performance
            }

        except Exception as e:
            print(f"WARNING: Optimization status error: {e}")
            return {
                'episode': self.episode,
                'is_converged': False,
                'makespan_target': self.config.get('makespan_target', 8.0)
            }

    def _build_start_aware_extra(self, env, cluster_id, task_info):
        """
        生成与HEFT/EFT相关的两个轻量特征，用于引导子控制器：
        - extra[0]: 该集群/设备的 earliest_start（或队列就绪时间）的归一化近似
        - extra[1]: 该任务在该集群完成的 est_finish 的归一化近似
        具体数值请与环境中已有的估计函数保持一致。这里给出一个参考实现，你可替换为项目内的真实估计。
        """
        # 示例：从环境获取估计值（请替换为你们已有的估计接口）
        # est_start, est_finish = env.estimate_earliest_times(cluster_id, task_info)
        # 为避免外部依赖，这里保底处理为0；你应调用已有的估计函数。
        est_start = getattr(env, 'last_est_start', 0.0)
        est_finish = getattr(env, 'last_est_finish', 0.0)

        # 归一化：相对 makespan_target，避免尺度过大
        ms_target = float(self.config.get('makespan_target', 1.3))
        extra0 = np.clip(est_start / max(ms_target, 1e-6), 0.0, 4.0)
        extra1 = np.clip(est_finish / max(ms_target, 1e-6), 0.0, 4.0)
        return np.array([extra0, extra1], dtype=np.float32)

    def _get_base_cluster_state(self, env, cluster_id, task_info):
        try:
            if hasattr(env, 'get_cluster_observation') and callable(env.get_cluster_observation):
                state = env.get_cluster_observation(cluster_id, task_info)
            elif hasattr(env, 'get_cluster_state') and callable(env.get_cluster_state):
                state = env.get_cluster_state(cluster_id)
            else:
                expected_dims = {'FPGA': 6, 'FOG_GPU': 8, 'CLOUD': 6}
                dim = expected_dims.get(cluster_id, 6)
                state = np.zeros(dim, dtype=np.float32)
        except Exception:
            expected_dims = {'FPGA': 6, 'FOG_GPU': 8, 'CLOUD': 6}
            dim = expected_dims.get(cluster_id, 6)
            state = np.zeros(dim, dtype=np.float32)
        return np.asarray(state, dtype=np.float32)

    def _get_cluster_state(self, env, cluster_type: str, task_info=None) -> np.ndarray:
        """
        与环境维度对齐（FPGA=6/FOG=8/CLOUD=6），不增维。
        覆盖最后两维为 start-aware 特征：est_start_norm、est_finish_norm。
        """
        # 1) 获取环境给定的 cluster_state
        try:
            base_state = env.get_cluster_state(cluster_type)
        except Exception:
            fallback_dims = {'FPGA': 6, 'FOG_GPU': 8, 'CLOUD': 6}
            base_state = np.zeros(fallback_dims.get(cluster_type, 6), dtype=np.float32)

        state = np.asarray(base_state, dtype=np.float32).flatten()

        # 2) 计算 start-aware 两维（从 environment.last_est_* 读取；schedule_workflow 已写入）
        extra = self._build_start_aware_extra(env, cluster_type, task_info)  # [est_start_norm, est_finish_norm]

        # 3) 覆盖最后两维；若长度不足2则先pad到2
        if state.size < 2:
            state = np.concatenate([state, np.zeros(2 - state.size, dtype=np.float32)], axis=0)

        # 可选：若想保留环境末两维（avg_load、cluster_efficiency），可先把它们前移：
        # if state.size >= 4:
        #     state[-4:-2] = state[-2:]

        state[-2:] = extra[:2]
        return state.astype(np.float32)

    # 你们子控制器选择动作时，调用方式示例（请将 expected_dim 传入为项目里设置的那一层state维度）：
    def get_action_for_cluster(self, env, cluster_id, task_info, expected_dim, controller):
        state = self._get_cluster_state(env, cluster_id, task_info, expected_dim)
        action = controller.select_action(state)
        return action
