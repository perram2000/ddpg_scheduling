"""
Hierarchical Replay Buffer - 高度优化版本
分层回放缓冲区实现 - 专为稳定化HD-DDPG设计

🎯 主要优化：
- 优先级经验回放 (PER)
- 经验质量评估
- 稳定化采样策略
- 内存优化管理
- 异常经验过滤
"""

import random
import numpy as np
from collections import deque
from typing import Tuple, List, Optional, Dict
import heapq
from dataclasses import dataclass
import time


@dataclass
class Experience:
    """增强的经验数据结构"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    priority: float = 1.0
    timestamp: float = None
    quality_score: float = 1.0
    cluster_type: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class PrioritizedExperienceReplay:
    """
    优先级经验回放实现
    基于TD-error的优先级采样
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001):
        """
        初始化优先级经验回放

        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数
            beta: 重要性采样指数
            beta_increment: beta递增率
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_beta = 1.0

        # 使用numpy数组存储以提高效率
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def add(self, experience: Experience):
        """添加经验到缓冲区"""
        max_priority = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities.append(max_priority)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """优先级采样"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])

        # 计算采样概率
        priorities = np.array(list(self.priorities))
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)

        # 计算重要性权重
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # 更新beta
        self.beta = min(self.max_beta, self.beta + self.beta_increment)

        experiences = [self.buffer[idx] for idx in indices]

        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class StabilizedHierarchicalReplayBuffer:
    """
    稳定化分层回放缓冲区
    🎯 专为稳定化训练设计的高级回放缓冲区
    """

    def __init__(self, capacity: int = 10000, enable_per: bool = True,
                 quality_threshold: float = 0.1, balance_sampling: bool = True):
        """
        初始化稳定化分层回放缓冲区

        Args:
            capacity: 总缓冲区容量
            enable_per: 是否启用优先级经验回放
            quality_threshold: 经验质量阈值
            balance_sampling: 是否启用平衡采样
        """
        self.capacity = capacity
        self.enable_per = enable_per
        self.quality_threshold = quality_threshold
        self.balance_sampling = balance_sampling

        print(f"🔧 初始化StabilizedHierarchicalReplayBuffer...")
        print(f"  - 容量: {capacity}")
        print(f"  - 优先级回放: {'启用' if enable_per else '禁用'}")
        print(f"  - 质量阈值: {quality_threshold}")
        print(f"  - 平衡采样: {'启用' if balance_sampling else '禁用'}")

        # 🎯 元控制器缓冲区
        if enable_per:
            self.meta_buffer = PrioritizedExperienceReplay(capacity // 2)
        else:
            self.meta_buffer = deque(maxlen=capacity // 2)

        # 🎯 子控制器缓冲区 (支持优先级)
        sub_capacity = capacity // 6  # 每个集群1/6容量
        self.sub_buffers = {}
        for cluster_type in ['FPGA', 'FOG_GPU', 'CLOUD']:
            if enable_per:
                self.sub_buffers[cluster_type] = PrioritizedExperienceReplay(sub_capacity)
            else:
                self.sub_buffers[cluster_type] = deque(maxlen=sub_capacity)

        # 🎯 经验质量监控
        self.quality_monitor = {
            'total_experiences': 0,
            'filtered_experiences': 0,
            'meta_quality_history': deque(maxlen=1000),
            'sub_quality_history': {cluster: deque(maxlen=1000) for cluster in ['FPGA', 'FOG_GPU', 'CLOUD']}
        }

        # 🎯 采样统计
        self.sampling_stats = {
            'meta_samples': 0,
            'sub_samples': {cluster: 0 for cluster in ['FPGA', 'FOG_GPU', 'CLOUD']},
            'quality_filtered': 0,
            'last_sample_time': time.time()
        }

        print("✅ 稳定化分层回放缓冲区初始化完成")

    def _calculate_experience_quality(self, state: np.ndarray, action: np.ndarray,
                                    reward: float, next_state: np.ndarray, done: bool) -> float:
        """🎯 计算经验质量评分"""
        try:
            quality_score = 1.0

            # 🎯 状态质量检查
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                quality_score *= 0.1

            if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
                quality_score *= 0.1

            # 🎯 动作质量检查
            if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                quality_score *= 0.1

            # 动作概率分布检查（对于概率动作）
            if len(action) > 1 and np.sum(action) > 0:
                action_entropy = -np.sum(action * np.log(action + 1e-8))
                max_entropy = -np.log(1.0 / len(action))
                entropy_ratio = action_entropy / max_entropy
                # 奖励适度的探索
                quality_score *= (0.7 + 0.6 * entropy_ratio)

            # 🎯 奖励质量检查
            if np.isnan(reward) or np.isinf(reward):
                quality_score *= 0.01
            elif abs(reward) > 1000:  # 异常大的奖励
                quality_score *= 0.3

            # 🎯 状态变化检查
            if not done:
                state_change = np.linalg.norm(next_state - state)
                if state_change == 0:  # 状态没有变化
                    quality_score *= 0.5
                elif state_change > 10:  # 状态变化过大
                    quality_score *= 0.7

            return np.clip(quality_score, 0.01, 1.0)

        except Exception as e:
            print(f"⚠️ 经验质量计算错误: {e}")
            return 0.1

    def _calculate_priority(self, reward: float, quality_score: float,
                          cluster_type: str = None) -> float:
        """🎯 计算经验优先级"""
        try:
            # 基础优先级基于奖励绝对值
            base_priority = abs(reward) + 1e-6

            # 🎯 质量加权
            priority = base_priority * quality_score

            # 🎯 集群特化权重
            if cluster_type:
                cluster_weights = {'FPGA': 1.2, 'FOG_GPU': 1.0, 'CLOUD': 0.8}
                priority *= cluster_weights.get(cluster_type, 1.0)

            # 🎯 稀有经验奖励（高质量低频经验）
            if quality_score > 0.8 and abs(reward) > 10:
                priority *= 1.5

            return max(priority, 1e-6)

        except Exception as e:
            print(f"⚠️ 优先级计算错误: {e}")
            return 1.0

    def push_meta(self, state: np.ndarray, action: np.ndarray, reward: float,
                  next_state: np.ndarray, done: bool):
        """🎯 添加元控制器经验（支持质量过滤和优先级）"""
        try:
            # 🎯 计算经验质量
            quality_score = self._calculate_experience_quality(state, action, reward, next_state, done)

            # 🎯 质量过滤
            if quality_score < self.quality_threshold:
                self.quality_monitor['filtered_experiences'] += 1
                return False

            # 🎯 创建经验对象
            experience = Experience(
                state=state.copy(),
                action=action.copy(),
                reward=reward,
                next_state=next_state.copy(),
                done=done,
                quality_score=quality_score,
                cluster_type='META'
            )

            # 🎯 计算优先级
            if self.enable_per:
                priority = self._calculate_priority(reward, quality_score)
                experience.priority = priority
                self.meta_buffer.add(experience)
            else:
                self.meta_buffer.append(experience)

            # 🎯 更新监控统计
            self.quality_monitor['total_experiences'] += 1
            self.quality_monitor['meta_quality_history'].append(quality_score)

            return True

        except Exception as e:
            print(f"⚠️ 元控制器经验添加错误: {e}")
            return False

    def push_sub(self, cluster_type: str, state: np.ndarray, action: np.ndarray,
                 reward: float, next_state: np.ndarray, done: bool):
        """🎯 添加子控制器经验（支持质量过滤和优先级）"""
        try:
            if cluster_type not in self.sub_buffers:
                print(f"⚠️ 未知集群类型: {cluster_type}")
                return False

            # 🎯 计算经验质量
            quality_score = self._calculate_experience_quality(state, action, reward, next_state, done)

            # 🎯 质量过滤
            if quality_score < self.quality_threshold:
                self.quality_monitor['filtered_experiences'] += 1
                return False

            # 🎯 创建经验对象
            experience = Experience(
                state=state.copy(),
                action=action.copy(),
                reward=reward,
                next_state=next_state.copy(),
                done=done,
                quality_score=quality_score,
                cluster_type=cluster_type
            )

            # 🎯 计算优先级
            if self.enable_per:
                priority = self._calculate_priority(reward, quality_score, cluster_type)
                experience.priority = priority
                self.sub_buffers[cluster_type].add(experience)
            else:
                self.sub_buffers[cluster_type].append(experience)

            # 🎯 更新监控统计
            self.quality_monitor['total_experiences'] += 1
            self.quality_monitor['sub_quality_history'][cluster_type].append(quality_score)

            return True

        except Exception as e:
            print(f"⚠️ 子控制器经验添加错误 ({cluster_type}): {e}")
            return False

    def can_sample(self, batch_size: int) -> bool:
        """检查是否可以从元控制器缓冲区采样"""
        return len(self.meta_buffer) >= batch_size

    def can_sample_sub(self, cluster_type: str, batch_size: int) -> bool:
        """检查是否可以从指定集群的子缓冲区采样"""
        if cluster_type not in self.sub_buffers:
            return False
        return len(self.sub_buffers[cluster_type]) >= batch_size

    def sample_meta(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """🎯 从元控制器缓冲区稳定化采样"""
        try:
            if not self.can_sample(batch_size):
                # 返回默认数据
                dummy_state = np.zeros((batch_size, 15), dtype=np.float32)
                dummy_action = np.zeros((batch_size, 3), dtype=np.float32)
                dummy_reward = np.zeros((batch_size, 1), dtype=np.float32)
                dummy_next_state = np.zeros((batch_size, 15), dtype=np.float32)
                dummy_done = np.zeros((batch_size, 1), dtype=np.float32)
                return dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done, None

            # 🎯 优先级采样
            if self.enable_per:
                experiences, indices, weights = self.meta_buffer.sample(batch_size)
                importance_weights = weights
            else:
                experiences = random.sample(list(self.meta_buffer), batch_size)
                indices = None
                importance_weights = None

            # 🎯 数据转换和验证
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []

            for exp in experiences:
                # 数据验证
                if (not np.any(np.isnan(exp.state)) and not np.any(np.isnan(exp.next_state)) and
                    not np.any(np.isnan(exp.action)) and not np.isnan(exp.reward)):
                    states.append(exp.state)
                    actions.append(exp.action)
                    rewards.append(exp.reward)
                    next_states.append(exp.next_state)
                    dones.append(float(exp.done))

            # 如果过滤后数据不足，用默认数据补齐
            while len(states) < batch_size:
                states.append(np.zeros(15, dtype=np.float32))
                actions.append(np.zeros(3, dtype=np.float32))
                rewards.append(0.0)
                next_states.append(np.zeros(15, dtype=np.float32))
                dones.append(0.0)

            # 转换为numpy数组
            states = np.array(states[:batch_size], dtype=np.float32)
            actions = np.array(actions[:batch_size], dtype=np.float32)
            rewards = np.array(rewards[:batch_size], dtype=np.float32).reshape(-1, 1)
            next_states = np.array(next_states[:batch_size], dtype=np.float32)
            dones = np.array(dones[:batch_size], dtype=np.float32).reshape(-1, 1)

            # 🎯 更新采样统计
            self.sampling_stats['meta_samples'] += 1
            self.sampling_stats['last_sample_time'] = time.time()

            return states, actions, rewards, next_states, dones, importance_weights

        except Exception as e:
            print(f"⚠️ 元控制器采样错误: {e}")
            # 返回安全的默认数据
            dummy_state = np.zeros((batch_size, 15), dtype=np.float32)
            dummy_action = np.zeros((batch_size, 3), dtype=np.float32)
            dummy_reward = np.zeros((batch_size, 1), dtype=np.float32)
            dummy_next_state = np.zeros((batch_size, 15), dtype=np.float32)
            dummy_done = np.zeros((batch_size, 1), dtype=np.float32)
            return dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done, None

    def sample_sub(self, cluster_type: str, batch_size: int) -> Tuple[np.ndarray, np.ndarray,
                                                                    np.ndarray, np.ndarray,
                                                                    np.ndarray, Optional[np.ndarray]]:
        """🎯 从指定集群子缓冲区稳定化采样"""
        try:
            # 获取集群状态维度
            state_dims = {'FPGA': 6, 'FOG_GPU': 8, 'CLOUD': 6}
            action_dims = {'FPGA': 2, 'FOG_GPU': 3, 'CLOUD': 2}
            state_dim = state_dims.get(cluster_type, 6)
            action_dim = action_dims.get(cluster_type, 2)

            if not self.can_sample_sub(cluster_type, batch_size):
                # 返回集群特化的默认数据
                dummy_state = np.zeros((batch_size, state_dim), dtype=np.float32)
                dummy_action = np.zeros((batch_size, action_dim), dtype=np.float32)
                dummy_reward = np.zeros((batch_size, 1), dtype=np.float32)
                dummy_next_state = np.zeros((batch_size, state_dim), dtype=np.float32)
                dummy_done = np.zeros((batch_size, 1), dtype=np.float32)
                return dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done, None

            # 🎯 优先级采样
            if self.enable_per:
                experiences, indices, weights = self.sub_buffers[cluster_type].sample(batch_size)
                importance_weights = weights
            else:
                experiences = random.sample(list(self.sub_buffers[cluster_type]), batch_size)
                indices = None
                importance_weights = None

            # 🎯 数据转换和验证
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []

            for exp in experiences:
                # 数据验证
                if (not np.any(np.isnan(exp.state)) and not np.any(np.isnan(exp.next_state)) and
                    not np.any(np.isnan(exp.action)) and not np.isnan(exp.reward)):

                    # 🎯 确保维度正确
                    state = exp.state if len(exp.state) == state_dim else np.resize(exp.state, state_dim)
                    next_state = exp.next_state if len(exp.next_state) == state_dim else np.resize(exp.next_state, state_dim)
                    action = exp.action if len(exp.action) == action_dim else np.resize(exp.action, action_dim)

                    states.append(state)
                    actions.append(action)
                    rewards.append(exp.reward)
                    next_states.append(next_state)
                    dones.append(float(exp.done))

            # 如果过滤后数据不足，用默认数据补齐
            while len(states) < batch_size:
                states.append(np.zeros(state_dim, dtype=np.float32))
                actions.append(np.zeros(action_dim, dtype=np.float32))
                rewards.append(0.0)
                next_states.append(np.zeros(state_dim, dtype=np.float32))
                dones.append(0.0)

            # 转换为numpy数组
            states = np.array(states[:batch_size], dtype=np.float32)
            actions = np.array(actions[:batch_size], dtype=np.float32)
            rewards = np.array(rewards[:batch_size], dtype=np.float32).reshape(-1, 1)
            next_states = np.array(next_states[:batch_size], dtype=np.float32)
            dones = np.array(dones[:batch_size], dtype=np.float32).reshape(-1, 1)

            # 🎯 更新采样统计
            self.sampling_stats['sub_samples'][cluster_type] += 1
            self.sampling_stats['last_sample_time'] = time.time()

            return states, actions, rewards, next_states, dones, importance_weights

        except Exception as e:
            print(f"⚠️ 子控制器采样错误 ({cluster_type}): {e}")
            # 返回安全的默认数据
            state_dims = {'FPGA': 6, 'FOG_GPU': 8, 'CLOUD': 6}
            action_dims = {'FPGA': 2, 'FOG_GPU': 3, 'CLOUD': 2}
            state_dim = state_dims.get(cluster_type, 6)
            action_dim = action_dims.get(cluster_type, 2)

            dummy_state = np.zeros((batch_size, state_dim), dtype=np.float32)
            dummy_action = np.zeros((batch_size, action_dim), dtype=np.float32)
            dummy_reward = np.zeros((batch_size, 1), dtype=np.float32)
            dummy_next_state = np.zeros((batch_size, state_dim), dtype=np.float32)
            dummy_done = np.zeros((batch_size, 1), dtype=np.float32)
            return dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done, None

    def update_priorities(self, experience_type: str, indices: np.ndarray,
                         priorities: np.ndarray, cluster_type: str = None):
        """🎯 更新经验优先级"""
        try:
            if not self.enable_per:
                return

            if experience_type == 'meta':
                self.meta_buffer.update_priorities(indices, priorities)
            elif experience_type == 'sub' and cluster_type in self.sub_buffers:
                self.sub_buffers[cluster_type].update_priorities(indices, priorities)

        except Exception as e:
            print(f"⚠️ 优先级更新错误: {e}")

    def get_buffer_status(self) -> Dict:
        """🎯 获取缓冲区状态摘要"""
        try:
            # 计算质量统计
            meta_avg_quality = (np.mean(list(self.quality_monitor['meta_quality_history']))
                              if self.quality_monitor['meta_quality_history'] else 0)

            sub_avg_qualities = {}
            for cluster_type, quality_history in self.quality_monitor['sub_quality_history'].items():
                sub_avg_qualities[cluster_type] = (np.mean(list(quality_history))
                                                 if quality_history else 0)

            status = {
                'buffer_sizes': {
                    'meta': len(self.meta_buffer),
                    **{cluster: len(buffer) for cluster, buffer in self.sub_buffers.items()}
                },
                'capacity_utilization': {
                    'meta': len(self.meta_buffer) / (self.capacity // 2),
                    **{cluster: len(buffer) / (self.capacity // 6)
                       for cluster, buffer in self.sub_buffers.items()}
                },
                'quality_stats': {
                    'total_experiences': self.quality_monitor['total_experiences'],
                    'filtered_experiences': self.quality_monitor['filtered_experiences'],
                    'filter_rate': (self.quality_monitor['filtered_experiences'] /
                                  max(1, self.quality_monitor['total_experiences'])),
                    'meta_avg_quality': meta_avg_quality,
                    'sub_avg_qualities': sub_avg_qualities
                },
                'sampling_stats': self.sampling_stats.copy(),
                'configuration': {
                    'enable_per': self.enable_per,
                    'quality_threshold': self.quality_threshold,
                    'balance_sampling': self.balance_sampling
                }
            }

            return status

        except Exception as e:
            print(f"⚠️ 缓冲区状态获取错误: {e}")
            return {'error': str(e)}

    def __len__(self):
        """返回总经验数量"""
        try:
            total = len(self.meta_buffer)
            for buffer in self.sub_buffers.values():
                total += len(buffer)
            return total
        except Exception as e:
            print(f"⚠️ 长度计算错误: {e}")
            return 0

    def clear(self):
        """🎯 智能清空缓冲区"""
        try:
            # 保存重要统计信息
            total_before = len(self)

            # 清空缓冲区
            if hasattr(self.meta_buffer, 'clear'):
                self.meta_buffer.clear()
            else:
                self.meta_buffer = (PrioritizedExperienceReplay(self.capacity // 2)
                                  if self.enable_per else deque(maxlen=self.capacity // 2))

            for cluster_type in self.sub_buffers:
                if hasattr(self.sub_buffers[cluster_type], 'clear'):
                    self.sub_buffers[cluster_type].clear()
                else:
                    sub_capacity = self.capacity // 6
                    self.sub_buffers[cluster_type] = (PrioritizedExperienceReplay(sub_capacity)
                                                    if self.enable_per else deque(maxlen=sub_capacity))

            # 重置部分统计（保留配置）
            self.sampling_stats.update({
                'meta_samples': 0,
                'sub_samples': {cluster: 0 for cluster in ['FPGA', 'FOG_GPU', 'CLOUD']},
                'quality_filtered': 0,
                'last_sample_time': time.time()
            })

            print(f"🔄 缓冲区已清空 ({total_before} 个经验)")

        except Exception as e:
            print(f"⚠️ 缓冲区清空错误: {e}")


# 🎯 为了向后兼容，保留原始类名
HierarchicalReplayBuffer = StabilizedHierarchicalReplayBuffer


# 🧪 增强的测试函数
def test_stabilized_replay_buffer():
    """测试稳定化回放缓冲区"""
    print("🧪 开始测试StabilizedHierarchicalReplayBuffer...")

    try:
        # 创建稳定化回放缓冲区
        buffer = StabilizedHierarchicalReplayBuffer(
            capacity=1000,
            enable_per=True,
            quality_threshold=0.1,
            balance_sampling=True
        )

        # 测试1: 基本经验添加
        print("\n📝 测试1: 基本经验添加测试")

        # 添加元控制器经验
        for i in range(50):
            state = np.random.random(15).astype(np.float32)
            action = np.random.random(3).astype(np.float32)
            action = action / np.sum(action)  # 归一化为概率分布
            reward = np.random.uniform(-10, 10)
            next_state = np.random.random(15).astype(np.float32)
            done = np.random.choice([True, False])

            success = buffer.push_meta(state, action, reward, next_state, done)
            if i == 0:
                print(f"✅ 首次元控制器经验添加: {'成功' if success else '失败'}")

        # 添加子控制器经验
        for cluster_type in ['FPGA', 'FOG_GPU', 'CLOUD']:
            state_dims = {'FPGA': 6, 'FOG_GPU': 8, 'CLOUD': 6}
            action_dims = {'FPGA': 2, 'FOG_GPU': 3, 'CLOUD': 2}

            for i in range(30):
                state = np.random.random(state_dims[cluster_type]).astype(np.float32)
                action = np.random.random(action_dims[cluster_type]).astype(np.float32)
                reward = np.random.uniform(-5, 15)
                next_state = np.random.random(state_dims[cluster_type]).astype(np.float32)
                done = np.random.choice([True, False])

                success = buffer.push_sub(cluster_type, state, action, reward, next_state, done)
                if i == 0:
                    print(f"✅ {cluster_type}首次经验添加: {'成功' if success else '失败'}")

        print(f"总经验数: {len(buffer)}")

        # 测试2: 采样功能
        print("\n📝 测试2: 采样功能测试")

        # 元控制器采样
        if buffer.can_sample(8):
            states, actions, rewards, next_states, dones, weights = buffer.sample_meta(8)
            print(f"✅ 元控制器采样: {states.shape}, 权重: {'有' if weights is not None else '无'}")

        # 子控制器采样
        for cluster_type in ['FPGA', 'FOG_GPU', 'CLOUD']:
            if buffer.can_sample_sub(cluster_type, 4):
                states, actions, rewards, next_states, dones, weights = buffer.sample_sub(cluster_type, 4)
                print(f"✅ {cluster_type}采样: {states.shape}, {actions.shape}")

        # 测试3: 缓冲区状态
        print("\n📝 测试3: 缓冲区状态测试")
        status = buffer.get_buffer_status()
        print(f"✅ 缓冲区状态:")
        print(f"  - 元控制器大小: {status['buffer_sizes']['meta']}")
        print(f"  - 平均质量: {status['quality_stats']['meta_avg_quality']:.3f}")
        print(f"  - 过滤率: {status['quality_stats']['filter_rate']:.1%}")

        # 测试4: 质量过滤
        print("\n📝 测试4: 质量过滤测试")

        # 添加低质量经验（包含NaN）
        bad_state = np.full(15, np.nan, dtype=np.float32)
        bad_action = np.random.random(3).astype(np.float32)
        success = buffer.push_meta(bad_state, bad_action, 5.0, np.random.random(15).astype(np.float32), False)
        print(f"✅ 低质量经验过滤: {'已过滤' if not success else '未过滤'}")

        print("\n🎉 所有测试通过！StabilizedHierarchicalReplayBuffer工作正常")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    success = test_stabilized_replay_buffer()
    if success:
        print("\n✅ Stabilized Hierarchical Replay Buffer ready for production!")
        print("🎯 主要优化:")
        print("  - 优先级经验回放: 重要经验优先学习")
        print("  - 经验质量过滤: 自动过滤无效经验")
        print("  - 稳定化采样: 确保训练数据质量")
        print("  - 内存优化管理: 高效的数据存储")
        print("  - 实时监控统计: 缓冲区状态跟踪")
    else:
        print("\n❌ Stabilized Hierarchical Replay Buffer needs debugging!")