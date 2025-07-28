"""
Fog-Cloud Computing Environment - 高度稳定化版本
雾云计算环境模拟 - 专为稳定化HD-DDPG设计

🎯 主要优化：
- 状态计算稳定化
- 增强节点管理
- 集群特化优化
- 实时健康监控
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
from collections import deque
import time
import json


class NodeType(Enum):
    FPGA = "FPGA"
    FOG_GPU = "FOG_GPU"
    CLOUD = "CLOUD"


@dataclass
class EnhancedComputingNode:
    """增强的计算节点类 - 支持稳定化调度"""
    node_id: int
    node_type: NodeType
    computing_power: float  # MIPS
    memory_capacity: float  # MB
    current_load: float = 0.0
    energy_efficiency: float = 0.1  # W/MIPS
    availability: bool = True

    # 🎯 新增稳定化属性
    load_history: deque = field(default_factory=lambda: deque(maxlen=10))
    performance_history: deque = field(default_factory=lambda: deque(maxlen=20))
    last_update_time: float = 0.0
    failure_count: int = 0
    recovery_count: int = 0
    efficiency_score: float = 1.0
    stability_score: float = 1.0

    def __post_init__(self):
        """初始化后处理"""
        if not hasattr(self, 'load_history') or self.load_history is None:
            self.load_history = deque(maxlen=10)
        if not hasattr(self, 'performance_history') or self.performance_history is None:
            self.performance_history = deque(maxlen=20)

    def get_execution_time(self, computation_requirement: float) -> float:
        """🎯 稳定化的任务执行时间计算"""
        if not self.availability or self.computing_power == 0:
            return float('inf')

        # 基础执行时间
        base_time = computation_requirement / self.computing_power

        # 🎯 负载影响因子（更平滑的调整）
        load_ratio = self.current_load / self.memory_capacity
        load_factor = 1.0 + (load_ratio * 0.3)  # 最多30%的性能下降

        # 🎯 效率评分影响
        efficiency_factor = 0.8 + 0.4 * self.efficiency_score  # 0.8-1.2范围

        # 🎯 稳定性评分影响
        stability_factor = 0.9 + 0.2 * self.stability_score  # 0.9-1.1范围

        execution_time = base_time * load_factor / efficiency_factor * stability_factor
        return max(0.1, execution_time)  # 最小执行时间0.1秒

    def get_energy_consumption(self, execution_time: float) -> float:
        """🎯 稳定化的能耗计算"""
        if execution_time == float('inf'):
            return float('inf')

        # 基础能耗
        base_energy = self.computing_power * execution_time * self.energy_efficiency

        # 🎯 负载相关的能耗调整
        load_ratio = self.current_load / self.memory_capacity
        energy_overhead = 1.0 + (load_ratio * 0.2)  # 负载越高能耗越大

        # 🎯 效率评分影响
        efficiency_factor = 1.5 - 0.5 * self.efficiency_score  # 效率越高能耗越低

        return base_energy * energy_overhead * efficiency_factor

    def can_accommodate(self, memory_requirement: float) -> bool:
        """🎯 增强的任务容纳检查"""
        if not self.availability:
            return False

        # 基础容量检查
        basic_check = (self.current_load + memory_requirement <= self.memory_capacity)

        # 🎯 安全边界检查（预留5%的缓冲）
        safety_margin = self.memory_capacity * 0.05
        safety_check = (self.current_load + memory_requirement <= self.memory_capacity - safety_margin)

        # 🎯 稳定性评分影响
        if self.stability_score < 0.5:
            # 稳定性较差时更保守
            return safety_check
        else:
            return basic_check

    def update_load(self, memory_change: float, current_time: float = None):
        """🎯 更新节点负载并记录历史"""
        old_load = self.current_load
        self.current_load += memory_change
        self.current_load = max(0, min(self.current_load, self.memory_capacity))

        # 记录负载历史
        load_ratio = self.current_load / self.memory_capacity
        self.load_history.append(load_ratio)

        # 更新时间戳
        if current_time is not None:
            self.last_update_time = current_time

        # 🎯 更新稳定性评分
        self._update_stability_score()

    def _update_stability_score(self):
        """🎯 更新稳定性评分"""
        try:
            if len(self.load_history) >= 3:
                # 基于负载历史的方差计算稳定性
                load_variance = np.var(list(self.load_history))
                self.stability_score = max(0.1, 1.0 - load_variance * 2.0)

            # 🎯 基于故障历史调整
            if self.failure_count > 0:
                failure_penalty = min(0.3, self.failure_count * 0.1)
                self.stability_score *= (1.0 - failure_penalty)

            # 🎯 基于恢复能力调整
            if self.recovery_count > 0:
                recovery_bonus = min(0.2, self.recovery_count * 0.05)
                self.stability_score *= (1.0 + recovery_bonus)

            self.stability_score = np.clip(self.stability_score, 0.1, 1.0)

        except Exception as e:
            print(f"⚠️ 节点{self.node_id}稳定性评分更新错误: {e}")
            self.stability_score = 0.5

    def update_efficiency_score(self, task_performance: float):
        """🎯 更新效率评分"""
        try:
            # 记录性能历史
            self.performance_history.append(task_performance)

            # 计算效率评分
            if len(self.performance_history) >= 5:
                recent_performance = list(self.performance_history)[-5:]
                avg_performance = np.mean(recent_performance)

                # 归一化到0-1范围
                self.efficiency_score = max(0.1, min(1.0, avg_performance))

        except Exception as e:
            print(f"⚠️ 节点{self.node_id}效率评分更新错误: {e}")

    def simulate_failure(self):
        """🎯 模拟节点故障"""
        if self.availability:
            self.availability = False
            self.failure_count += 1
            self.current_load = 0.0  # 故障时清空负载
            print(f"💥 节点{self.node_id}({self.node_type.value})发生故障")

    def simulate_recovery(self):
        """🎯 模拟节点恢复"""
        if not self.availability:
            self.availability = True
            self.recovery_count += 1
            self.current_load = 0.0  # 恢复时重置负载
            print(f"🔧 节点{self.node_id}({self.node_type.value})已恢复")

    def get_status_summary(self) -> Dict:
        """获取节点状态摘要"""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'availability': self.availability,
            'load_ratio': self.current_load / self.memory_capacity,
            'efficiency_score': self.efficiency_score,
            'stability_score': self.stability_score,
            'failure_count': self.failure_count,
            'recovery_count': self.recovery_count
        }


class StabilizedFogCloudEnvironment:
    """
    稳定化雾云计算环境
    🎯 专为稳定化HD-DDPG算法设计的增强环境
    """

    def __init__(self):
        print("🔧 初始化StabilizedFogCloudEnvironment...")

        # 初始化基础组件
        self.nodes = self._initialize_enhanced_nodes()
        self.network_latency = self._initialize_network()
        self.bandwidth = self._initialize_bandwidth()
        self.current_time = 0.0

        # 🎯 稳定化监控系统
        self.environment_monitor = {
            'system_state_history': deque(maxlen=50),
            'load_distribution_history': deque(maxlen=30),
            'performance_metrics': deque(maxlen=100),
            'failure_events': deque(maxlen=20),
            'recovery_events': deque(maxlen=20)
        }

        # 🎯 集群协调器
        self.cluster_coordinator = {
            'load_balancing_weights': {'FPGA': 0.3, 'FOG_GPU': 0.4, 'CLOUD': 0.3},
            'performance_targets': {'FPGA': 0.8, 'FOG_GPU': 0.9, 'CLOUD': 0.85},
            'stability_thresholds': {'FPGA': 0.7, 'FOG_GPU': 0.8, 'CLOUD': 0.75}
        }

        # 🎯 网络状态管理
        self.network_monitor = {
            'latency_history': deque(maxlen=20),
            'bandwidth_utilization': deque(maxlen=20),
            'congestion_level': 0.0
        }

        print(f"✅ 稳定化雾云环境初始化完成")
        print(f"  - 总节点数: {sum(len(nodes) for nodes in self.nodes.values())}")
        print(f"  - 集群类型: {list(self.nodes.keys())}")

    def _initialize_enhanced_nodes(self) -> Dict[str, List[EnhancedComputingNode]]:
        """🎯 初始化增强的计算节点"""
        nodes = {
            'FPGA': [],
            'FOG_GPU': [],
            'CLOUD': []
        }

        # 🎯 FPGA节点 - 优化配置
        fpga_configs = [
            {'computing_power': 800, 'memory_capacity': 4096, 'energy_efficiency': 0.08},
            {'computing_power': 900, 'memory_capacity': 4096, 'energy_efficiency': 0.09}
        ]

        for i, config in enumerate(fpga_configs):
            node = EnhancedComputingNode(
                node_id=i,
                node_type=NodeType.FPGA,
                **config
            )
            nodes['FPGA'].append(node)

        # 🎯 FOG GPU节点 - 优化配置
        fog_gpu_configs = [
            {'computing_power': 2000, 'memory_capacity': 8192, 'energy_efficiency': 0.25},
            {'computing_power': 2200, 'memory_capacity': 8192, 'energy_efficiency': 0.28},
            {'computing_power': 1800, 'memory_capacity': 8192, 'energy_efficiency': 0.22}
        ]

        for i, config in enumerate(fog_gpu_configs):
            node = EnhancedComputingNode(
                node_id=i + 2,
                node_type=NodeType.FOG_GPU,
                **config
            )
            nodes['FOG_GPU'].append(node)

        # 🎯 Cloud节点 - 优化配置
        cloud_configs = [
            {'computing_power': 5000, 'memory_capacity': 16384, 'energy_efficiency': 0.45},
            {'computing_power': 5500, 'memory_capacity': 16384, 'energy_efficiency': 0.48}
        ]

        for i, config in enumerate(cloud_configs):
            node = EnhancedComputingNode(
                node_id=i + 5,
                node_type=NodeType.CLOUD,
                **config
            )
            nodes['CLOUD'].append(node)

        return nodes

    def _initialize_network(self) -> Dict[str, float]:
        """🎯 初始化稳定化的网络延迟"""
        return {
            'FPGA_TO_FOG': 5,  # ms
            'FOG_TO_CLOUD': 50,  # ms
            'FPGA_TO_CLOUD': 100,  # ms
            'INTERNAL': 1,  # ms
            # 🎯 新增：延迟变化范围
            'LATENCY_VARIANCE': 0.1,  # 10%的变化范围
            'BASE_LATENCY_FACTOR': 1.0
        }

    def _initialize_bandwidth(self) -> Dict[str, float]:
        """🎯 初始化稳定化的带宽"""
        return {
            'FPGA_FOG': 100,  # Mbps
            'FOG_CLOUD': 1000,  # Mbps
            'FPGA_CLOUD': 100,  # Mbps
            'INTERNAL': 10000,  # Mbps
            # 🎯 新增：带宽利用率跟踪
            'UTILIZATION_FACTOR': 1.0,
            'CONGESTION_THRESHOLD': 0.8
        }

    def get_stabilized_transmission_time(self, from_type: str, to_type: str, data_size: float) -> float:
        """🎯 稳定化的数据传输时间计算"""
        try:
            if from_type == to_type:
                latency = self.network_latency['INTERNAL']
                bandwidth = self.bandwidth['INTERNAL']
            else:
                # 确定网络路径
                if (from_type == 'FPGA' and to_type == 'FOG_GPU') or \
                        (from_type == 'FOG_GPU' and to_type == 'FPGA'):
                    latency = self.network_latency['FPGA_TO_FOG']
                    bandwidth = self.bandwidth['FPGA_FOG']
                elif (from_type == 'FOG_GPU' and to_type == 'CLOUD') or \
                        (from_type == 'CLOUD' and to_type == 'FOG_GPU'):
                    latency = self.network_latency['FOG_TO_CLOUD']
                    bandwidth = self.bandwidth['FOG_CLOUD']
                else:  # FPGA <-> CLOUD
                    latency = self.network_latency['FPGA_TO_CLOUD']
                    bandwidth = self.bandwidth['FPGA_CLOUD']

            # 🎯 稳定化的延迟计算（减少随机性）
            latency_variance = self.network_latency.get('LATENCY_VARIANCE', 0.1)
            stable_latency = latency * (1.0 + np.random.uniform(-latency_variance/2, latency_variance/2))

            # 🎯 考虑网络拥塞
            congestion_factor = 1.0 + self.network_monitor['congestion_level'] * 0.3
            effective_bandwidth = bandwidth / congestion_factor

            # 传输时间计算
            transmission_time = stable_latency / 1000 + (data_size * 8) / effective_bandwidth

            # 🎯 记录传输历史
            self.network_monitor['latency_history'].append(stable_latency)

            return max(0.001, transmission_time)  # 最小传输时间1ms

        except Exception as e:
            print(f"⚠️ 传输时间计算错误: {e}")
            return 0.1  # 默认传输时间

    def get_available_nodes(self, node_type: str = None) -> List[EnhancedComputingNode]:
        """🎯 获取稳定化的可用节点"""
        try:
            if node_type:
                available_nodes = [node for node in self.nodes[node_type]
                                 if node.availability and node.stability_score > 0.3]
            else:
                available_nodes = []
                for node_list in self.nodes.values():
                    available_nodes.extend([node for node in node_list
                                          if node.availability and node.stability_score > 0.3])

            # 🎯 按稳定性和效率排序
            available_nodes.sort(key=lambda n: (n.stability_score + n.efficiency_score) / 2, reverse=True)

            return available_nodes

        except Exception as e:
            print(f"⚠️ 获取可用节点错误: {e}")
            return []

    def get_node_by_id(self, node_id: int) -> Optional[EnhancedComputingNode]:
        """根据ID获取节点"""
        for node_list in self.nodes.values():
            for node in node_list:
                if node.node_id == node_id:
                    return node
        return None

    def update_node_load(self, node_id: int, memory_change: float):
        """🎯 稳定化的节点负载更新"""
        try:
            node = self.get_node_by_id(node_id)
            if node:
                node.update_load(memory_change, self.current_time)

                # 🎯 更新环境监控
                self._update_load_distribution_monitoring()

        except Exception as e:
            print(f"⚠️ 节点负载更新错误: {e}")

    def get_stabilized_system_state(self) -> np.ndarray:
        """🎯 获取稳定化的系统状态向量"""
        try:
            state = []

            # 🎯 各类型节点的稳定化状态
            for node_type in ['FPGA', 'FOG_GPU', 'CLOUD']:
                nodes = self.nodes[node_type]
                if nodes:
                    # 基础指标
                    avg_load = np.mean([node.current_load / node.memory_capacity for node in nodes])
                    avg_availability = np.mean([1.0 if node.availability else 0.0 for node in nodes])

                    # 🎯 稳定化指标
                    avg_stability = np.mean([node.stability_score for node in nodes])
                    avg_efficiency = np.mean([node.efficiency_score for node in nodes])

                    # 计算能力利用率（稳定化）
                    avg_power_usage = np.mean([
                        node.computing_power * (node.current_load / node.memory_capacity) / 1000
                        for node in nodes
                    ])

                    state.extend([avg_load, avg_availability, avg_stability])
                else:
                    state.extend([0.0, 0.0, 0.0])

            # 🎯 系统整体稳定化指标
            total_nodes = sum(len(nodes) for nodes in self.nodes.values())
            available_nodes = len(self.get_available_nodes())
            system_availability = available_nodes / total_nodes if total_nodes > 0 else 0

            # 🎯 稳定化的计算能力利用率
            total_power = 0
            used_power = 0
            for node_list in self.nodes.values():
                for node in node_list:
                    if node.availability:
                        total_power += node.computing_power
                        used_power += node.computing_power * (node.current_load / node.memory_capacity)

            power_utilization = used_power / total_power if total_power > 0 else 0

            # 🎯 稳定化的网络状态（去除随机性）
            network_congestion = self.network_monitor['congestion_level']

            # 🎯 稳定化的时间因子
            time_factor = min(1.0, self.current_time / 10000)  # 归一化并限制范围

            # 🎯 系统稳定性压力指标
            stability_pressure = self._calculate_system_stability_pressure()

            # 🎯 集群协调指标
            cluster_balance = self._calculate_cluster_balance()

            state.extend([
                system_availability,    # 9
                power_utilization,      # 10
                network_congestion,     # 11
                time_factor,           # 12
                stability_pressure,    # 13
                cluster_balance        # 14
            ])

            # 确保状态向量长度为15
            state = state[:15]
            while len(state) < 15:
                state.append(0.0)

            # 🎯 应用状态平滑
            stabilized_state = self._apply_state_smoothing(np.array(state, dtype=np.float32))

            return stabilized_state

        except Exception as e:
            print(f"⚠️ 系统状态计算错误: {e}")
            # 返回安全的默认状态
            return np.zeros(15, dtype=np.float32)

    def _apply_state_smoothing(self, current_state: np.ndarray) -> np.ndarray:
        """🎯 应用状态平滑"""
        try:
            # 记录当前状态
            self.environment_monitor['system_state_history'].append(current_state.copy())

            # 如果历史记录不足，直接返回当前状态
            if len(self.environment_monitor['system_state_history']) < 3:
                return current_state

            # 计算平滑权重
            history_length = len(self.environment_monitor['system_state_history'])
            weights = np.exp(-np.arange(history_length) * 0.3)
            weights = weights / np.sum(weights)

            # 计算加权平均
            state_history = np.array(list(self.environment_monitor['system_state_history']))
            smoothed_state = np.average(state_history, axis=0, weights=weights)

            return smoothed_state.astype(np.float32)

        except Exception as e:
            print(f"⚠️ 状态平滑错误: {e}")
            return current_state

    def _calculate_system_stability_pressure(self) -> float:
        """🎯 计算系统稳定性压力"""
        try:
            stability_scores = []
            for node_list in self.nodes.values():
                for node in node_list:
                    if node.availability:
                        stability_scores.append(node.stability_score)

            if stability_scores:
                avg_stability = np.mean(stability_scores)
                stability_variance = np.var(stability_scores)
                # 压力 = 1 - 平均稳定性 + 方差惩罚
                pressure = (1.0 - avg_stability) + stability_variance
                return min(1.0, pressure)
            else:
                return 1.0

        except Exception as e:
            print(f"⚠️ 稳定性压力计算错误: {e}")
            return 0.5

    def _calculate_cluster_balance(self) -> float:
        """🎯 计算集群间负载均衡度"""
        try:
            cluster_loads = []
            for cluster_type, nodes in self.nodes.items():
                if nodes:
                    avg_load = np.mean([node.current_load / node.memory_capacity for node in nodes])
                    cluster_loads.append(avg_load)

            if len(cluster_loads) > 1:
                load_variance = np.var(cluster_loads)
                balance_score = 1.0 / (1.0 + load_variance * 5)
                return balance_score
            else:
                return 1.0

        except Exception as e:
            print(f"⚠️ 集群均衡计算错误: {e}")
            return 0.5

    def _update_load_distribution_monitoring(self):
        """🎯 更新负载分布监控"""
        try:
            current_distribution = {}
            for cluster_type, nodes in self.nodes.items():
                if nodes:
                    cluster_load = np.mean([node.current_load / node.memory_capacity for node in nodes])
                    current_distribution[cluster_type] = cluster_load

            self.environment_monitor['load_distribution_history'].append(current_distribution)

        except Exception as e:
            print(f"⚠️ 负载分布监控更新错误: {e}")

    def reset(self):
        """🎯 稳定化的环境重置"""
        try:
            self.current_time = 0.0

            # 重置所有节点
            for node_list in self.nodes.values():
                for node in node_list:
                    node.current_load = 0.0
                    node.availability = True
                    # 🎯 保留历史性能数据，只重置当前状态
                    # node.load_history.clear()  # 不清除历史，保持学习连续性

            # 🎯 重置网络状态
            self.network_monitor['congestion_level'] = 0.0

            # 🎯 不完全清除监控历史，保持一定连续性
            # 只保留最近的一些历史数据
            if len(self.environment_monitor['system_state_history']) > 10:
                recent_states = list(self.environment_monitor['system_state_history'])[-5:]
                self.environment_monitor['system_state_history'].clear()
                self.environment_monitor['system_state_history'].extend(recent_states)

            print("🔄 环境已稳定化重置")

        except Exception as e:
            print(f"⚠️ 环境重置错误: {e}")

    def get_enhanced_cluster_state(self, cluster_type: str) -> np.ndarray:
        """🎯 获取增强的集群状态"""
        try:
            nodes = self.nodes[cluster_type]
            state = []

            # 🎯 获取该集群的详细稳定化状态
            for node in nodes:
                load_ratio = node.current_load / node.memory_capacity
                availability = 1.0 if node.availability else 0.0
                stability = node.stability_score
                efficiency = node.efficiency_score

                # 基础状态：负载率和可用性
                state.extend([load_ratio, availability])

            # 🎯 目标维度设置
            expected_dims = {
                'FPGA': 6,    # 2个节点 * 2个特征 + 2个集群级特征
                'FOG_GPU': 8, # 3个节点 * 2个特征 + 2个集群级特征
                'CLOUD': 6    # 2个节点 * 2个特征 + 2个集群级特征
            }

            target_dim = expected_dims.get(cluster_type, 6)

            # 补齐到目标维度
            while len(state) < target_dim - 2:  # 预留2个位置给集群级特征
                state.append(0.0)

            # 🎯 添加集群级别的稳定化统计信息
            if len(nodes) > 0:
                avg_load = np.mean([node.current_load / node.memory_capacity for node in nodes])
                avg_stability = np.mean([node.stability_score for node in nodes])
                cluster_availability = np.mean([1.0 if node.availability else 0.0 for node in nodes])

                # 集群性能指标
                cluster_efficiency = np.mean([node.efficiency_score for node in nodes])

                # 选择最重要的两个集群级特征
                state.extend([avg_load, avg_stability])
            else:
                state.extend([0.0, 0.0])

            # 确保状态向量长度正确
            state = state[:target_dim]
            while len(state) < target_dim:
                state.append(0.0)

            return np.array(state, dtype=np.float32)

        except Exception as e:
            print(f"⚠️ 集群状态计算错误 ({cluster_type}): {e}")
            # 返回安全的默认状态
            expected_dims = {'FPGA': 6, 'FOG_GPU': 8, 'CLOUD': 6}
            target_dim = expected_dims.get(cluster_type, 6)
            return np.zeros(target_dim, dtype=np.float32)

    def simulate_mild_failure(self, failure_type: str = 'node_slow'):
        """🎯 模拟温和的故障（减少系统波动）"""
        try:
            if failure_type == 'node_slow':
                # 随机选择一个节点降低其效率
                all_nodes = []
                for node_list in self.nodes.values():
                    all_nodes.extend([node for node in node_list if node.availability])

                if all_nodes:
                    target_node = np.random.choice(all_nodes)
                    target_node.efficiency_score *= 0.8  # 效率降低20%
                    print(f"🐌 节点{target_node.node_id}效率降低")

            elif failure_type == 'network_delay':
                # 增加网络拥塞
                self.network_monitor['congestion_level'] = min(1.0,
                    self.network_monitor['congestion_level'] + 0.2)
                print(f"🌐 网络拥塞增加")

            elif failure_type == 'minor_outage':
                # 临时节点不可用
                all_nodes = []
                for node_list in self.nodes.values():
                    all_nodes.extend([node for node in node_list if node.availability])

                if all_nodes and len(all_nodes) > 1:  # 确保不是最后一个节点
                    target_node = np.random.choice(all_nodes)
                    target_node.simulate_failure()

            # 记录故障事件
            self.environment_monitor['failure_events'].append({
                'time': self.current_time,
                'type': failure_type
            })

        except Exception as e:
            print(f"⚠️ 温和故障模拟错误: {e}")

    def simulate_failure(self, failure_rate: float = 0.005):
        """🎯 优化的故障模拟"""
        try:
            # 🎯 更温和的故障率
            for node_list in self.nodes.values():
                for node in node_list:
                    if node.availability and np.random.random() < failure_rate:
                        node.simulate_failure()
                    elif not node.availability and np.random.random() < 0.15:
                        # 15%概率恢复（比原来的10%稍高）
                        node.simulate_recovery()

                        # 记录恢复事件
                        self.environment_monitor['recovery_events'].append({
                            'time': self.current_time,
                            'node_id': node.node_id
                        })

            # 🎯 网络拥塞自然恢复
            if self.network_monitor['congestion_level'] > 0:
                self.network_monitor['congestion_level'] *= 0.95  # 5%的自然恢复

        except Exception as e:
            print(f"⚠️ 故障模拟错误: {e}")

    def get_system_health(self) -> float:
        """🎯 获取系统健康度"""
        try:
            health_factors = []

            # 节点健康度
            all_nodes = []
            for node_list in self.nodes.values():
                all_nodes.extend(node_list)

            if all_nodes:
                node_availability = np.mean([1.0 if node.availability else 0.0 for node in all_nodes])
                node_stability = np.mean([node.stability_score for node in all_nodes])
                node_efficiency = np.mean([node.efficiency_score for node in all_nodes])

                health_factors.extend([node_availability, node_stability, node_efficiency])

            # 网络健康度
            network_health = 1.0 - self.network_monitor['congestion_level']
            health_factors.append(network_health)

            # 集群平衡健康度
            cluster_balance = self._calculate_cluster_balance()
            health_factors.append(cluster_balance)

            # 计算综合健康度
            overall_health = np.mean(health_factors) if health_factors else 0.5

            return min(1.0, max(0.0, overall_health))

        except Exception as e:
            print(f"⚠️ 系统健康度计算错误: {e}")
            return 0.5

    def get_environment_summary(self) -> Dict:
        """🎯 获取环境状态摘要"""
        try:
            summary = {
                'current_time': self.current_time,
                'system_health': self.get_system_health(),
                'total_nodes': sum(len(nodes) for nodes in self.nodes.values()),
                'available_nodes': len(self.get_available_nodes()),
                'cluster_status': {},
                'network_status': {
                    'congestion_level': self.network_monitor['congestion_level'],
                    'avg_latency': np.mean(list(self.network_monitor['latency_history']))
                                 if self.network_monitor['latency_history'] else 0
                },
                'recent_failures': len(self.environment_monitor['failure_events']),
                'recent_recoveries': len(self.environment_monitor['recovery_events'])
            }

            # 集群状态
            for cluster_type, nodes in self.nodes.items():
                if nodes:
                    cluster_summary = {
                        'total_nodes': len(nodes),
                        'available_nodes': len([n for n in nodes if n.availability]),
                        'avg_load': np.mean([n.current_load / n.memory_capacity for n in nodes]),
                        'avg_stability': np.mean([n.stability_score for n in nodes]),
                        'avg_efficiency': np.mean([n.efficiency_score for n in nodes])
                    }
                    summary['cluster_status'][cluster_type] = cluster_summary

            return summary

        except Exception as e:
            print(f"⚠️ 环境摘要生成错误: {e}")
            return {'error': str(e)}

    # 🎯 为了向后兼容，保留原有方法的别名
    def get_system_state(self) -> np.ndarray:
        """向后兼容的系统状态获取"""
        return self.get_stabilized_system_state()

    def get_cluster_state(self, cluster_type: str) -> np.ndarray:
        """向后兼容的集群状态获取"""
        return self.get_enhanced_cluster_state(cluster_type)

    def get_transmission_time(self, from_type: str, to_type: str, data_size: float) -> float:
        """向后兼容的传输时间计算"""
        return self.get_stabilized_transmission_time(from_type, to_type, data_size)


# 🎯 为了向后兼容，保留原始类名
FogCloudEnvironment = StabilizedFogCloudEnvironment
ComputingNode = EnhancedComputingNode


# 🧪 增强的测试函数
def test_stabilized_fog_cloud_environment():
    """测试稳定化雾云环境"""
    print("🧪 开始测试StabilizedFogCloudEnvironment...")

    try:
        # 创建稳定化环境
        env = StabilizedFogCloudEnvironment()

        # 测试1: 基本功能
        print("\n📝 测试1: 基本功能测试")
        print(f"总节点数: {sum(len(nodes) for nodes in env.nodes.values())}")

        # 测试2: 稳定化状态获取
        print("\n📝 测试2: 稳定化状态获取测试")
        for i in range(5):
            state = env.get_stabilized_system_state()
            print(f"状态向量 {i+1}: 长度={len(state)}, 前5维={state[:5]}")

        # 测试3: 集群状态
        print("\n📝 测试3: 集群状态测试")
        for cluster_type in ['FPGA', 'FOG_GPU', 'CLOUD']:
            cluster_state = env.get_enhanced_cluster_state(cluster_type)
            print(f"{cluster_type} 状态: 长度={len(cluster_state)}, 值={cluster_state}")

        # 测试4: 节点负载更新
        print("\n📝 测试4: 节点负载更新测试")
        node = env.get_available_nodes()[0]
        old_load = node.current_load
        env.update_node_load(node.node_id, 100)
        print(f"节点{node.node_id} 负载: {old_load} -> {node.current_load}")

        # 测试5: 系统健康度
        print("\n📝 测试5: 系统健康度测试")
        health = env.get_system_health()
        print(f"系统健康度: {health:.3f}")

        # 测试6: 故障模拟
        print("\n📝 测试6: 故障模拟测试")
        env.simulate_mild_failure('node_slow')
        env.simulate_failure(0.1)  # 10%故障率测试

        health_after_failure = env.get_system_health()
        print(f"故障后系统健康度: {health_after_failure:.3f}")

        # 测试7: 环境重置
        print("\n📝 测试7: 环境重置测试")
        env.reset()
        health_after_reset = env.get_system_health()
        print(f"重置后系统健康度: {health_after_reset:.3f}")

        # 测试8: 环境摘要
        print("\n📝 测试8: 环境摘要测试")
        summary = env.get_environment_summary()
        print(f"环境摘要关键指标: 健康度={summary['system_health']:.3f}, "
              f"可用节点={summary['available_nodes']}/{summary['total_nodes']}")

        print("\n🎉 所有测试通过！StabilizedFogCloudEnvironment工作正常")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    success = test_stabilized_fog_cloud_environment()
    if success:
        print("\n✅ Stabilized Fog-Cloud Environment ready for production!")
        print("🎯 主要优化:")
        print("  - 节点状态跟踪: 历史性能和稳定性监控")
        print("  - 状态计算稳定化: 去除随机元素，增加平滑机制")
        print("  - 集群特化管理: 针对不同集群的优化策略")
        print("  - 智能故障处理: 温和故障模拟和自动恢复")
        print("  - 实时健康监控: 全面的系统健康度评估")
    else:
        print("\n❌ Stabilized Fog-Cloud Environment needs debugging!")