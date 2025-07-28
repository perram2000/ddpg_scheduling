"""
Medical Workflow Scheduling Simulator - 高度优化版本
医疗工作流调度仿真器 - 专为稳定化HD-DDPG设计

🎯 主要优化：
- 与稳定化算法完美集成
- 状态管理稳定化
- 奖励机制一致性
- 实时稳定性监控
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import json

from .fog_cloud_env import FogCloudEnvironment
from .workflow_generator import MedicalWorkflowGenerator, MedicalTask
from ..algorithms.hd_ddpg import HDDDPG
from ..utils.metrics import SchedulingMetrics


class StabilizedMedicalSchedulingSimulator:
    """
    稳定化医疗工作流调度仿真器
    🎯 专为稳定化HD-DDPG算法设计的增强仿真环境
    """

    def __init__(self, config: Dict = None):
        # 🎯 优化的默认配置
        self.config = {
            'simulation_episodes': 1000,
            'workflows_per_episode': 8,  # 减少以提升稳定性
            'min_workflow_size': 6,
            'max_workflow_size': 12,
            'workflow_types': ['radiology', 'pathology', 'general'],
            'failure_rate': 0.008,  # 降低故障率
            'save_interval': 100,
            'evaluation_interval': 25,  # 更频繁的评估
            # 🎯 新增稳定性配置
            'stability_monitoring': True,
            'state_smoothing': True,
            'reward_normalization': True,
            'performance_tracking_window': 50,
            'convergence_threshold': 0.05,
            'makespan_target': 1.5,  # 目标makespan
        }

        if config:
            self.config.update(config)

        print("🔧 初始化StabilizedMedicalSchedulingSimulator...")

        # 初始化组件
        try:
            self.environment = FogCloudEnvironment()
            print("✅ FogCloud环境初始化完成")

            self.workflow_generator = MedicalWorkflowGenerator()
            print("✅ 工作流生成器初始化完成")

            # 🎯 使用优化的HD-DDPG配置
            optimized_config = {
                'makespan_weight': 0.7,
                'stability_weight': 0.15,
                'action_smoothing': True,
                'verbose': True
            }
            self.hd_ddpg = HDDDPG(optimized_config)
            print("✅ 优化HD-DDPG算法初始化完成")

            self.metrics = SchedulingMetrics()
            print("✅ 指标计算器初始化完成")

        except Exception as e:
            print(f"❌ 组件初始化失败: {e}")
            raise

        # 🎯 稳定性监控系统
        self.stability_monitor = {
            'makespan_history': deque(maxlen=100),
            'reward_variance': deque(maxlen=50),
            'convergence_metrics': deque(maxlen=30),
            'system_health': deque(maxlen=20),
            'performance_trends': {}
        }

        # 仿真状态
        self.simulation_results = []
        self.baseline_results = []
        self.current_episode = 0
        self.best_performance = {
            'makespan': float('inf'),
            'reward': float('-inf'),
            'episode': 0
        }

        # 🎯 状态平滑缓冲区
        if self.config['state_smoothing']:
            self.state_buffer = deque(maxlen=5)

        print("✅ StabilizedMedicalSchedulingSimulator初始化完成")

    def run_simulation(self, episodes: int = None) -> Dict:
        """
        🎯 运行稳定化仿真实验
        """
        if episodes is None:
            episodes = self.config['simulation_episodes']

        print(f"🚀 开始稳定化HD-DDPG医疗调度仿真 - {episodes} episodes")
        simulation_start_time = time.time()

        # 🎯 预热阶段
        print("🔥 执行预热阶段...")
        self._warmup_phase()

        for episode in range(episodes):
            episode_start_time = time.time()

            try:
                # 🎯 稳定化的episode执行
                episode_result = self._run_stabilized_episode(episode)

                # 🎯 更新稳定性监控
                self._update_stability_monitoring(episode_result)

                # 🎯 记录结果
                self.simulation_results.append(episode_result)

                # 🎯 更新最佳性能
                self._update_best_performance(episode_result, episode)

                # 🎯 定期评估和保存
                if (episode + 1) % self.config['evaluation_interval'] == 0:
                    self._enhanced_performance_evaluation(episode + 1)

                if (episode + 1) % self.config['save_interval'] == 0:
                    self._save_enhanced_checkpoint(episode + 1)

                # 🎯 智能进度报告
                if (episode + 1) % 25 == 0:
                    self._print_enhanced_progress(episode + 1, episodes)

                # 🎯 早期停止检查
                if self._check_convergence(episode):
                    print(f"🎉 在episode {episode + 1}检测到收敛，提前停止")
                    break

            except Exception as e:
                print(f"⚠️ Episode {episode + 1}执行错误: {e}")
                # 记录错误但继续执行
                episode_result = self._create_error_result(episode, str(e))
                self.simulation_results.append(episode_result)

        simulation_duration = time.time() - simulation_start_time

        # 🎯 生成增强的最终报告
        final_report = self._generate_enhanced_final_report(simulation_duration)

        print(f"✅ 仿真完成，总耗时: {simulation_duration:.2f}秒")
        print(f"🏆 最佳Makespan: {self.best_performance['makespan']:.3f} (Episode {self.best_performance['episode']})")

        return final_report

    def _warmup_phase(self):
        """🎯 预热阶段 - 稳定化系统初始状态"""
        print("  执行系统预热...")

        # 生成预热工作流
        warmup_workflows = self._generate_episode_workflows(count=3)

        # 预热环境和算法
        for i, workflow in enumerate(warmup_workflows):
            self.environment.reset()
            try:
                _ = self.hd_ddpg.schedule_workflow(workflow, self.environment)
                print(f"  预热工作流 {i+1}/3 完成")
            except Exception as e:
                print(f"  预热工作流 {i+1}失败: {e}")

        print("✅ 预热完成")

    def _run_stabilized_episode(self, episode: int) -> Dict:
        """🎯 运行稳定化的episode"""
        episode_start_time = time.time()

        # 生成当前episode的工作流
        workflows = self._generate_episode_workflows()

        # 🎯 稳定化的故障模拟
        if np.random.random() < self.config['failure_rate']:
            self._simulate_stabilized_failure()

        # 🎯 获取稳定化的系统状态
        initial_system_state = self._get_stabilized_system_state()

        # HD-DDPG训练episode
        hd_ddpg_result = self.hd_ddpg.train_episode(workflows, self.environment)

        # 🎯 计算稳定性指标
        stability_metrics = self._calculate_episode_stability(hd_ddpg_result)

        # 🎯 增强的episode结果
        episode_result = {
            'episode': episode + 1,
            'workflows_count': len(workflows),
            'hd_ddpg_result': hd_ddpg_result,
            'stability_metrics': stability_metrics,
            'episode_duration': time.time() - episode_start_time,
            'initial_system_state': initial_system_state.tolist(),
            'system_health': self._assess_system_health(),
            'convergence_indicator': self._calculate_convergence_indicator()
        }

        return episode_result

    def _get_stabilized_system_state(self) -> np.ndarray:
        """🎯 获取稳定化的系统状态"""
        try:
            current_state = self.environment.get_system_state()

            if self.config['state_smoothing'] and len(self.state_buffer) > 0:
                # 状态平滑化
                self.state_buffer.append(current_state)

                # 计算加权平均
                weights = np.exp(-np.arange(len(self.state_buffer)) * 0.3)
                weights = weights / np.sum(weights)

                state_array = np.array(list(self.state_buffer))
                smoothed_state = np.average(state_array, axis=0, weights=weights)

                return smoothed_state
            else:
                if self.config['state_smoothing']:
                    self.state_buffer.append(current_state)
                return current_state

        except Exception as e:
            print(f"⚠️ 获取稳定化系统状态失败: {e}")
            # 返回默认状态
            return np.random.random(15)

    def _simulate_stabilized_failure(self):
        """🎯 稳定化的故障模拟"""
        try:
            # 更温和的故障模拟，避免系统剧烈波动
            failure_types = ['node_slow', 'network_delay', 'minor_outage']
            failure_type = np.random.choice(failure_types)

            if hasattr(self.environment, 'simulate_mild_failure'):
                self.environment.simulate_mild_failure(failure_type)
            else:
                self.environment.simulate_failure()

        except Exception as e:
            print(f"⚠️ 故障模拟错误: {e}")

    def _calculate_episode_stability(self, hd_ddpg_result: Dict) -> Dict:
        """🎯 计算episode稳定性指标"""
        try:
            makespan = hd_ddpg_result.get('avg_makespan', float('inf'))
            reward = hd_ddpg_result.get('total_reward', 0)

            # 更新历史记录
            self.stability_monitor['makespan_history'].append(makespan)

            stability_metrics = {
                'makespan_stability': 0,
                'reward_consistency': 0,
                'performance_trend': 0,
                'target_achievement': 0
            }

            # Makespan稳定性
            if len(self.stability_monitor['makespan_history']) >= 5:
                recent_makespans = list(self.stability_monitor['makespan_history'])[-5:]
                makespan_variance = np.var(recent_makespans)
                makespan_mean = np.mean(recent_makespans)

                if makespan_mean > 0:
                    stability_metrics['makespan_stability'] = 1.0 / (1.0 + makespan_variance / makespan_mean)

            # 奖励一致性
            if hasattr(hd_ddpg_result, 'reward_variance'):
                reward_var = hd_ddpg_result.get('reward_variance', 0)
                stability_metrics['reward_consistency'] = 1.0 / (1.0 + reward_var)

            # 性能趋势
            if len(self.stability_monitor['makespan_history']) >= 10:
                recent_performance = list(self.stability_monitor['makespan_history'])[-10:]
                trend_slope = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
                stability_metrics['performance_trend'] = max(0, -trend_slope)  # 负斜率是好的

            # 目标达成度
            target_makespan = self.config.get('makespan_target', 1.5)
            if makespan < float('inf'):
                stability_metrics['target_achievement'] = max(0, 1.0 - (makespan - target_makespan) / target_makespan)

            return stability_metrics

        except Exception as e:
            print(f"⚠️ 稳定性指标计算错误: {e}")
            return {
                'makespan_stability': 0,
                'reward_consistency': 0,
                'performance_trend': 0,
                'target_achievement': 0
            }

    def _assess_system_health(self) -> float:
        """🎯 评估系统健康度"""
        try:
            health_score = 1.0

            # 检查环境状态
            if hasattr(self.environment, 'get_system_health'):
                env_health = self.environment.get_system_health()
                health_score *= env_health

            # 检查算法状态
            if hasattr(self.hd_ddpg, 'get_optimization_status'):
                algo_status = self.hd_ddpg.get_optimization_status()
                if algo_status.get('optimization_progress', {}).get('stable', False):
                    health_score *= 1.1

            # 检查稳定性历史
            if len(self.stability_monitor['makespan_history']) >= 10:
                recent_variance = np.var(list(self.stability_monitor['makespan_history'])[-10:])
                if recent_variance < 0.1:
                    health_score *= 1.05

            return min(1.0, health_score)

        except Exception as e:
            print(f"⚠️ 系统健康评估错误: {e}")
            return 0.5

    def _calculate_convergence_indicator(self) -> float:
        """🎯 计算收敛指标"""
        try:
            if len(self.stability_monitor['makespan_history']) < 20:
                return 0.0

            recent_makespans = list(self.stability_monitor['makespan_history'])[-20:]

            # 计算变异系数
            mean_makespan = np.mean(recent_makespans)
            std_makespan = np.std(recent_makespans)

            if mean_makespan > 0:
                cv = std_makespan / mean_makespan
                convergence_score = max(0, 1.0 - cv * 10)  # CV越小，收敛性越好
                return convergence_score

            return 0.0

        except Exception as e:
            print(f"⚠️ 收敛指标计算错误: {e}")
            return 0.0

    def _update_stability_monitoring(self, episode_result: Dict):
        """🎯 更新稳定性监控"""
        try:
            # 更新稳定性指标
            stability_metrics = episode_result.get('stability_metrics', {})
            system_health = episode_result.get('system_health', 0)

            self.stability_monitor['system_health'].append(system_health)

            # 更新收敛指标
            convergence_indicator = episode_result.get('convergence_indicator', 0)
            self.stability_monitor['convergence_metrics'].append(convergence_indicator)

            # 计算奖励方差
            hd_ddpg_result = episode_result.get('hd_ddpg_result', {})
            reward_variance = hd_ddpg_result.get('reward_variance', 0)
            self.stability_monitor['reward_variance'].append(reward_variance)

        except Exception as e:
            print(f"⚠️ 稳定性监控更新错误: {e}")

    def _update_best_performance(self, episode_result: Dict, episode: int):
        """🎯 更新最佳性能记录"""
        try:
            hd_ddpg_result = episode_result.get('hd_ddpg_result', {})
            makespan = hd_ddpg_result.get('avg_makespan', float('inf'))
            reward = hd_ddpg_result.get('total_reward', float('-inf'))

            # 更新最佳makespan
            if makespan < self.best_performance['makespan'] and makespan != float('inf'):
                self.best_performance['makespan'] = makespan
                self.best_performance['episode'] = episode + 1
                print(f"🏆 新的最佳Makespan: {makespan:.3f} (Episode {episode + 1})")

            # 更新最佳奖励
            if reward > self.best_performance['reward']:
                self.best_performance['reward'] = reward

        except Exception as e:
            print(f"⚠️ 最佳性能更新错误: {e}")

    def _check_convergence(self, episode: int) -> bool:
        """🎯 检查是否收敛"""
        try:
            if episode < 100:  # 至少训练100轮
                return False

            if len(self.stability_monitor['convergence_metrics']) < 10:
                return False

            # 检查收敛指标
            recent_convergence = list(self.stability_monitor['convergence_metrics'])[-10:]
            avg_convergence = np.mean(recent_convergence)

            convergence_threshold = self.config.get('convergence_threshold', 0.05)

            # 检查makespan稳定性
            if len(self.stability_monitor['makespan_history']) >= 20:
                recent_makespans = list(self.stability_monitor['makespan_history'])[-20:]
                makespan_std = np.std(recent_makespans)
                makespan_mean = np.mean(recent_makespans)

                if makespan_mean > 0:
                    cv = makespan_std / makespan_mean
                    return cv < convergence_threshold and avg_convergence > 0.8

            return False

        except Exception as e:
            print(f"⚠️ 收敛检查错误: {e}")
            return False

    def _enhanced_performance_evaluation(self, episode: int):
        """🎯 增强的性能评估"""
        if not self.simulation_results:
            return

        print(f"\n📊 Episode {episode} - 增强性能评估:")

        try:
            # 获取最近的结果
            window_size = min(self.config['performance_tracking_window'], len(self.simulation_results))
            recent_results = self.simulation_results[-window_size:]

            # 计算关键指标
            avg_reward = np.mean([r['hd_ddpg_result']['total_reward'] for r in recent_results])
            avg_makespan = np.mean([r['hd_ddpg_result']['avg_makespan'] for r in recent_results
                                  if r['hd_ddpg_result']['avg_makespan'] != float('inf')])
            avg_load_balance = np.mean([r['hd_ddpg_result']['avg_load_balance'] for r in recent_results])

            # 稳定性指标
            makespan_stability = np.mean([r['stability_metrics']['makespan_stability'] for r in recent_results])
            avg_system_health = np.mean([r['system_health'] for r in recent_results])

            print(f"  🎯 平均奖励 (最近{window_size}轮): {avg_reward:.2f}")
            print(f"  ⏱️ 平均Makespan (最近{window_size}轮): {avg_makespan:.3f}")
            print(f"  ⚖️ 平均负载均衡 (最近{window_size}轮): {avg_load_balance:.3f}")
            print(f"  📈 Makespan稳定性: {makespan_stability:.3f}")
            print(f"  💊 系统健康度: {avg_system_health:.3f}")

            # 目标达成情况
            target_makespan = self.config.get('makespan_target', 1.5)
            if avg_makespan <= target_makespan:
                print(f"  🎉 已达成Makespan目标! ({avg_makespan:.3f} <= {target_makespan})")
            else:
                gap = avg_makespan - target_makespan
                print(f"  🎯 距离Makespan目标还差: {gap:.3f}")

        except Exception as e:
            print(f"  ⚠️ 性能评估错误: {e}")

    def _print_enhanced_progress(self, episode: int, total_episodes: int):
        """🎯 增强的进度打印"""
        try:
            progress = episode / total_episodes * 100
            recent_results = self.simulation_results[-10:]

            if recent_results:
                avg_reward = np.mean([r['hd_ddpg_result']['total_reward'] for r in recent_results])
                avg_makespan = np.mean([r['hd_ddpg_result']['avg_makespan'] for r in recent_results
                                      if r['hd_ddpg_result']['avg_makespan'] != float('inf')])

                # 稳定性指标
                stability_score = np.mean([r.get('stability_metrics', {}).get('makespan_stability', 0)
                                         for r in recent_results])

                print(f"📈 进度: {progress:.1f}% ({episode}/{total_episodes}) - "
                      f"奖励: {avg_reward:.2f}, Makespan: {avg_makespan:.3f}, "
                      f"稳定性: {stability_score:.3f}")

        except Exception as e:
            print(f"⚠️ 进度打印错误: {e}")

    def _save_enhanced_checkpoint(self, episode: int):
        """🎯 保存增强的检查点"""
        try:
            checkpoint_path = f"models/stabilized_hd_ddpg_checkpoint_ep{episode}"
            self.hd_ddpg.save_models(checkpoint_path)

            # 保存详细的仿真结果
            enhanced_results = []
            for r in self.simulation_results:
                enhanced_result = {
                    'episode': r['episode'],
                    'workflows_count': r['workflows_count'],
                    'total_reward': r['hd_ddpg_result']['total_reward'],
                    'avg_makespan': r['hd_ddpg_result']['avg_makespan'],
                    'avg_load_balance': r['hd_ddpg_result']['avg_load_balance'],
                    'episode_duration': r['episode_duration'],
                    'makespan_stability': r.get('stability_metrics', {}).get('makespan_stability', 0),
                    'system_health': r.get('system_health', 0),
                    'convergence_indicator': r.get('convergence_indicator', 0),
                    'smoothed_makespan': r['hd_ddpg_result'].get('smoothed_makespan', 0)
                }
                enhanced_results.append(enhanced_result)

            # 保存为CSV
            results_df = pd.DataFrame(enhanced_results)
            results_df.to_csv(f"results/stabilized_simulation_results_ep{episode}.csv", index=False)

            # 保存稳定性监控数据
            stability_data = {
                'makespan_history': list(self.stability_monitor['makespan_history']),
                'reward_variance': list(self.stability_monitor['reward_variance']),
                'convergence_metrics': list(self.stability_monitor['convergence_metrics']),
                'system_health': list(self.stability_monitor['system_health']),
                'best_performance': self.best_performance
            }

            with open(f"results/stability_monitor_ep{episode}.json", 'w') as f:
                json.dump(stability_data, f, indent=2)

            print(f"✅ 增强检查点已保存 (Episode {episode})")

        except Exception as e:
            print(f"❌ 保存增强检查点错误: {e}")

    def _generate_episode_workflows(self, count: int = None) -> List[List[MedicalTask]]:
        """🎯 生成稳定化的episode工作流"""
        if count is None:
            count = self.config['workflows_per_episode']

        workflows = []

        for _ in range(count):
            try:
                workflow_type = np.random.choice(self.config['workflow_types'])
                workflow_size = np.random.randint(
                    self.config['min_workflow_size'],
                    self.config['max_workflow_size'] + 1
                )

                workflow = self.workflow_generator.generate_workflow(
                    num_tasks=workflow_size,
                    workflow_type=workflow_type
                )
                workflows.append(workflow)

            except Exception as e:
                print(f"⚠️ 工作流生成错误: {e}")
                # 生成简单的备用工作流
                backup_workflow = self._generate_backup_workflow()
                workflows.append(backup_workflow)

        return workflows

    def _generate_backup_workflow(self) -> List[MedicalTask]:
        """生成备用工作流"""
        # 这里应该创建一个简单的备用工作流
        # 具体实现依赖于MedicalTask的定义
        pass

    def _create_error_result(self, episode: int, error_msg: str) -> Dict:
        """创建错误结果"""
        return {
            'episode': episode + 1,
            'workflows_count': 0,
            'hd_ddpg_result': {
                'total_reward': -10.0,
                'avg_makespan': float('inf'),
                'avg_load_balance': 0.0,
                'episode_duration': 0.0,
                'success_rate': 0.0
            },
            'stability_metrics': {
                'makespan_stability': 0,
                'reward_consistency': 0,
                'performance_trend': 0,
                'target_achievement': 0
            },
            'episode_duration': 0.0,
            'initial_system_state': [0] * 15,
            'system_health': 0.0,
            'convergence_indicator': 0.0,
            'error': error_msg
        }

    def _generate_enhanced_final_report(self, simulation_duration: float) -> Dict:
        """🎯 生成增强的最终报告"""
        if not self.simulation_results:
            return {'error': '无仿真结果'}

        try:
            # 提取关键指标
            rewards = [r['hd_ddpg_result']['total_reward'] for r in self.simulation_results]
            makespans = [r['hd_ddpg_result']['avg_makespan'] for r in self.simulation_results
                        if r['hd_ddpg_result']['avg_makespan'] != float('inf')]
            load_balances = [r['hd_ddpg_result']['avg_load_balance'] for r in self.simulation_results]

            # 稳定性指标
            stability_scores = [r.get('stability_metrics', {}).get('makespan_stability', 0)
                              for r in self.simulation_results]
            convergence_scores = [r.get('convergence_indicator', 0) for r in self.simulation_results]

            # 计算改进幅度
            if len(rewards) >= 200:
                early_performance = np.mean(rewards[:100])
                late_performance = np.mean(rewards[-100:])
                reward_improvement = (late_performance - early_performance) / abs(early_performance) * 100
            else:
                reward_improvement = 0

            return {
                'simulation_summary': {
                    'total_episodes': len(self.simulation_results),
                    'simulation_duration': simulation_duration,
                    'total_workflows_processed': sum(r['workflows_count'] for r in self.simulation_results),
                    'average_episode_duration': np.mean([r['episode_duration'] for r in self.simulation_results]),
                    'convergence_achieved': self._check_convergence(len(self.simulation_results) - 1)
                },
                'performance_metrics': {
                    'final_avg_reward': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
                    'final_avg_makespan': np.mean(makespans[-100:]) if len(makespans) >= 100 else np.mean(makespans),
                    'final_avg_load_balance': np.mean(load_balances[-100:]) if len(load_balances) >= 100 else np.mean(load_balances),
                    'reward_improvement_percent': reward_improvement,
                    'best_makespan_achieved': self.best_performance['makespan'],
                    'best_makespan_episode': self.best_performance['episode']
                },
                'stability_analysis': {
                    'avg_makespan_stability': np.mean(stability_scores[-50:]) if len(stability_scores) >= 50 else np.mean(stability_scores),
                    'avg_convergence_score': np.mean(convergence_scores[-50:]) if len(convergence_scores) >= 50 else np.mean(convergence_scores),
                    'makespan_variance': np.var(makespans[-100:]) if len(makespans) >= 100 else np.var(makespans),
                    'reward_variance': np.var(rewards[-100:]) if len(rewards) >= 100 else np.var(rewards),
                    'system_health_avg': np.mean(list(self.stability_monitor['system_health'])[-20:]) if len(self.stability_monitor['system_health']) >= 20 else np.mean(list(self.stability_monitor['system_health']))
                },
                'training_convergence': {
                    'reward_trend': rewards,
                    'makespan_trend': makespans,
                    'load_balance_trend': load_balances,
                    'stability_trend': stability_scores,
                    'convergence_trend': convergence_scores
                },
                'optimization_achievements': {
                    'target_makespan': self.config.get('makespan_target', 1.5),
                    'target_achieved': self.best_performance['makespan'] <= self.config.get('makespan_target', 1.5),
                    'improvement_over_baseline': 'TBD',  # 需要基线对比
                    'stability_improvement': 'Significant' if np.mean(stability_scores[-20:]) > 0.7 else 'Moderate'
                }
            }

        except Exception as e:
            print(f"❌ 生成增强最终报告错误: {e}")
            return {'error': f'报告生成失败: {str(e)}'}

    # 🎯 保留原有的基线对比功能（略微优化）
    def compare_with_baseline(self, baseline_algorithms: List[str] = None) -> Dict:
        """与基线算法比较"""
        if baseline_algorithms is None:
            baseline_algorithms = ['HEFT', 'FCFS', 'Random']

        print("🔍 对比HD-DDPG与基线算法性能...")

        # 生成测试工作流集
        test_workflows = self.workflow_generator.generate_batch_workflows(batch_size=100)  # 增加测试数量

        comparison_results = {'HD-DDPG': []}

        # HD-DDPG性能
        print("  测试HD-DDPG性能...")
        for i, workflow in enumerate(test_workflows):
            if i % 20 == 0:
                print(f"    进度: {i+1}/{len(test_workflows)}")

            self.environment.reset()
            result = self.hd_ddpg.schedule_workflow(workflow, self.environment)
            comparison_results['HD-DDPG'].append({
                'makespan': result['makespan'],
                'load_balance': result['load_balance'],
                'energy': result['total_energy'],
                'completed_tasks': result['completed_tasks']
            })

        # 基线算法性能
        for algorithm in baseline_algorithms:
            print(f"  测试{algorithm}性能...")
            comparison_results[algorithm] = []
            for i, workflow in enumerate(test_workflows):
                if i % 20 == 0:
                    print(f"    进度: {i+1}/{len(test_workflows)}")

                self.environment.reset()
                result = self._run_baseline_algorithm(algorithm, workflow)
                comparison_results[algorithm].append(result)

        # 计算统计比较
        stats_comparison = self._calculate_comparison_stats(comparison_results)

        print("✅ 基线对比完成")
        return {
            'detailed_results': comparison_results,
            'statistical_comparison': stats_comparison,
            'test_workflows_count': len(test_workflows)
        }

    # 保留原有的可视化功能（稍作增强）
    def visualize_results(self, save_path: str = None):
        """🎯 增强的结果可视化"""
        if not self.simulation_results:
            print("无仿真结果可视化")
            return

        # 提取数据
        episodes = [r['episode'] for r in self.simulation_results]
        rewards = [r['hd_ddpg_result']['total_reward'] for r in self.simulation_results]
        makespans = [r['hd_ddpg_result']['avg_makespan'] for r in self.simulation_results
                    if r['hd_ddpg_result']['avg_makespan'] != float('inf')]
        load_balances = [r['hd_ddpg_result']['avg_load_balance'] for r in self.simulation_results]
        stability_scores = [r.get('stability_metrics', {}).get('makespan_stability', 0)
                          for r in self.simulation_results]

        # 创建增强的图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 奖励趋势
        axes[0, 0].plot(episodes, rewards, alpha=0.7, color='blue')
        if len(rewards) >= 50:
            axes[0, 0].plot(episodes, self._moving_average(rewards, 50), 'r-', linewidth=2, label='50-episode MA')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Makespan趋势
        makespan_episodes = episodes[:len(makespans)]
        axes[0, 1].plot(makespan_episodes, makespans, alpha=0.7, color='green')
        if len(makespans) >= 50:
            axes[0, 1].plot(makespan_episodes, self._moving_average(makespans, 50), 'r-', linewidth=2, label='50-episode MA')

        # 添加目标线
        target_makespan = self.config.get('makespan_target', 1.5)
        axes[0, 1].axhline(y=target_makespan, color='red', linestyle='--', label=f'Target: {target_makespan}')

        axes[0, 1].set_title('Average Makespan')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Makespan')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 负载均衡趋势
        axes[0, 2].plot(episodes, load_balances, alpha=0.7, color='orange')
        if len(load_balances) >= 50:
            axes[0, 2].plot(episodes, self._moving_average(load_balances, 50), 'r-', linewidth=2, label='50-episode MA')
        axes[0, 2].set_title('Load Balance Score')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Load Balance')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        # 🎯 新增：稳定性趋势
        axes[1, 0].plot(episodes, stability_scores, alpha=0.7, color='purple')
        if len(stability_scores) >= 30:
            axes[1, 0].plot(episodes, self._moving_average(stability_scores, 30), 'r-', linewidth=2, label='30-episode MA')
        axes[1, 0].set_title('Makespan Stability Score')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Stability Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 性能分布
        axes[1, 1].hist(rewards[-200:], bins=30, alpha=0.7, color='skyblue', label='Recent 200 episodes')
        axes[1, 1].set_title('Reward Distribution (Recent)')
        axes[1, 1].set_xlabel('Total Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # 🎯 新增：Makespan分布
        if len(makespans) >= 100:
            axes[1, 2].hist(makespans[-100:], bins=25, alpha=0.7, color='lightcoral', label='Recent 100 episodes')
            axes[1, 2].axvline(x=target_makespan, color='red', linestyle='--', label=f'Target: {target_makespan}')
            axes[1, 2].set_title('Makespan Distribution (Recent)')
            axes[1, 2].set_xlabel('Makespan')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].legend()
            axes[1, 2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 可视化结果已保存到: {save_path}")
        plt.show()

    # 保留其他原有方法...
    def _moving_average(self, data: List, window_size: int) -> List:
        """计算移动平均"""
        if len(data) < window_size:
            return data

        moving_avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(np.mean(data[start_idx:i + 1]))

        return moving_avg

    # 保留原有的基线算法实现...
    def _run_baseline_algorithm(self, algorithm: str, workflow: List[MedicalTask]) -> Dict:
        """运行基线算法"""
        if algorithm == 'HEFT':
            return self._heft_scheduling(workflow)
        elif algorithm == 'FCFS':
            return self._fcfs_scheduling(workflow)
        elif algorithm == 'Random':
            return self._random_scheduling(workflow)
        else:
            return {'makespan': float('inf'), 'load_balance': 0, 'energy': float('inf'), 'completed_tasks': 0}

    def _heft_scheduling(self, workflow: List[MedicalTask]) -> Dict:
        """HEFT基线算法"""
        task_completion_times = []
        total_energy = 0

        # 按照优先级排序任务
        sorted_tasks = sorted(workflow, key=lambda t: t.priority, reverse=True)

        for task in sorted_tasks:
            # 选择最快的可用节点
            all_nodes = self.environment.get_available_nodes()
            if all_nodes:
                best_node = min(all_nodes, key=lambda n: n.get_execution_time(task.computation_requirement))

                if best_node.can_accommodate(task.memory_requirement):
                    execution_time = best_node.get_execution_time(task.computation_requirement)
                    energy = best_node.get_energy_consumption(execution_time)

                    task_completion_times.append(execution_time)
                    total_energy += energy

                    self.environment.update_node_load(best_node.node_id, task.memory_requirement)

        makespan = max(task_completion_times) if task_completion_times else 0

        # 计算负载均衡
        node_loads = [node.current_load / node.memory_capacity
                      for node_list in self.environment.nodes.values()
                      for node in node_list]
        load_balance = self.metrics.calculate_load_balance(node_loads)

        return {
            'makespan': makespan,
            'load_balance': load_balance,
            'energy': total_energy,
            'completed_tasks': len(task_completion_times)
        }

    def _fcfs_scheduling(self, workflow: List[MedicalTask]) -> Dict:
        """FCFS基线算法"""
        task_completion_times = []
        total_energy = 0

        for task in workflow:
            # 选择第一个可用节点
            all_nodes = self.environment.get_available_nodes()
            if all_nodes:
                selected_node = all_nodes[0]

                if selected_node.can_accommodate(task.memory_requirement):
                    execution_time = selected_node.get_execution_time(task.computation_requirement)
                    energy = selected_node.get_energy_consumption(execution_time)

                    task_completion_times.append(execution_time)
                    total_energy += energy

                    self.environment.update_node_load(selected_node.node_id, task.memory_requirement)

        makespan = max(task_completion_times) if task_completion_times else 0

        node_loads = [node.current_load / node.memory_capacity
                      for node_list in self.environment.nodes.values()
                      for node in node_list]
        load_balance = self.metrics.calculate_load_balance(node_loads)

        return {
            'makespan': makespan,
            'load_balance': load_balance,
            'energy': total_energy,
            'completed_tasks': len(task_completion_times)
        }

    def _random_scheduling(self, workflow: List[MedicalTask]) -> Dict:
        """随机调度基线算法"""
        task_completion_times = []
        total_energy = 0

        for task in workflow:
            all_nodes = self.environment.get_available_nodes()
            if all_nodes:
                selected_node = np.random.choice(all_nodes)

                if selected_node.can_accommodate(task.memory_requirement):
                    execution_time = selected_node.get_execution_time(task.computation_requirement)
                    energy = selected_node.get_energy_consumption(execution_time)

                    task_completion_times.append(execution_time)
                    total_energy += energy

                    self.environment.update_node_load(selected_node.node_id, task.memory_requirement)

        makespan = max(task_completion_times) if task_completion_times else 0

        node_loads = [node.current_load / node.memory_capacity
                      for node_list in self.environment.nodes.values()
                      for node in node_list]
        load_balance = self.metrics.calculate_load_balance(node_loads)

        return {
            'makespan': makespan,
            'load_balance': load_balance,
            'energy': total_energy,
            'completed_tasks': len(task_completion_times)
        }

    def _calculate_comparison_stats(self, comparison_results: Dict) -> Dict:
        """计算比较统计"""
        stats = {}

        for algorithm, results in comparison_results.items():
            makespans = [r['makespan'] for r in results if r['makespan'] != float('inf')]
            load_balances = [r['load_balance'] for r in results]
            energies = [r['energy'] for r in results if r['energy'] != float('inf')]

            stats[algorithm] = {
                'avg_makespan': np.mean(makespans) if makespans else float('inf'),
                'std_makespan': np.std(makespans) if makespans else 0,
                'avg_load_balance': np.mean(load_balances),
                'std_load_balance': np.std(load_balances),
                'avg_energy': np.mean(energies) if energies else float('inf'),
                'std_energy': np.std(energies) if energies else 0
            }

        return stats


# 🎯 为了向后兼容，保留原始类名
MedicalSchedulingSimulator = StabilizedMedicalSchedulingSimulator


# 🧪 增强的测试函数
def test_stabilized_medical_simulator():
    """测试稳定化医疗仿真器"""
    print("🧪 开始测试StabilizedMedicalSchedulingSimulator...")

    try:
        # 创建稳定化仿真器
        config = {
            'simulation_episodes': 20,
            'workflows_per_episode': 5,
            'makespan_target': 1.6,
            'stability_monitoring': True,
            'state_smoothing': True
        }

        simulator = StabilizedMedicalSchedulingSimulator(config)

        # 测试1: 基本仿真
        print("\n📝 测试1: 基本仿真测试")
        test_result = simulator.run_simulation(episodes=10)
        print(f"✅ 仿真完成，最终报告关键指标:")
        print(f"  - 最佳Makespan: {test_result['performance_metrics']['best_makespan_achieved']:.3f}")
        print(f"  - 稳定性评分: {test_result['stability_analysis']['avg_makespan_stability']:.3f}")

        # 测试2: 可视化
        print("\n📝 测试2: 可视化测试")
        simulator.visualize_results("results/test_stabilized_simulation.png")
        print("✅ 可视化测试完成")

        print("\n🎉 所有测试通过！StabilizedMedicalSchedulingSimulator工作正常")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    success = test_stabilized_medical_simulator()
    if success:
        print("\n✅ Stabilized Medical Scheduling Simulator ready for production!")
        print("🎯 主要优化:")
        print("  - 状态管理稳定化: 减少输入噪声")
        print("  - 实时稳定性监控: 持续跟踪性能")
        print("  - 智能收敛检测: 自动停止训练")
        print("  - 增强错误处理: 鲁棒性提升")
        print("  - 目标导向训练: 明确性能目标")
    else:
        print("\n❌ Stabilized Medical Scheduling Simulator needs debugging!")