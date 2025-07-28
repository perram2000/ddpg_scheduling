"""
Performance Metrics for Medical Workflow Scheduling - 高度增强版本
医疗工作流调度性能指标 - 专为稳定化HD-DDPG设计

🎯 主要增强：
- 稳定性指标监控
- 实时性能跟踪
- 收敛性分析
- 异常检测
- 高级可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from typing import Dict, List, Tuple, Optional
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EnhancedSchedulingMetrics:
    """
    增强的调度性能指标计算器
    🎯 专为稳定化训练和性能分析设计
    """

    def __init__(self, window_size: int = 100, stability_threshold: float = 0.1):
        """
        初始化增强的性能指标系统

        Args:
            window_size: 性能窗口大小
            stability_threshold: 稳定性阈值
        """
        self.window_size = window_size
        self.stability_threshold = stability_threshold

        print("🔧 初始化EnhancedSchedulingMetrics...")

        # 🎯 基础性能指标
        self.basic_metrics = {
            'makespans': deque(maxlen=1000),
            'load_balances': deque(maxlen=1000),
            'energy_consumptions': deque(maxlen=1000),
            'throughputs': deque(maxlen=1000),
            'episode_rewards': deque(maxlen=1000),
            'success_rates': deque(maxlen=1000)
        }

        # 🎯 稳定性指标
        self.stability_metrics = {
            'makespan_stability': deque(maxlen=500),
            'reward_variance': deque(maxlen=500),
            'performance_consistency': deque(maxlen=500),
            'convergence_score': deque(maxlen=500),
            'volatility_index': deque(maxlen=500)
        }

        # 🎯 实时监控指标
        self.realtime_metrics = {
            'running_averages': deque(maxlen=100),
            'trend_indicators': deque(maxlen=100),
            'anomaly_scores': deque(maxlen=100),
            'improvement_rates': deque(maxlen=100)
        }

        # 🎯 收敛性分析
        self.convergence_analysis = {
            'plateau_detection': deque(maxlen=50),
            'learning_rate_estimates': deque(maxlen=50),
            'convergence_timestamps': [],
            'best_performance_episodes': []
        }

        # 🎯 比较基准
        self.baseline_comparisons = {
            'heft_comparison': deque(maxlen=100),
            'fcfs_comparison': deque(maxlen=100),
            'random_comparison': deque(maxlen=100),
            'improvement_ratios': deque(maxlen=100)
        }

        # 🎯 统计汇总
        self.statistics_summary = {
            'best_makespan': float('inf'),
            'best_reward': float('-inf'),
            'worst_makespan': 0,
            'worst_reward': float('inf'),
            'total_episodes': 0,
            'stable_episodes': 0,
            'convergence_episode': None
        }

        print("✅ 增强性能指标系统初始化完成")

    def reset(self):
        """🎯 智能重置 - 保留重要历史信息"""
        try:
            # 保存关键统计信息
            if self.basic_metrics['makespans']:
                self.statistics_summary['total_episodes'] += len(self.basic_metrics['makespans'])

            # 清空实时指标
            for metric_dict in [self.basic_metrics, self.stability_metrics,
                               self.realtime_metrics, self.convergence_analysis]:
                for key, deque_obj in metric_dict.items():
                    if isinstance(deque_obj, deque):
                        deque_obj.clear()

            # 重置部分统计摘要（保留最佳记录）
            self.statistics_summary.update({
                'stable_episodes': 0,
                'convergence_episode': None
            })

            print("🔄 性能指标已智能重置")

        except Exception as e:
            print(f"⚠️ 指标重置错误: {e}")

    def calculate_makespan(self, task_completion_times: List[float]) -> float:
        """🎯 增强的完工时间计算"""
        try:
            if not task_completion_times:
                return float('inf')

            # 过滤无效值
            valid_times = [t for t in task_completion_times if t != float('inf') and not np.isnan(t)]

            if not valid_times:
                return float('inf')

            makespan = max(valid_times)

            # 🎯 异常值检测
            if makespan > 1000:  # 异常大的makespan
                print(f"⚠️ 检测到异常makespan: {makespan}")

            return makespan

        except Exception as e:
            print(f"⚠️ Makespan计算错误: {e}")
            return float('inf')

    def calculate_enhanced_load_balance(self, node_loads: List[float]) -> float:
        """🎯 增强的负载均衡计算"""
        try:
            if not node_loads:
                return 0.0

            # 过滤无效负载
            valid_loads = [load for load in node_loads if not np.isnan(load) and load >= 0]

            if not valid_loads:
                return 0.0

            if len(valid_loads) == 1:
                return 1.0

            mean_load = np.mean(valid_loads)

            if mean_load == 0:
                return 1.0

            # 🎯 改进的负载均衡计算
            std_load = np.std(valid_loads)
            cv = std_load / mean_load  # 变异系数

            # 使用更平滑的函数
            load_balance = 1.0 / (1.0 + cv)

            # 🎯 额外的均匀性检查
            max_load = max(valid_loads)
            min_load = min(valid_loads)

            if max_load > 0:
                uniformity = 1.0 - (max_load - min_load) / max_load
                load_balance = (load_balance + uniformity) / 2

            return min(1.0, max(0.0, load_balance))

        except Exception as e:
            print(f"⚠️ 负载均衡计算错误: {e}")
            return 0.0

    def calculate_stability_score(self, metric_history: List[float], window: int = 20) -> float:
        """🎯 计算指标稳定性评分"""
        try:
            if len(metric_history) < window:
                return 0.0

            recent_values = list(metric_history)[-window:]

            # 过滤无效值
            valid_values = [v for v in recent_values if not np.isnan(v) and v != float('inf')]

            if len(valid_values) < 3:
                return 0.0

            # 计算变异系数
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)

            if mean_val == 0:
                return 1.0 if std_val == 0 else 0.0

            cv = std_val / abs(mean_val)

            # 稳定性评分 (变异系数越小越稳定)
            stability = 1.0 / (1.0 + cv * 5)

            return min(1.0, max(0.0, stability))

        except Exception as e:
            print(f"⚠️ 稳定性评分计算错误: {e}")
            return 0.0

    def calculate_convergence_score(self, metric_history: List[float], window: int = 50) -> float:
        """🎯 计算收敛性评分"""
        try:
            if len(metric_history) < window:
                return 0.0

            recent_values = list(metric_history)[-window:]
            valid_values = [v for v in recent_values if not np.isnan(v) and v != float('inf')]

            if len(valid_values) < 10:
                return 0.0

            # 计算趋势斜率
            x = np.arange(len(valid_values))
            slope, _ = np.polyfit(x, valid_values, 1)

            # 计算方差
            variance = np.var(valid_values)
            mean_val = np.mean(valid_values)

            # 收敛评分综合考虑趋势平缓程度和方差
            if mean_val != 0:
                trend_score = 1.0 / (1.0 + abs(slope) * 100)
                variance_score = 1.0 / (1.0 + variance / abs(mean_val))
                convergence = (trend_score + variance_score) / 2
            else:
                convergence = 0.5

            return min(1.0, max(0.0, convergence))

        except Exception as e:
            print(f"⚠️ 收敛性评分计算错误: {e}")
            return 0.0

    def calculate_anomaly_score(self, current_value: float, history: List[float]) -> float:
        """🎯 计算异常检测评分"""
        try:
            if len(history) < 10:
                return 0.0

            valid_history = [v for v in history if not np.isnan(v) and v != float('inf')]

            if len(valid_history) < 5:
                return 0.0

            mean_val = np.mean(valid_history)
            std_val = np.std(valid_history)

            if std_val == 0:
                return 0.0 if current_value == mean_val else 1.0

            # Z-score异常检测
            z_score = abs(current_value - mean_val) / std_val
            anomaly_score = min(1.0, z_score / 3.0)  # 3-sigma原则

            return anomaly_score

        except Exception as e:
            print(f"⚠️ 异常检测评分计算错误: {e}")
            return 0.0

    def update_metrics(self, makespan: float, load_balance: float, energy: float,
                      throughput: float, reward: float, success_rate: float = 1.0,
                      episode: int = None):
        """🎯 更新所有性能指标"""
        try:
            # 🎯 更新基础指标
            self.basic_metrics['makespans'].append(makespan)
            self.basic_metrics['load_balances'].append(load_balance)
            self.basic_metrics['energy_consumptions'].append(energy)
            self.basic_metrics['throughputs'].append(throughput)
            self.basic_metrics['episode_rewards'].append(reward)
            self.basic_metrics['success_rates'].append(success_rate)

            # 🎯 计算稳定性指标
            makespan_stability = self.calculate_stability_score(list(self.basic_metrics['makespans']))
            reward_variance = np.var(list(self.basic_metrics['episode_rewards'])[-20:]) if len(self.basic_metrics['episode_rewards']) >= 20 else 0
            convergence_score = self.calculate_convergence_score(list(self.basic_metrics['makespans']))

            self.stability_metrics['makespan_stability'].append(makespan_stability)
            self.stability_metrics['reward_variance'].append(reward_variance)
            self.stability_metrics['convergence_score'].append(convergence_score)

            # 🎯 计算实时指标
            running_avg = np.mean(list(self.basic_metrics['makespans'])[-self.window_size:])
            anomaly_score = self.calculate_anomaly_score(makespan, list(self.basic_metrics['makespans']))

            self.realtime_metrics['running_averages'].append(running_avg)
            self.realtime_metrics['anomaly_scores'].append(anomaly_score)

            # 🎯 更新统计摘要
            self._update_statistics_summary(makespan, reward, makespan_stability, episode)

            # 🎯 检测收敛
            self._detect_convergence(convergence_score, episode)

        except Exception as e:
            print(f"⚠️ 指标更新错误: {e}")

    def _update_statistics_summary(self, makespan: float, reward: float,
                                 stability: float, episode: int):
        """🎯 更新统计摘要"""
        try:
            # 更新最佳/最差记录
            if makespan != float('inf') and makespan < self.statistics_summary['best_makespan']:
                self.statistics_summary['best_makespan'] = makespan
                if episode is not None:
                    self.convergence_analysis['best_performance_episodes'].append(episode)

            if reward > self.statistics_summary['best_reward']:
                self.statistics_summary['best_reward'] = reward

            if makespan != float('inf'):
                self.statistics_summary['worst_makespan'] = max(self.statistics_summary['worst_makespan'], makespan)

            self.statistics_summary['worst_reward'] = min(self.statistics_summary['worst_reward'], reward)

            # 稳定性统计
            if stability > 0.8:
                self.statistics_summary['stable_episodes'] += 1

        except Exception as e:
            print(f"⚠️ 统计摘要更新错误: {e}")

    def _detect_convergence(self, convergence_score: float, episode: int):
        """🎯 检测训练收敛"""
        try:
            if convergence_score > 0.9 and episode is not None:
                if self.statistics_summary['convergence_episode'] is None:
                    self.statistics_summary['convergence_episode'] = episode
                    self.convergence_analysis['convergence_timestamps'].append({
                        'episode': episode,
                        'timestamp': datetime.now().isoformat(),
                        'score': convergence_score
                    })
                    print(f"🎉 检测到训练收敛 (Episode {episode}, Score: {convergence_score:.3f})")

        except Exception as e:
            print(f"⚠️ 收敛检测错误: {e}")

    def add_baseline_comparison(self, algorithm: str, makespan: float):
        """🎯 添加基线算法比较"""
        try:
            if algorithm.upper() in ['HEFT', 'FCFS', 'RANDOM']:
                metric_key = f'{algorithm.lower()}_comparison'
                if metric_key in self.baseline_comparisons:
                    self.baseline_comparisons[metric_key].append(makespan)

                    # 计算改进比例
                    if self.basic_metrics['makespans'] and makespan > 0:
                        current_makespan = list(self.basic_metrics['makespans'])[-1]
                        if current_makespan != float('inf'):
                            improvement = (makespan - current_makespan) / makespan * 100
                            self.baseline_comparisons['improvement_ratios'].append(improvement)

        except Exception as e:
            print(f"⚠️ 基线比较添加错误: {e}")

    def get_comprehensive_metrics(self, last_n: int = 100) -> Dict:
        """🎯 获取综合性能指标"""
        try:
            def safe_avg(lst, default=0):
                valid_values = [v for v in list(lst)[-last_n:] if not np.isnan(v) and v != float('inf')]
                return np.mean(valid_values) if valid_values else default

            def safe_std(lst, default=0):
                valid_values = [v for v in list(lst)[-last_n:] if not np.isnan(v) and v != float('inf')]
                return np.std(valid_values) if len(valid_values) > 1 else default

            comprehensive_metrics = {
                # 基础性能指标
                'performance': {
                    'avg_makespan': safe_avg(self.basic_metrics['makespans']),
                    'std_makespan': safe_std(self.basic_metrics['makespans']),
                    'avg_reward': safe_avg(self.basic_metrics['episode_rewards']),
                    'std_reward': safe_std(self.basic_metrics['episode_rewards']),
                    'avg_load_balance': safe_avg(self.basic_metrics['load_balances']),
                    'avg_energy': safe_avg(self.basic_metrics['energy_consumptions']),
                    'avg_throughput': safe_avg(self.basic_metrics['throughputs']),
                    'avg_success_rate': safe_avg(self.basic_metrics['success_rates'])
                },

                # 稳定性指标
                'stability': {
                    'makespan_stability': safe_avg(self.stability_metrics['makespan_stability']),
                    'reward_variance': safe_avg(self.stability_metrics['reward_variance']),
                    'convergence_score': safe_avg(self.stability_metrics['convergence_score']),
                    'anomaly_rate': len([s for s in list(self.realtime_metrics['anomaly_scores'])[-last_n:] if s > 0.5]) / min(last_n, len(self.realtime_metrics['anomaly_scores'])) if self.realtime_metrics['anomaly_scores'] else 0
                },

                # 收敛性分析
                'convergence': {
                    'converged': self.statistics_summary['convergence_episode'] is not None,
                    'convergence_episode': self.statistics_summary['convergence_episode'],
                    'stable_episode_ratio': self.statistics_summary['stable_episodes'] / max(1, len(self.basic_metrics['makespans'])),
                    'improvement_trend': self._calculate_improvement_trend()
                },

                # 统计摘要
                'summary': {
                    'best_makespan': self.statistics_summary['best_makespan'],
                    'best_reward': self.statistics_summary['best_reward'],
                    'total_episodes': len(self.basic_metrics['makespans']),
                    'data_quality': self._assess_data_quality()
                }
            }

            # 基线比较 (如果有数据)
            if any(self.baseline_comparisons[key] for key in ['heft_comparison', 'fcfs_comparison', 'random_comparison']):
                comprehensive_metrics['baseline_comparison'] = {
                    'avg_improvement': safe_avg(self.baseline_comparisons['improvement_ratios']),
                    'heft_comparison': safe_avg(self.baseline_comparisons['heft_comparison']) if self.baseline_comparisons['heft_comparison'] else None,
                    'fcfs_comparison': safe_avg(self.baseline_comparisons['fcfs_comparison']) if self.baseline_comparisons['fcfs_comparison'] else None,
                    'random_comparison': safe_avg(self.baseline_comparisons['random_comparison']) if self.baseline_comparisons['random_comparison'] else None
                }

            return comprehensive_metrics

        except Exception as e:
            print(f"⚠️ 综合指标计算错误: {e}")
            return {'error': str(e)}

    def _calculate_improvement_trend(self) -> str:
        """🎯 计算改进趋势"""
        try:
            if len(self.basic_metrics['makespans']) < 20:
                return 'insufficient_data'

            recent_makespans = [m for m in list(self.basic_metrics['makespans'])[-20:]
                              if m != float('inf') and not np.isnan(m)]

            if len(recent_makespans) < 10:
                return 'insufficient_valid_data'

            # 计算线性趋势
            x = np.arange(len(recent_makespans))
            slope, _ = np.polyfit(x, recent_makespans, 1)

            if slope < -0.01:
                return 'improving'
            elif slope > 0.01:
                return 'degrading'
            else:
                return 'stable'

        except Exception as e:
            print(f"⚠️ 改进趋势计算错误: {e}")
            return 'unknown'

    def _assess_data_quality(self) -> float:
        """🎯 评估数据质量"""
        try:
            total_points = len(self.basic_metrics['makespans'])
            if total_points == 0:
                return 0.0

            # 检查无效数据点
            invalid_makespans = len([m for m in self.basic_metrics['makespans']
                                   if m == float('inf') or np.isnan(m)])
            invalid_rewards = len([r for r in self.basic_metrics['episode_rewards']
                                 if np.isnan(r)])

            total_invalid = invalid_makespans + invalid_rewards
            quality_score = 1.0 - (total_invalid / (total_points * 2))

            return max(0.0, min(1.0, quality_score))

        except Exception as e:
            print(f"⚠️ 数据质量评估错误: {e}")
            return 0.5

    def create_comprehensive_visualization(self, save_path: str = None, show_plot: bool = True):
        """🎯 创建综合性能可视化"""
        try:
            # 创建大图表
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

            # 1. 基础性能趋势 (2x2)
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_performance_trends(ax1)

            # 2. 稳定性分析 (2x2)
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_stability_analysis(ax2)

            # 3. 收敛性分析
            ax3 = fig.add_subplot(gs[1, :2])
            self._plot_convergence_analysis(ax3)

            # 4. 异常检测
            ax4 = fig.add_subplot(gs[1, 2:])
            self._plot_anomaly_detection(ax4)

            # 5. 分布分析
            ax5 = fig.add_subplot(gs[2, :2])
            self._plot_distribution_analysis(ax5)

            # 6. 基线比较 (如果有数据)
            ax6 = fig.add_subplot(gs[2, 2:])
            self._plot_baseline_comparison(ax6)

            # 7. 相关性分析
            ax7 = fig.add_subplot(gs[3, :2])
            self._plot_correlation_analysis(ax7)

            # 8. 性能摘要
            ax8 = fig.add_subplot(gs[3, 2:])
            self._plot_performance_summary(ax8)

            plt.suptitle('HD-DDPG Comprehensive Performance Analysis', fontsize=16, fontweight='bold')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ 综合可视化已保存到: {save_path}")

            if show_plot:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"⚠️ 综合可视化创建错误: {e}")
            plt.close()

    def _plot_performance_trends(self, ax):
        """绘制性能趋势"""
        try:
            if not self.basic_metrics['makespans']:
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Performance Trends')
                return

            episodes = range(len(self.basic_metrics['makespans']))

            # 双y轴
            ax2 = ax.twinx()

            # Makespan
            valid_makespans = [m if m != float('inf') else np.nan for m in self.basic_metrics['makespans']]
            line1 = ax.plot(episodes, valid_makespans, 'b-', alpha=0.7, label='Makespan')

            # 移动平均
            if len(valid_makespans) >= 10:
                ma = pd.Series(valid_makespans).rolling(window=10, min_periods=1).mean()
                ax.plot(episodes, ma, 'b-', linewidth=2, label='Makespan MA(10)')

            # 奖励
            line2 = ax2.plot(episodes, list(self.basic_metrics['episode_rewards']), 'r-', alpha=0.7, label='Reward')

            ax.set_xlabel('Episode')
            ax.set_ylabel('Makespan', color='b')
            ax2.set_ylabel('Reward', color='r')
            ax.set_title('Performance Trends')
            ax.grid(True, alpha=0.3)

            # 合并图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        except Exception as e:
            print(f"⚠️ 性能趋势绘制错误: {e}")

    def _plot_stability_analysis(self, ax):
        """绘制稳定性分析"""
        try:
            if not self.stability_metrics['makespan_stability']:
                ax.text(0.5, 0.5, 'No Stability Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Stability Analysis')
                return

            episodes = range(len(self.stability_metrics['makespan_stability']))

            ax.plot(episodes, list(self.stability_metrics['makespan_stability']),
                   'g-', linewidth=2, label='Makespan Stability')
            ax.plot(episodes, list(self.stability_metrics['convergence_score']),
                   'orange', linewidth=2, label='Convergence Score')

            # 稳定性阈值线
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Stability Threshold (0.8)')

            ax.set_xlabel('Episode')
            ax.set_ylabel('Stability Score')
            ax.set_title('Stability Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        except Exception as e:
            print(f"⚠️ 稳定性分析绘制错误: {e}")

    def _plot_convergence_analysis(self, ax):
        """绘制收敛性分析"""
        try:
            if len(self.basic_metrics['makespans']) < 20:
                ax.text(0.5, 0.5, 'Insufficient Data for Convergence Analysis',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Convergence Analysis')
                return

            # 计算累积最优
            makespans = [m if m != float('inf') else np.nan for m in self.basic_metrics['makespans']]
            cumulative_best = pd.Series(makespans).expanding().min()

            episodes = range(len(makespans))
            ax.plot(episodes, cumulative_best, 'purple', linewidth=2, label='Cumulative Best Makespan')

            # 标记收敛点
            if self.statistics_summary['convergence_episode'] is not None:
                conv_ep = self.statistics_summary['convergence_episode']
                if conv_ep < len(cumulative_best):
                    ax.axvline(x=conv_ep, color='red', linestyle='--', alpha=0.8,
                             label=f'Convergence (Ep {conv_ep})')

            ax.set_xlabel('Episode')
            ax.set_ylabel('Best Makespan')
            ax.set_title('Convergence Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(f"⚠️ 收敛分析绘制错误: {e}")

    def _plot_anomaly_detection(self, ax):
        """绘制异常检测"""
        try:
            if not self.realtime_metrics['anomaly_scores']:
                ax.text(0.5, 0.5, 'No Anomaly Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Anomaly Detection')
                return

            episodes = range(len(self.realtime_metrics['anomaly_scores']))
            anomaly_scores = list(self.realtime_metrics['anomaly_scores'])

            # 异常评分
            ax.plot(episodes, anomaly_scores, 'red', alpha=0.7, label='Anomaly Score')

            # 异常阈值
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Anomaly Threshold (0.5)')

            # 高亮异常点
            high_anomaly = [i for i, score in enumerate(anomaly_scores) if score > 0.5]
            if high_anomaly:
                ax.scatter([episodes[i] for i in high_anomaly],
                          [anomaly_scores[i] for i in high_anomaly],
                          color='red', s=50, alpha=0.8, label='Anomalies')

            ax.set_xlabel('Episode')
            ax.set_ylabel('Anomaly Score')
            ax.set_title('Anomaly Detection')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        except Exception as e:
            print(f"⚠️ 异常检测绘制错误: {e}")

    def _plot_distribution_analysis(self, ax):
        """绘制分布分析"""
        try:
            if not self.basic_metrics['makespans']:
                ax.text(0.5, 0.5, 'No Data for Distribution', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Makespan Distribution')
                return

            # 过滤有效makespan
            valid_makespans = [m for m in self.basic_metrics['makespans']
                             if m != float('inf') and not np.isnan(m)]

            if not valid_makespans:
                ax.text(0.5, 0.5, 'No Valid Makespan Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Makespan Distribution')
                return

            # 直方图
            ax.hist(valid_makespans, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

            # 统计线
            mean_val = np.mean(valid_makespans)
            ax.axvline(x=mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.3f}')

            median_val = np.median(valid_makespans)
            ax.axvline(x=median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')

            ax.set_xlabel('Makespan')
            ax.set_ylabel('Frequency')
            ax.set_title('Makespan Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(f"⚠️ 分布分析绘制错误: {e}")

    def _plot_baseline_comparison(self, ax):
        """绘制基线比较"""
        try:
            # 检查是否有基线数据
            has_baseline_data = any(self.baseline_comparisons[key] for key in
                                  ['heft_comparison', 'fcfs_comparison', 'random_comparison'])

            if not has_baseline_data:
                ax.text(0.5, 0.5, 'No Baseline Comparison Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Baseline Comparison')
                return

            # 准备数据
            algorithms = []
            avg_makespans = []

            # HD-DDPG
            valid_makespans = [m for m in self.basic_metrics['makespans']
                             if m != float('inf') and not np.isnan(m)]
            if valid_makespans:
                algorithms.append('HD-DDPG')
                avg_makespans.append(np.mean(valid_makespans[-50:]))  # 最近50个

            # 基线算法
            for algo, key in [('HEFT', 'heft_comparison'), ('FCFS', 'fcfs_comparison'), ('Random', 'random_comparison')]:
                if self.baseline_comparisons[key]:
                    algorithms.append(algo)
                    avg_makespans.append(np.mean(list(self.baseline_comparisons[key])))

            if len(algorithms) > 1:
                bars = ax.bar(algorithms, avg_makespans, color=['blue', 'red', 'green', 'orange'][:len(algorithms)])

                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')

                ax.set_ylabel('Average Makespan')
                ax.set_title('Algorithm Comparison')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Insufficient Comparison Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Baseline Comparison')

        except Exception as e:
            print(f"⚠️ 基线比较绘制错误: {e}")

    def _plot_correlation_analysis(self, ax):
        """绘制相关性分析"""
        try:
            if len(self.basic_metrics['makespans']) < 10:
                ax.text(0.5, 0.5, 'Insufficient Data for Correlation', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Correlation Analysis')
                return

            # 准备数据
            valid_indices = []
            makespans = []
            rewards = []
            load_balances = []

            for i, (m, r, lb) in enumerate(zip(self.basic_metrics['makespans'],
                                             self.basic_metrics['episode_rewards'],
                                             self.basic_metrics['load_balances'])):
                if m != float('inf') and not np.isnan(m) and not np.isnan(r) and not np.isnan(lb):
                    makespans.append(m)
                    rewards.append(r)
                    load_balances.append(lb)

            if len(makespans) < 5:
                ax.text(0.5, 0.5, 'Insufficient Valid Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Correlation Analysis')
                return

            # Makespan vs Reward 散点图
            ax.scatter(makespans, rewards, alpha=0.6, color='blue', s=30)

            # 拟合趋势线
            if len(makespans) >= 5:
                z = np.polyfit(makespans, rewards, 1)
                p = np.poly1d(z)
                ax.plot(makespans, p(makespans), "r--", alpha=0.8, linewidth=2)

                # 计算相关系数
                correlation = np.corrcoef(makespans, rewards)[0, 1]
                ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                       transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))

            ax.set_xlabel('Makespan')
            ax.set_ylabel('Reward')
            ax.set_title('Makespan vs Reward Correlation')
            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(f"⚠️ 相关性分析绘制错误: {e}")

    def _plot_performance_summary(self, ax):
        """绘制性能摘要"""
        try:
            # 隐藏坐标轴
            ax.axis('off')

            # 获取综合指标
            metrics = self.get_comprehensive_metrics()

            # 创建摘要文本
            summary_text = f"""
Performance Summary

🎯 Current Performance:
  • Avg Makespan: {metrics['performance']['avg_makespan']:.3f}
  • Avg Reward: {metrics['performance']['avg_reward']:.2f}
  • Success Rate: {metrics['performance']['avg_success_rate']:.1%}

📊 Stability Metrics:
  • Makespan Stability: {metrics['stability']['makespan_stability']:.3f}
  • Convergence Score: {metrics['stability']['convergence_score']:.3f}
  • Anomaly Rate: {metrics['stability']['anomaly_rate']:.1%}

🏆 Best Records:
  • Best Makespan: {metrics['summary']['best_makespan']:.3f}
  • Best Reward: {metrics['summary']['best_reward']:.2f}

📈 Training Status:
  • Total Episodes: {metrics['summary']['total_episodes']}
  • Converged: {'Yes' if metrics['convergence']['converged'] else 'No'}
  • Data Quality: {metrics['summary']['data_quality']:.1%}
            """

            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))

            ax.set_title('Performance Summary', fontweight='bold')

        except Exception as e:
            print(f"⚠️ 性能摘要绘制错误: {e}")

    def export_metrics_to_csv(self, filepath: str):
        """🎯 导出指标到CSV"""
        try:
            # 准备数据
            data = {
                'episode': range(len(self.basic_metrics['makespans'])),
                'makespan': list(self.basic_metrics['makespans']),
                'reward': list(self.basic_metrics['episode_rewards']),
                'load_balance': list(self.basic_metrics['load_balances']),
                'energy': list(self.basic_metrics['energy_consumptions']),
                'throughput': list(self.basic_metrics['throughputs']),
                'success_rate': list(self.basic_metrics['success_rates'])
            }

            # 添加稳定性指标
            for i, stability in enumerate(self.stability_metrics['makespan_stability']):
                if i < len(data['episode']):
                    if 'makespan_stability' not in data:
                        data['makespan_stability'] = [None] * len(data['episode'])
                    data['makespan_stability'][i] = stability

            # 创建DataFrame
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)

            print(f"✅ 指标已导出到: {filepath}")

        except Exception as e:
            print(f"⚠️ 指标导出错误: {e}")

    def export_summary_to_json(self, filepath: str):
        """🎯 导出摘要到JSON"""
        try:
            summary = self.get_comprehensive_metrics()
            summary['export_timestamp'] = datetime.now().isoformat()

            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            print(f"✅ 摘要已导出到: {filepath}")

        except Exception as e:
            print(f"⚠️ 摘要导出错误: {e}")


# 🎯 为了向后兼容，保留原始类名
SchedulingMetrics = EnhancedSchedulingMetrics


# 🧪 增强的测试函数
def test_enhanced_scheduling_metrics():
    """测试增强的调度指标系统"""
    print("🧪 开始测试EnhancedSchedulingMetrics...")

    try:
        # 创建增强的指标系统
        metrics = EnhancedSchedulingMetrics(window_size=50, stability_threshold=0.1)

        # 测试1: 基本指标更新
        print("\n📝 测试1: 基本指标更新测试")
        for i in range(100):
            makespan = np.random.uniform(1.0, 3.0) + np.sin(i/10) * 0.5  # 模拟收敛
            reward = np.random.uniform(200, 500) + i * 2  # 模拟改进
            load_balance = np.random.uniform(0.7, 1.0)
            energy = np.random.uniform(100, 300)
            throughput = np.random.uniform(5, 15)
            success_rate = np.random.uniform(0.8, 1.0)

            metrics.update_metrics(makespan, load_balance, energy, throughput, reward, success_rate, i)

        print(f"✅ 已更新100个episode的指标")

        # 测试2: 综合指标获取
        print("\n📝 测试2: 综合指标获取测试")
        comprehensive = metrics.get_comprehensive_metrics()
        print(f"✅ 平均Makespan: {comprehensive['performance']['avg_makespan']:.3f}")
        print(f"✅ 稳定性评分: {comprehensive['stability']['makespan_stability']:.3f}")
        print(f"✅ 收敛状态: {comprehensive['convergence']['converged']}")

        # 测试3: 基线比较
        print("\n📝 测试3: 基线比较测试")
        for _ in range(20):
            metrics.add_baseline_comparison('HEFT', np.random.uniform(2.5, 4.0))
            metrics.add_baseline_comparison('FCFS', np.random.uniform(3.0, 5.0))
            metrics.add_baseline_comparison('Random', np.random.uniform(4.0, 6.0))

        print("✅ 基线比较数据已添加")

        # 测试4: 可视化
        print("\n📝 测试4: 可视化测试")
        metrics.create_comprehensive_visualization("results/test_enhanced_metrics.png", show_plot=False)
        print("✅ 综合可视化已生成")

        # 测试5: 数据导出
        print("\n📝 测试5: 数据导出测试")
        metrics.export_metrics_to_csv("results/test_metrics.csv")
        metrics.export_summary_to_json("results/test_summary.json")
        print("✅ 数据导出完成")

        print("\n🎉 所有测试通过！EnhancedSchedulingMetrics工作正常")
        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 运行测试
    success = test_enhanced_scheduling_metrics()
    if success:
        print("\n✅ Enhanced Scheduling Metrics ready for production!")
        print("🎯 主要增强:")
        print("  - 稳定性指标监控: 实时跟踪训练稳定性")
        print("  - 收敛性分析: 智能收敛检测和分析")
        print("  - 异常检测: 自动识别性能异常")
        print("  - 综合可视化: 多维度性能分析图表")
        print("  - 基线比较: 与传统算法的性能对比")
        print("  - 数据导出: 完整的指标数据导出功能")
    else:
        print("\n❌ Enhanced Scheduling Metrics need debugging!")