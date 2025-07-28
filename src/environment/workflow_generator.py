"""
Medical Workflow Generator
医疗工作流生成器
"""

import numpy as np
import networkx as nx
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class MedicalTask:
    """医疗任务类"""
    task_id: int
    task_type: str  # 'IMAGE_PROCESSING', 'DATA_ANALYSIS', 'ML_INFERENCE', etc.
    computation_requirement: float  # MIPS
    data_size: float  # MB
    memory_requirement: float  # MB
    deadline: float  # seconds
    priority: int  # 1-5, 5 is highest
    dependencies: List[int]  # 前置任务ID列表


class MedicalWorkflowGenerator:
    """医疗工作流生成器"""

    def __init__(self):
        self.task_types = {
            'IMAGE_PROCESSING': {
                'computation_range': (100, 800),
                'data_range': (50, 200),
                'memory_range': (512, 2048)
            },
            'DATA_ANALYSIS': {
                'computation_range': (200, 600),
                'data_range': (10, 100),
                'memory_range': (256, 1024)
            },
            'ML_INFERENCE': {
                'computation_range': (300, 1000),
                'data_range': (20, 150),
                'memory_range': (1024, 4096)
            },
            'REPORT_GENERATION': {
                'computation_range': (50, 200),
                'data_range': (5, 50),
                'memory_range': (128, 512)
            },
            'DATABASE_QUERY': {
                'computation_range': (20, 100),
                'data_range': (1, 20),
                'memory_range': (64, 256)
            },
            'VISUALIZATION': {
                'computation_range': (150, 400),
                'data_range': (30, 120),
                'memory_range': (512, 1536)
            }
        }

    def generate_workflow(self, num_tasks: int = 10, workflow_type: str = 'radiology') -> List[MedicalTask]:
        """生成医疗工作流"""
        tasks = []

        # 创建DAG结构
        dag = self._create_dag_structure(num_tasks)

        # 为每个节点创建医疗任务
        for task_id in range(num_tasks):
            task_type = self._select_task_type(workflow_type, task_id, num_tasks)
            task = self._create_medical_task(
                task_id=task_id,
                task_type=task_type,
                dependencies=list(dag.predecessors(task_id))
            )
            tasks.append(task)

        return tasks

    def _create_dag_structure(self, num_tasks: int) -> nx.DiGraph:
        """创建DAG结构"""
        dag = nx.DiGraph()
        dag.add_nodes_from(range(num_tasks))

        # 确保连通性 - 创建主路径
        for i in range(num_tasks - 1):
            if random.random() < 0.7:  # 70%概率添加边
                dag.add_edge(i, i + 1)

        # 添加额外的依赖关系
        for i in range(num_tasks):
            for j in range(i + 2, min(i + 5, num_tasks)):
                if random.random() < 0.3:  # 30%概率添加跨层依赖
                    dag.add_edge(i, j)

        # 确保是DAG（无环）
        if not nx.is_directed_acyclic_graph(dag):
            dag = nx.DiGraph()
            dag.add_nodes_from(range(num_tasks))
            # 重新创建简单的链式结构
            for i in range(num_tasks - 1):
                dag.add_edge(i, i + 1)

        return dag

    def _select_task_type(self, workflow_type: str, task_id: int, total_tasks: int) -> str:
        """根据工作流类型和位置选择任务类型"""
        position_ratio = task_id / total_tasks

        if workflow_type == 'radiology':
            if position_ratio < 0.2:
                return 'DATABASE_QUERY'
            elif position_ratio < 0.4:
                return 'IMAGE_PROCESSING'
            elif position_ratio < 0.7:
                return 'ML_INFERENCE'
            elif position_ratio < 0.9:
                return 'DATA_ANALYSIS'
            else:
                return 'REPORT_GENERATION'

        elif workflow_type == 'pathology':
            if position_ratio < 0.3:
                return 'IMAGE_PROCESSING'
            elif position_ratio < 0.6:
                return 'ML_INFERENCE'
            elif position_ratio < 0.8:
                return 'DATA_ANALYSIS'
            else:
                return 'REPORT_GENERATION'

        else:  # 通用医疗工作流
            return random.choice(list(self.task_types.keys()))

    def _create_medical_task(self, task_id: int, task_type: str, dependencies: List[int]) -> MedicalTask:
        """创建医疗任务"""
        task_spec = self.task_types[task_type]

        # 随机生成任务属性
        computation = random.uniform(*task_spec['computation_range'])
        data_size = random.uniform(*task_spec['data_range'])
        memory = random.uniform(*task_spec['memory_range'])

        # 基于任务类型设置优先级
        priority_map = {
            'ML_INFERENCE': 5,
            'IMAGE_PROCESSING': 4,
            'DATA_ANALYSIS': 3,
            'VISUALIZATION': 2,
            'REPORT_GENERATION': 2,
            'DATABASE_QUERY': 1
        }
        priority = priority_map.get(task_type, 3)

        # 设置截止时间（基于计算需求）
        base_deadline = computation / 100  # 基础时间
        deadline = base_deadline * random.uniform(1.5, 3.0)

        return MedicalTask(
            task_id=task_id,
            task_type=task_type,
            computation_requirement=computation,
            data_size=data_size,
            memory_requirement=memory,
            deadline=deadline,
            priority=priority,
            dependencies=dependencies
        )

    def generate_batch_workflows(self, batch_size: int = 10) -> List[List[MedicalTask]]:
        """生成批量工作流"""
        workflows = []
        workflow_types = ['radiology', 'pathology', 'general']

        for _ in range(batch_size):
            workflow_type = random.choice(workflow_types)
            num_tasks = random.randint(5, 15)
            workflow = self.generate_workflow(num_tasks, workflow_type)
            workflows.append(workflow)

        return workflows

    def visualize_workflow(self, tasks: List[MedicalTask], save_path: str = None):
        """可视化工作流DAG"""
        import matplotlib.pyplot as plt

        # 创建图
        G = nx.DiGraph()

        # 添加节点和边
        for task in tasks:
            G.add_node(task.task_id,
                       label=f"T{task.task_id}\n{task.task_type[:3]}\n{task.computation_requirement:.0f}MIPS")

            for dep in task.dependencies:
                G.add_edge(dep, task.task_id)

        # 绘制图
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)

        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                               node_size=1000, alpha=0.7)

        # 绘制边
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                               arrows=True, arrowsize=20, alpha=0.6)

        # 绘制标签
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        plt.title("Medical Workflow DAG")
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# 使用示例
if __name__ == "__main__":
    generator = MedicalWorkflowGenerator()

    # 生成单个工作流
    workflow = generator.generate_workflow(num_tasks=10, workflow_type='radiology')

    print("Generated Medical Workflow:")
    for task in workflow:
        print(f"Task {task.task_id}: {task.task_type} "
              f"({task.computation_requirement:.0f} MIPS, "
              f"{task.data_size:.0f} MB, "
              f"Priority: {task.priority})")

    # 可视化工作流
    generator.visualize_workflow(workflow)