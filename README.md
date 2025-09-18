# Two-tier DDPG-based Scheduling for Sequential MIoT Workflows

[![Python 3.8+]]
[![TensorFlow 2.x]]
[![License: MIT]]

This repository provides the official implementation for the paper: **"Cloud-Fog-Edge Collaborative Computing For Sequential MIoT Workflow: A Two-tier DDPG-based Scheduling Framework"**.

This research addresses the NP-hard challenge of scheduling sequential Medical Internet of Things (MIoT) workflows across a heterogeneous cloud-fog-edge infrastructure. The primary objective is to minimize the total execution time (makespan) while respecting the system's computational, memory, and communication constraints. We introduce a novel scheduling framework leveraging a **Two-tier Deep Deterministic Policy Gradient (DDPG)** algorithm, which learns an adaptive scheduling policy through direct interaction with a simulated environment.

## Framework Overview

Our proposed system is built upon a three-layer computing architecture tailored for the diverse demands of MIoT workflows. The scheduling intelligence is driven by a two-tier reinforcement learning agent, which mirrors the physical hierarchy of the infrastructure.

### 1. Three-Layer Computing Infrastructure
The system architecture is composed of three distinct layers:
- **Cloud Layer**: A centralized tier with powerful computing resources (e.g., high-performance GPUs), suited for the most computationally demanding tasks but incurring the highest communication latency.
- **Fog Layer**: An intermediate tier that offers a strategic balance between significant computational power and closer proximity to the network edge.
- **Edge Layer**: A distributed tier located closest to the MIoT devices (e.g., FPGAs), providing ultra-low latency for real-time processing tasks, albeit with limited resources.

![System Architecture](src/environment/figures/system_framework.png)
*Fig. 1: The two-tier DDPG-based scheduling framework for processing sequential MIoT task workflows in cloud-fog-edge infrastructures.*

### 2. Two-tier DDPG Scheduler
To effectively navigate the trade-offs in this multi-layered system, we designed a hierarchical DRL agent that decomposes the complex scheduling decision into two distinct levels:
- **Global Controller (Meta-Controller)**: This agent makes high-level, coarse-grained decisions. It observes the global system state and selects the optimal execution layer (Cloud, Fog, or Edge) for an incoming task.
- **Local Controllers (Sub-Controllers)**: A set of specialized agents, one for each computing layer. Once the Global Controller has chosen a layer, the corresponding Local Controller is activated to perform a fine-grained assignment, selecting the best-suited node within that layer.

This hierarchical decomposition allows the agent to learn specialized policies for both inter-layer and intra-layer resource management, which significantly enhances the learning efficiency and scalability of the scheduler.

## Methodology

We formulate the sequential workflow scheduling problem as a **Markov Decision Process (MDP)** and employ our Two-tier DDPG framework to find an optimal scheduling policy.

### MDP Formulation
- **State Space ($\mathcal{S}$)**: The state representation is hierarchically structured.
  - **Global State ($S_{\text{global}}$)**: An aggregate view of the system, capturing key statistics for each layer, such as average computational load, total available memory, and inter-layer network congestion. It also includes the characteristics of the current task awaiting scheduling.
  - **Local State ($S_{\text{local}}^{(\ell)}$)**: A detailed, node-specific feature set for each layer $\ell$, encompassing individual node residual memory, expected finish times of queued tasks, and current utilization rates.

- **Action Space ($\mathcal{A}$)**: A scheduling action $a_t$ is a composite tuple $(\ell, n)$, representing the layer selection $\ell$ by the Global Controller and the node selection $n$ by the corresponding Local Controller. The action space is dynamically masked at the local level to exclude nodes that do not meet the task's memory requirements.

- **Reward Function ($\mathcal{R}$)**: The reward function is engineered to guide the agent toward minimizing the workflow makespan. Its primary component is a penalty proportional to the task's total time cost ($T_{\text{cost}}$). This is supplemented with auxiliary terms, including a bonus for successful workflow completion and a shaping reward for promoting efficient resource utilization (e.g., load balancing).
  $$ R_t = -T_{\text{cost}}(v_t, n) + \beta_1 \cdot R_{\text{bonus}} + \beta_2 \cdot R_{\text{eff}} $$

## Repository Structure

```
ddpg_scheduling/
├── src/
│   ├── algorithms/
│   │   ├── hd_ddpg.py            # Main Hierarchical DDPG algorithm implementation.
│   │   ├── meta_controller.py    # Global Controller (Actor-Critic network).
│   │   └── sub_controllers.py    # Local Controllers (one for each layer).
│   ├── environment/
│   │   ├── medical_simulator.py  # Core discrete-event simulation engine.
│   │   ├── fog_cloud_env.py      # OpenAI Gym-style environment wrapper.
│   │   └── workflow_generator.py # Generates synthetic sequential MIoT workflows.
│   └── utils/
│       ├── replay_buffer.py      # Implementation of the experience replay buffer.
│       └── metrics.py            # Utility functions for performance tracking.
├── experiments/
│   └── train_hd_ddpg.py          # Main script for training the DRL agent.
├── models/                       # Directory for saving trained model checkpoints.
├── results/                      # Directory for storing experiment logs and plots.
└── requirements.txt              # Python package dependencies.
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/perram2000/ddpg_scheduling.git
    cd ddpg_scheduling
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Scheduler
To train the Two-tier DDPG agent, execute the main training script from the `experiments` directory. You can specify experiment parameters, such as workload complexity and infrastructure settings, via command-line arguments or a configuration file.

```bash
python experiments/train_hd_ddpg.py --workload_level L3 --log_dir ./results/training_logs
```
- `--workload_level`: Specifies the difficulty of the training workflows (e.g., L1, L2, L3, L4).
- `--log_dir`: The directory where training logs and model checkpoints will be saved.

### Evaluating a Trained Model
To evaluate a pre-trained policy and compare its performance against the baseline algorithms (HEFT, Greedy, FCFS, Random), you can run the evaluation script (assuming `evaluate.py` is created in `experiments`).

```bash
python experiments/evaluate.py --model_path models/two_tier_ddpg_final.h5 --test_data_level L4
```
- `--model_path`: Path to the saved model file (`.h5` or other TensorFlow format).
- `--test_data_level`: The difficulty level of the test workflows to be generated for evaluation.

## Experimental Results

Our framework was rigorously evaluated against four baseline algorithms: **HEFT**, **Greedy**, **FCFS**, and **Random**. The results confirm the effectiveness of our Two-tier DDPG approach, particularly in complex and communication-heavy scenarios.

### Training Convergence
The agent exhibits stable and efficient learning. The episodic reward shows a consistent upward trend, while the workflow makespan correspondingly decreases, converging to a near-optimal policy after approximately 400 episodes.

![Training Curves](results/figures/training_curves.png)

### Makespan Performance
Our approach significantly outperforms naive baselines and the Greedy heuristic. As workflow complexity increases, the performance gap between our online DRL scheduler and the powerful offline HEFT benchmark narrows, highlighting the adaptiveness and strength of our method in dynamic environments.

![Makespan Comparison](results/figures/makespan_comparison.png)

## Citation

If you use this code or find our work helpful in your research, please cite our paper:

```bibtex
@inproceedings{fu2024two-tier,
  title={Cloud-Fog-Edge Collaborative Computing For Sequential MIoT Workflow: A Two-tier DDPG-based Scheduling Framework},
  author={Fu, Yuhao and Liu, Yalin and Tao, Bishenghui and Ruan, Junhong},
  booktitle={Lecture Notes in Computer Science},
  year={2024},
  organization={Springer},
  note={\url{https://github.com/perram2000/ddpg_scheduling}}
}
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements
This work was supported in part by the grant from the Research Grants Council of the Hong Kong Special Administrative Region, China, under project No. UGC/FDS16/E15/24, and in part by the UGC Research Matching Grant Scheme under Project No.: 2024/3003.
