# Two-tier DDPG-based Scheduling for MIoT Workflows

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the paper: **"Cloud-Fog-Edge Collaborative Computing For Sequential MIoT Workflow: A Two-tier DDPG-based Scheduling Framework"**.

This work addresses the NP-hard problem of scheduling sequential Medical Internet of Things (MIoT) workflows across a heterogeneous cloud-fog-edge infrastructure. Our primary objective is to minimize the total execution time (makespan) while considering computational, memory, and communication constraints. We propose a novel scheduling framework based on a **Two-tier Deep Deterministic Policy Gradient (DDPG)** algorithm that learns an adaptive scheduling policy through interaction with a simulated environment.

## Framework Overview

Our proposed system operates on a three-layer computing architecture, designed to handle the diverse requirements of MIoT workflows. The scheduling itself is managed by a two-tier reinforcement learning agent.

### 1. Three-Layer Computing Infrastructure
The physical system is modeled as a hierarchical infrastructure comprising:
- **Cloud Layer**: A centralized tier with powerful computing resources (e.g., high-performance GPUs), ideal for computationally intensive tasks but subject to high communication latency.
- **Fog Layer**: An intermediate tier with moderate computing power, offering a balance between computational capacity and proximity to the data source.
- **Edge Layer**: A distributed tier closest to the MIoT devices (e.g., FPGAs), providing ultra-low latency for real-time processing but with limited resources.

![System Architecture](image/Colin_system(alin).pdf)
*Fig. 1: The two-tier DDPG-based scheduling framework for processing sequential MIoT task workflows in cloud-fog-edge infrastructures.*

### 2. Two-tier DDPG Scheduler
To effectively manage the trade-offs within this architecture, we designed a hierarchical DRL agent that decouples the scheduling decision into two levels:
- **Global Controller**: This agent is responsible for high-level, coarse-grained decisions. It observes the overall system state and selects the most appropriate execution layer (Cloud, Fog, or Edge) for a given task.
- **Local Controllers**: A set of specialized agents, one for each layer. Once the Global Controller selects a layer, the corresponding Local Controller is activated to make a fine-grained decision, assigning the task to a specific node within that layer.

This hierarchical structure allows the agent to learn specialized policies for inter-layer and intra-layer resource management, significantly improving learning efficiency and scalability.

## Methodology

We formulate the workflow scheduling problem as a **Markov Decision Process (MDP)** and solve it using our Two-tier DDPG framework.

### MDP Formulation
- **State Space ($\mathcal{S}$)**: The state is hierarchically structured.
  - **Global State ($S_{\text{global}}$)**: Captures aggregated statistics for each layer, such as average computational load, total available memory, and inter-layer network congestion. It also includes the characteristics of the current task to be scheduled.
  - **Local State ($S_{\text{local}}^{(\ell)}$)**: Provides fine-grained, node-specific features for each layer $\ell$, including individual node residual memory, expected finish times of queued tasks, and current utilization rates.

- **Action Space ($\mathcal{A}$)**: An action $a_t$ is a tuple $(\ell, n)$, representing the selection of a layer $\ell$ by the Global Controller and a node $n$ within that layer by the Local Controller. The action space is dynamically masked to ensure memory feasibility.

- **Reward Function ($\mathcal{R}$)**: The reward function is designed to guide the agent towards minimizing the makespan. The primary component is a penalty proportional to the task's total time cost ($T_{\text{cost}}$). Auxiliary terms are included to provide a bonus for successful workflow completion and to reward efficient resource utilization (e.g., load balancing).
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

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **(Optional) GPU Setup:**
    If you have a compatible NVIDIA GPU, you can use the provided script to help set up TensorFlow with GPU support. Please ensure your CUDA and cuDNN versions are compatible.
    ```bash
    python setup_tensorflow_gpu.py
    ```

## Usage

### Training a New Model
To train the Two-tier DDPG agent, run the `train.py` script with a specified configuration file.

```bash
python src/train.py --config experiments/config_L3.yaml --log_dir ./training_logs
```
- `--config`: Path to the experiment configuration file, which defines infrastructure, workload, and hyperparameters.
- `--log_dir`: Directory to save training logs and model checkpoints.

### Evaluating a Trained Model
To evaluate a trained policy and compare it against baseline algorithms (HEFT, Greedy, etc.), use the `evaluate.py` script.

```bash
python src/evaluate.py --model_path models/two_tier_ddpg_final.h5 --test_data_level L4
```
- `--model_path`: Path to the saved Keras/TensorFlow model file.
- `--test_data_level`: The difficulty level (L1-L4) of the test workflows to generate.

## Experimental Results

Our framework was extensively evaluated against four baseline algorithms: **HEFT**, **Greedy**, **FCFS**, and **Random**. The results demonstrate the superiority of our Two-tier DDPG approach, especially in complex and communication-intensive scenarios.

### Training Convergence
The agent demonstrates stable learning, with the episodic reward consistently increasing and the workflow makespan decreasing, converging to a near-optimal policy around 400 episodes.

![Training Curves](image/training_curves_side_by_side.png)

### Makespan Comparison
Our approach significantly outperforms naive baselines and the Greedy heuristic. As workflow complexity increases, the performance gap to the powerful offline HEFT benchmark narrows, showcasing the adaptiveness of our online DRL-based scheduler.

![Makespan Comparison](image/makespan_total_lines_all_algorithms.png)

## Citation
If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{fu2024two-tier,
  title={Cloud-Fog-Edge Collaborative Computing For Sequential MIoT Workflow: A Two-tier DDPG-based Scheduling Framework},
  author={Fu, Yuhao and Liu, Yalin and Tao, Bishenghui and Ruan, Junhong},
  booktitle={Lecture Notes in Computer Science},
  year={2024},
  organization={Springer}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
This work was supported in part by the grant from the Research Grants Council of the Hong Kong Special Administrative Region, China, under project No. UGC/FDS16/E15/24, and in part by the UGC Research Matching Grant Scheme under Project No.: 2024/3003.
