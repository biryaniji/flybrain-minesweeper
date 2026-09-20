.

🪰 FlyBrain Minesweeper: Spatial RL in 3D
An end-to-end Reinforcement Learning system where an autonomous agent learns to deduce and clear Minesweeper boards using spatial convolutional representations, brought to life through a real-time 3D simulation environment.

💡 Overview
Minesweeper is fundamentally a game of spatial constraint satisfaction and probabilistic reasoning. While random guessing yields a meager baseline win rate (~19% on dense configurations), FlyBrain leverages a Convolutional Actor-Critic network (PPO-trained) to discover local neighborhood heuristics, identify safe clusters, and flag hidden hazards.

Rather than confining the agent to static terminal logs, the policy is integrated into an interactive 3D studio environment powered by Panda3D/Ursina, featuring a custom animated agent physically evaluating and sweeping tiles in real time.

🧠 System Architecture
Observation Tensor: The board state is fed as a multi-channel 2D spatial grid (revealed numbers, unrevealed tiles, boundaries).

Spatial Feature Extractor: Two-stage 2D convolutions with small receptive fields preserve spatial locality and capture neighborhood risk topologies.

Actor-Critic Decoupling:

Actor Head: Outputs spatial action logits masked dynamically to prevent illegal selections.

Critic Head: Estimates expected future discounted state value V(s) to guide policy gradient stability.

Agentic Visualizer: An interactive 3D physics viewport rendering dynamic camera framing, real-time wing-flap kinematics, and tile manipulation.

📂 Project Structure
Plaintext
flybrain-minesweeper/
├── brain.py             # Actor-Critic CNN architecture (PyTorch)
├── environment.py       # Custom OpenAI Gym-compatible Minesweeper engine
├── main.py              # PPO training loop & experience buffer
├── play.py              # Lightweight headless/terminal evaluation harness
├── visual_3d.py         # Full 3D interactive Ursina visualization runtime
├── minesweeper_cnn.pt   # Pre-trained actor-critic checkpoint
└── requirements.txt     # Environment dependencies
🚀 Quick Start
1. Installation
Clone the repository and install the runtime requirements:

Bash
git clone https://github.com/biryaniji/flybrain-minesweeper.git
cd flybrain-minesweeper
pip install -r requirements.txt
(Dependencies: torch, numpy, ursina)

2. Launch the 3D Interactive Agent
Run the visual playthrough using your local model weights:

Bash
python visual_3d.py
🎮 Viewport Controls
Input	Action
Right-Click + Drag	Orbit around the 3D board
Scroll Wheel	Zoom in / out
Space / Escape	Pause / Exit simulation
📊 Performance & Behavior
Baseline (Random Policy): ~19% completion rate.

FlyBrain Policy: Converges to consistent deduction patterns, correctly resolving safe adjacency loops and clearing boards with ~50–70% efficiency on dense 4x4 grids.

Dynamic Animations: Instant visual feedback for deduction chains, tile clearances, safe sweeps (green state), and detonate failures (inverted flight state).

🛠️ Built With
PyTorch — Deep neural network training & tensor operations

Ursina Engine — Lightweight 3D rendering pipeline

NumPy — Matrix manipulations & board state generation
