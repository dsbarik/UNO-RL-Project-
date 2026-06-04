# Deep UNO Arena 🃏

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-%230194E2.svg?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-%23E92063.svg?style=flat&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)

Deep UNO Arena is a fully functional UNO game played against a Deep Q-Network (DQN) agent, wrapped in a polished, responsive, and visually appealing web interface built with Streamlit. The agent is trained using Reinforcement Learning to master UNO strategies, including handling action masks for illegal moves, drawing cards, and deploying wild cards.

## Features

* **Custom UNO Environment (`environment.py`)**: A fully simulated UNO game engine using OpenAI Gym-like interfaces. It supports state vectors, action masking, deck shuffling, discard piles, and game rules (Draw +2, Wild +4, Skip, Reverse).
* **Deep Q-Network Agent (`model.py`, `train.py`, `memory.py`)**: A PyTorch-based DQN trained to maximize the chance of winning. Uses a Replay Buffer for experience replay and dynamically masks illegal moves during predictions to enforce valid gameplay.
* **Interactive Web UI (`app.py`)**: A beautiful, custom-styled Streamlit application allowing a human player to compete against the trained AI agent in real time.
* **Experiment Tracking**: Training metrics (Win rate, Episode Length, Loss, Reward) are logged continuously via MLflow.

## Project Structure

* `environment.py`: Defines the `Deck`, `Card`, and `UnoEnv` classes. It manages the physical deck, hand tracking, action masking (61 possible actions), and the reward functions for RL.
* `model.py`: Contains the `DQN` PyTorch model (Multi-Layer Perceptron: 113 input nodes -> 128 -> 128 -> 61 output nodes).
* `memory.py`: Implements the `ReplayBuffer` for experience replay during training.
* `train.py`: The main training loop. It sets up the MLflow experiment, initializes the network, explores/exploits actions, optimizes the loss function, and saves the trained agent.
* `app.py`: The Streamlit frontend. It loads the PyTorch model (`uno_agent.pth`) and renders the interactive Deep UNO Arena game.
* `main.py`: A simple testing script to verify deck logic.

## Setup and Installation

### Prerequisites

* Python 3.12+
* `uv` or `pip`

### Installation

Clone the repository and install the dependencies:

```bash
# If using uv
uv sync

# If using pip
pip install -e .
```

## Usage

### 1. Training the Agent

If you want to train your own agent from scratch, run the training script:

```bash
python train.py
```

This will run for 5000 episodes by default and save the best model weights to `uno_agent.pth`. You can stop the training process by pressing `Ctrl + C`.

**Monitoring Training**:
The training script integrates with MLflow. To view the training metrics dashboard (e.g., win rates and loss), open another terminal and run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then navigate to `http://localhost:5000` in your browser.

### 2. Playing the Game

To challenge the trained neural network, start the Streamlit web application:

```bash
streamlit run app.py
```

A browser window will open automatically, presenting the Deep UNO Arena. Good luck!

## Technologies Used

* **PyTorch**: Deep learning and model training.
* **Streamlit**: Web interface and frontend logic.
* **MLflow**: Training metric tracking and model versioning.
* **NumPy**: Vectorized state management for RL.
* **Pydantic**: Robust data models for Cards and Deck.
