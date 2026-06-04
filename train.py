import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from environment import UnoEnv
from memory import ReplayBuffer
from model import DQN


def train():
    # Detect Hardware
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Hardware Acceleration Active: Using Apple Silicon MPS (Metal).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Fallback Active: Using CPU.")

    # Set up MLflow Experiment with the SQLite fix
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("UNO_DQN_Training")

    env = UnoEnv()
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Hyperparameters
    learning_rate = 1e-3
    epsilon_start = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    batch_size = 64
    episodes = 5000
    gamma = 0.99

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayBuffer()

    # --- Metrics Tracking ---
    history_rewards = []
    history_wins = []

    print("Starting Training Loop...")

    # Start MLflow tracking run
    with mlflow.start_run(run_name="DQN_Base_Agent"):
        # Log all hyperparameters
        mlflow.log_params({
            "learning_rate": learning_rate,
            "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "batch_size": batch_size,
            "episodes": episodes,
            "gamma": gamma,
            "device": str(device),
        })

        epsilon = epsilon_start
        progress_bar = tqdm(range(episodes), desc="Learning", leave=True)

        for episode in progress_bar:
            state = env.reset()
            done = False

            ep_reward = 0
            ep_length = 0
            won_game = False
            ep_losses = []  # Track loss per episode

            while not done:
                mask = env.get_legal_mask()
                ep_length += 1

                if np.random.random() < epsilon:
                    legal_actions = np.where(mask == 1.0)[0]
                    action = np.random.choice(legal_actions)
                else:
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_values = policy_net(state_t).squeeze(0).cpu().numpy()

                    q_values = q_values + (1.0 - mask) * -1e9
                    action = np.argmax(q_values)

                next_state, reward, done, next_mask = env.step(action)
                memory.push(state, action, reward, next_state, done, mask, next_mask)

                ep_reward += reward

                if done and reward == 100.0:
                    won_game = True

                state = next_state

                # Optimization Sequence
                if len(memory.buffer) > batch_size:
                    (
                        b_state,
                        b_action,
                        b_reward,
                        b_next_state,
                        b_done,
                        b_mask,
                        b_next_mask,
                    ) = memory.sample(batch_size)

                    b_state = b_state.to(device)
                    b_action = b_action.to(device)
                    b_reward = b_reward.to(device)
                    b_next_state = b_next_state.to(device)
                    b_done = b_done.to(device)
                    b_mask = b_mask.to(device)
                    b_next_mask = b_next_mask.to(device)

                    q_values = (
                        policy_net(b_state).gather(1, b_action.unsqueeze(1)).squeeze(1)
                    )

                    with torch.no_grad():
                        next_q_values = target_net(b_next_state)
                        next_q_values = next_q_values + (1.0 - b_next_mask) * -1e9
                        max_next_q = next_q_values.max(1)[0]
                        expected_q_values = b_reward + (
                            gamma * max_next_q * (1.0 - b_done)
                        )

                    loss = nn.MSELoss()(q_values, expected_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    ep_losses.append(loss.item())

            # Log metrics to MLflow at the end of each episode
            avg_loss = np.mean(ep_losses) if ep_losses else 0.0

            # --- UNCOMMENTED: We NEED this for the RL dashboard! ---
            mlflow.log_metrics(
                {
                    "reward": ep_reward,
                    "win": 1 if won_game else 0,
                    "episode_length": ep_length,
                    "avg_loss": avg_loss,
                    "epsilon": epsilon,
                },
                step=episode,
            )

            # Update histories for local tqdm display
            history_rewards.append(ep_reward)
            history_wins.append(1 if won_game else 0)

            # Update Target Network
            if episode % 10 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Update the tqdm postfix stats every 50 episodes
            if episode % 50 == 0 and episode > 0:
                recent_win_rate = np.mean(history_wins[-50:]) * 100
                recent_reward = np.mean(history_rewards[-50:])
                progress_bar.set_postfix({
                    "Win %": f"{recent_win_rate:.1f}",
                    "Reward": f"{recent_reward:.1f}",
                    "Loss": f"{avg_loss:.2f}",
                })

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print("Training Cycle Finished. Saving model to MLflow and locally...")

        # Save locally for Streamlit
        torch.save(policy_net.state_dict(), "uno_agent.pth")

        # Log the PyTorch model artifact to MLflow
        mlflow.pytorch.log_model(policy_net, "model")
        # Log the local weights file as an artifact
        mlflow.log_artifact("uno_agent.pth")


if __name__ == "__main__":
    train()
