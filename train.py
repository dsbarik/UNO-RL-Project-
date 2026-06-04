import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import UnoEnv
from memory import ReplayBuffer
from model import DQN


def train():
    # Detect Hardware (Apple Silicon MPS, CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Hardware Acceleration Active: Using Apple Silicon MPS (Metal).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Fallback Active: Using CPU.")

    env = UnoEnv()
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = ReplayBuffer()

    epsilon = 1.0
    batch_size = 32
    episodes = 1000

    print("Starting Training Loop...")

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            mask = env.get_legal_mask()

            if np.random.random() < epsilon:
                # Pick a random LEGAL action
                legal_actions = np.where(mask == 1.0)[0]
                action = np.random.choice(legal_actions)
            else:
                # Let the AI choose the best LEGAL action
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = policy_net(state_t).squeeze(0).cpu().numpy()

                # Apply mask to penalize illegal moves heavily
                q_values = q_values + (1.0 - mask) * -1e9
                action = np.argmax(q_values)

            next_state, reward, done, next_mask = env.step(action)
            memory.push(state, action, reward, next_state, done, mask, next_mask)
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

                # Push tensors to GPU
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
                    expected_q_values = b_reward + (0.99 * max_next_q * (1.0 - b_done))

                loss = nn.MSELoss()(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update Target Network and Decay Epsilon
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {episode}/{episodes} processed.")

        epsilon = max(0.1, epsilon * 0.995)

    print("Training Cycle Finished. Saving policy state matrix...")
    torch.save(policy_net.state_dict(), "uno_agent.pth")


if __name__ == "__main__":
    train()
