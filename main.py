import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# 1. Correctly importing your specific file and class
from environment import MinesweeperEnv 
from brain import MineSweeperCNN

def train():
    grid_size = 4
    
    # 2. Initializing the environment
    env = MinesweeperEnv() 
    model = MineSweeperCNN(grid_size=grid_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5000
    gamma = 0.99
    wins = 0
    
    print("Initiating CNN PPO Training...")
    
    for episode in range(1, epochs + 1):
        state = env.reset()
        
        log_probs = []
        values = []
        rewards = []
        masks = []
        
        total_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state)
            action_logits, state_value = model(state_tensor)
            
            valid_actions = env.get_valid_actions()
            action_logits = action_logits.squeeze(0)
            action_logits[~torch.tensor(valid_actions, dtype=torch.bool)] = -1e9
            
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            
            next_state, reward, done, won = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            values.append(state_value)
            rewards.append(reward)
            masks.append(1 - int(done))
            
            state = next_state
            total_reward += reward
            if won: wins += 1
            
        returns = []
        discounted_sum = 0
        for r, m in zip(reversed(rewards), reversed(masks)):
            discounted_sum = r + (gamma * discounted_sum * m)
            returns.insert(0, discounted_sum)
            
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        values = torch.cat(values).squeeze()
        if values.dim() == 0: values = values.unsqueeze(0)
        log_probs = torch.stack(log_probs)
        
        advantages = returns - values.detach()
        
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + 0.5 * critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 10 == 0:
            win_rate = (wins / 10) * 100
            print(f"Episode: {episode:4d} | Total Reward: {total_reward:5.1f} | Win Rate: {win_rate:4.1f}%")
            wins = 0

 # Save the trained model weights
    torch.save(model.state_dict(), 'minesweeper_cnn.pt')
    print("\nTraining complete! Model saved as 'minesweeper_cnn.pt'")

if __name__ == "__main__":
    train()
  