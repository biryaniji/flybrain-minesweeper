import torch
import time
import numpy as np
from environment import MinesweeperEnv
from brain import MineSweeperCNN

print("Script started...", flush=True)

def render_board(env):
    obs = env.get_observation()
    size = env.size
    print("   " + " ".join([str(i) for i in range(size)]))
    print("  " + "-" * (size * 2 + 1))
    for r in range(size):
        row_str = f"{r} |"
        for c in range(size):
            val = obs[r, c]
            if val == -2:
                row_str += " . "
            elif val == 0:
                row_str += "   "
            else:
                row_str += f" {int(val)} "
        print(row_str)
    print()

def play_visual_game():
    grid_size = 4
    env = MinesweeperEnv(size=grid_size, n_mines=2)
    model = MineSweeperCNN(grid_size=grid_size)
    
    try:
        model.load_state_dict(torch.load('minesweeper_cnn.pt', weights_only=True))
        model.eval()
        print("Loaded 'minesweeper_cnn.pt' successfully!", flush=True)
    except FileNotFoundError:
        print("Error: 'minesweeper_cnn.pt' not found in this folder!", flush=True)
        return

    state = env.reset()
    done = False
    step_count = 0
    
    print("\n=== STARTING AI VISUAL PLAYTHROUGH ===", flush=True)
    print("Legend: [ . ] = Hidden Cell | [ Numbers ] = Safe Clues\n", flush=True)
    render_board(env)
    
    while not done:
        state_tensor = torch.FloatTensor(state)
        
        with torch.no_grad():
            action_logits, state_value = model(state_tensor)
        
        valid_actions = env.get_valid_actions()
        action_logits = action_logits.squeeze(0)
        action_logits[~torch.tensor(valid_actions, dtype=torch.bool)] = -1e9
        
        action = torch.argmax(action_logits).item()
        r, c = divmod(action, grid_size)
        
        print(f"Step {step_count + 1}: AI selects cell -> Row {r}, Column {c}", flush=True)
        
        state, reward, done, won = env.step(action)
        render_board(env)
        step_count += 1
        
        time.sleep(1.5)
        
        if done:
            if won:
                print("🎉 VICTORY! The CNN successfully cleared the board!", flush=True)
            else:
                print("💥 BOOM! The AI triggered a mine.", flush=True)

if __name__ == "__main__":
    play_visual_game()