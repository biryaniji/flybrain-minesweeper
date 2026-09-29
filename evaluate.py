"""
Evaluate a trained agent over many greedy games and compare it to a random clicker.

Usage:
  python evaluate.py --arch cnn --checkpoint minesweeper_cnn.pt
  python evaluate.py --arch fly --checkpoint agent_fly_connectome.pt
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from brain import build_model
from environment import MinesweeperEnv


def wilson_interval(wins, n, z=1.96):
    """95% confidence interval for a win rate."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def play_games(env, n_games, policy, model=None):
    wins, steps_all, steps_win = 0, [], []
    for _ in range(n_games):
        state = env.reset()
        done, won, steps = False, False, 0
        while not done:
            mask = env.get_valid_actions().astype(bool)
            if policy == "random":
                action = int(np.random.choice(np.flatnonzero(mask)))
            else:
                with torch.no_grad():
                    logits, _ = model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
                logits = logits.squeeze(0).masked_fill(~torch.as_tensor(mask), -1e9)
                action = int(torch.argmax(logits).item())
            state, _, done, won = env.step(action)
            steps += 1
        wins += int(won)
        steps_all.append(steps)
        if won:
            steps_win.append(steps)

    low, high = wilson_interval(wins, n_games)
    return {
        "games": n_games,
        "wins": wins,
        "win_rate": round(wins / n_games, 4),
        "win_rate_ci95": [round(low, 4), round(high, 4)],
        "avg_clicks": round(float(np.mean(steps_all)), 2),
        "avg_clicks_in_wins": round(float(np.mean(steps_win)), 2) if steps_win else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["cnn", "fly"], default="cnn")
    p.add_argument("--checkpoint", default="minesweeper_cnn.pt")
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--size", type=int, default=4)
    p.add_argument("--mines", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--unsafe-first-click", action="store_true",
                   help="evaluate under the old rules (first click can be a mine)")
    args = p.parse_args()

    model = build_model(args.arch, args.size, init="random")
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.eval()

    safe = not args.unsafe_first_click
    env = MinesweeperEnv(size=args.size, n_mines=args.mines, safe_first_click=safe)

    np.random.seed(args.seed)
    agent = play_games(env, args.games, "agent", model)
    np.random.seed(args.seed)  # same boards for the baseline
    baseline = play_games(env, args.games, "random")

    result = {
        "checkpoint": args.checkpoint,
        "arch": args.arch,
        "board": f"{args.size}x{args.size}, {args.mines} mines",
        "safe_first_click": safe,
        "agent_greedy": agent,
        "random_baseline": baseline,
    }

    os.makedirs("results", exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out = os.path.join("results", f"eval_{stem}{'' if safe else '_unsafe'}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()