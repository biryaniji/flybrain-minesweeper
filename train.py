"""
PPO training for the Minesweeper agent.

What makes this actual PPO (the old main.py was a one-update-per-episode
actor-critic):
  * collects a fixed-size rollout across many episodes
  * GAE advantages
  * clipped probability-ratio objective against the rollout policy
  * several epochs of minibatch updates on each rollout
  * entropy bonus, grad clipping, early stop on KL

Usage:
  python train.py --arch cnn
  python train.py --arch fly --init connectome
  python train.py --arch fly --init shuffled
  python train.py --arch fly --init random
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from brain import build_model
from environment import MinesweeperEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["cnn", "fly"], default="cnn")
    p.add_argument("--init", choices=["random", "connectome", "shuffled"], default="random",
                   help="weight init for --arch fly")
    p.add_argument("--connectome", default="fly_connectome.pt")
    p.add_argument("--freeze", action="store_true", help="freeze connectome weights (fly only)")
    p.add_argument("--size", type=int, default=4)
    p.add_argument("--mines", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--rollout-steps", type=int, default=1024)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--out", default=None, help="checkpoint path (default depends on arch/init)")
    return p.parse_args()


def run_name(args):
    return "cnn" if args.arch == "cnn" else f"fly_{args.init}"


def masked_dist(logits, mask):
    # mask: True = clickable. Illegal cells get ~zero probability.
    return Categorical(logits=logits.masked_fill(~mask, -1e9))


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    name = run_name(args)
    out_path = args.out or ("minesweeper_cnn.pt" if args.arch == "cnn" else f"agent_{name}.pt")
    os.makedirs("logs", exist_ok=True)
    csv_path = os.path.join("logs", f"{name}_seed{args.seed}.csv")

    env = MinesweeperEnv(size=args.size, n_mines=args.mines)
    model = build_model(args.arch, args.size, args.init, args.connectome, args.freeze, args.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, eps=1e-5)

    n_cells = args.size * args.size
    T = args.rollout_steps
    obs_buf = torch.zeros(T, args.size, args.size)
    mask_buf = torch.zeros(T, n_cells, dtype=torch.bool)
    act_buf = torch.zeros(T, dtype=torch.long)
    logp_buf = torch.zeros(T)
    val_buf = torch.zeros(T)
    rew_buf = torch.zeros(T)
    done_buf = torch.zeros(T)

    num_updates = max(1, args.total_steps // T)
    state = env.reset()
    ep_return, ep_len = 0.0, 0
    global_step, total_episodes = 0, 0
    start = time.time()

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["update", "global_step", "episodes", "win_rate", "mean_return", "mean_length",
                     "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"])

    print(f"Training '{name}' with PPO for {num_updates} updates ({num_updates * T:,} steps)")

    for update in range(1, num_updates + 1):
        # Linear learning-rate annealing
        frac = 1.0 - (update - 1) / num_updates
        for g in optimizer.param_groups:
            g["lr"] = frac * args.lr

        # ------------------------------------------------ 1. collect rollout
        model.eval()
        ep_returns, ep_wins, ep_lens = [], [], []
        for t in range(T):
            obs = torch.as_tensor(state, dtype=torch.float32)
            mask = torch.as_tensor(env.get_valid_actions(), dtype=torch.bool)
            with torch.no_grad():
                logits, value = model(obs.unsqueeze(0))
                dist = masked_dist(logits.squeeze(0), mask)
                action = dist.sample()

            next_state, reward, done, won = env.step(action.item())

            obs_buf[t] = obs
            mask_buf[t] = mask
            act_buf[t] = action
            logp_buf[t] = dist.log_prob(action)
            val_buf[t] = value.squeeze()
            rew_buf[t] = reward
            done_buf[t] = float(done)

            ep_return += reward
            ep_len += 1
            global_step += 1

            if done:
                ep_returns.append(ep_return)
                ep_wins.append(int(won))
                ep_lens.append(ep_len)
                ep_return, ep_len = 0.0, 0
                state = env.reset()
            else:
                state = next_state

        total_episodes += len(ep_returns)

        # ------------------------------------------------ 2. GAE advantages
        with torch.no_grad():
            _, last_value = model(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
            last_value = last_value.squeeze()

        advantages = torch.zeros(T)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else val_buf[t + 1]
            next_nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + args.gamma * next_value * next_nonterminal - val_buf[t]
            last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf

        # ------------------------------------------------ 3. PPO update
        model.train()
        idx = np.arange(T)
        clip_fracs, pg_losses, v_losses, entropies = [], [], [], []
        approx_kl = torch.tensor(0.0)

        for epoch in range(args.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, T, args.minibatch_size):
                mb = idx[s:s + args.minibatch_size]

                logits, values = model(obs_buf[mb])
                dist = masked_dist(logits, mask_buf[mb])
                new_logp = dist.log_prob(act_buf[mb])
                entropy = dist.entropy().mean()

                log_ratio = new_logp - logp_buf[mb]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs.append(((ratio - 1.0).abs() > args.clip).float().mean().item())

                mb_adv = advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - args.clip, 1 + args.clip),
                ).mean()
                v_loss = 0.5 * ((values.squeeze(-1) - returns[mb]) ** 2).mean()
                loss = pg_loss - args.ent_coef * entropy + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.item())

            if args.target_kl and approx_kl.item() > args.target_kl:
                break

        # ------------------------------------------------ 4. log
        win_rate = float(np.mean(ep_wins)) if ep_wins else 0.0
        mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        mean_len = float(np.mean(ep_lens)) if ep_lens else 0.0
        writer.writerow([update, global_step, total_episodes, round(win_rate, 4), round(mean_ret, 3),
                         round(mean_len, 3), round(float(np.mean(pg_losses)), 5),
                         round(float(np.mean(v_losses)), 5), round(float(np.mean(entropies)), 5),
                         round(approx_kl.item(), 5), round(float(np.mean(clip_fracs)), 4)])
        csv_file.flush()

        if update % 10 == 0 or update == 1 or update == num_updates:
            print(f"update {update:4d}/{num_updates} | steps {global_step:8,d} | "
                  f"win rate {win_rate * 100:5.1f}% | return {mean_ret:6.2f} | "
                  f"entropy {np.mean(entropies):.3f} | {time.time() - start:6.0f}s")

    csv_file.close()
    torch.save(model.state_dict(), out_path)
    with open(os.path.join("logs", f"{name}_seed{args.seed}_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nSaved model  -> {out_path}")
    print(f"Saved curve  -> {csv_path}")


if __name__ == "__main__":
    train(parse_args())