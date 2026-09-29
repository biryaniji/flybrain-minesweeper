"""
Export a trained model to ONNX for the browser (onnxruntime-web) and check that
ONNX outputs match PyTorch.

Usage:
  python export_onnx.py --arch cnn --checkpoint minesweeper_cnn.pt --out web_assets/flybrain.onnx

Input the browser must send:  "board"  float32 [batch, 4, 4]  (-2 = hidden, 0-8 = numbers)
Outputs:                       "logits" [batch, 16], "value" [batch, 1]
"""
import argparse
import os

import numpy as np
import torch

from brain import build_model
from environment import MinesweeperEnv


def export(model, dummy, out):
    kwargs = dict(
        input_names=["board"],
        output_names=["logits", "value"],
        dynamic_axes={"board": {0: "batch"}, "logits": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
    )
    try:
        # Newer PyTorch defaults to the dynamo exporter; the classic one is simpler here.
        torch.onnx.export(model, dummy, out, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, out, **kwargs)


def sample_boards(size, mines, n=64, seed=0):
    """Real mid-game boards to compare PyTorch vs ONNX on."""
    np.random.seed(seed)
    env = MinesweeperEnv(size=size, n_mines=mines)
    boards = []
    while len(boards) < n:
        state, done = env.reset(), False
        boards.append(state.copy())
        while not done and len(boards) < n:
            action = np.random.choice(np.flatnonzero(env.get_valid_actions()))
            state, _, done, _ = env.step(action)
            if not done:
                boards.append(state.copy())
    return np.stack(boards).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["cnn", "fly"], default="cnn")
    p.add_argument("--checkpoint", default="minesweeper_cnn.pt")
    p.add_argument("--out", default="web_assets/flybrain.onnx")
    p.add_argument("--size", type=int, default=4)
    p.add_argument("--mines", type=int, default=2)
    args = p.parse_args()

    model = build_model(args.arch, args.size, init="random")
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    export(model, torch.zeros(1, args.size, args.size), args.out)
    print(f"Exported -> {args.out} ({os.path.getsize(args.out) / 1024:.1f} KB)")

    # Verify
    import onnxruntime as ort
    boards = sample_boards(args.size, args.mines)
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    onnx_logits, onnx_value = sess.run(None, {"board": boards})
    with torch.no_grad():
        pt_logits, pt_value = model(torch.from_numpy(boards))

    max_diff = max(np.abs(onnx_logits - pt_logits.numpy()).max(),
                   np.abs(onnx_value - pt_value.numpy()).max())
    same_moves = (onnx_logits.argmax(1) == pt_logits.numpy().argmax(1)).mean()
    print(f"Max output difference: {max_diff:.2e} | same chosen cell on {same_moves * 100:.0f}% of boards")
    if max_diff > 1e-4:
        raise SystemExit("ONNX output does not match PyTorch. Do not ship this file.")
    print("ONNX export verified.")


if __name__ == "__main__":
    main()