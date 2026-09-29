"""
Build a 16 -> 128 -> 16 weight matrix from a sample of real FlyWire synapses.

Fixes vs. the original:
  * Picks neurons so they are actually connected: the hidden layer is the
    strongest downstream targets of the 16 input neurons, and the outputs are
    the strongest targets of the hidden layer. Previously inputs/hidden/outputs
    were picked independently, so most of W1/W2 could end up zero.
  * Vectorised with pandas instead of looping over 50k rows with `in list`.
  * Weights are explicit synapse counts (the synapse table has no 'size'
    column, so the old code was silently using 1.0 per synapse anyway).
  * Saves the neuron IDs and density stats so the website can show them.

Note: the neurons are chosen by connectivity within this sample, not by cell
type. They are NOT specifically Kenyon cells or mushroom-body neurons.

Usage:
  python extract_connectome.py --limit 200000
"""
import argparse
import json

import torch
from caveclient import CAVEclient


def top_targets(edges, sources, exclude, k):
    sub = edges[edges.pre_pt_root_id.isin(sources) & ~edges.post_pt_root_id.isin(exclude)]
    return sub.groupby("post_pt_root_id")["count"].sum().nlargest(k).index.tolist()


def to_matrix(edges, pre_ids, post_ids):
    pre_index = {nid: i for i, nid in enumerate(pre_ids)}
    post_index = {nid: i for i, nid in enumerate(post_ids)}
    sub = edges[edges.pre_pt_root_id.isin(pre_index) & edges.post_pt_root_id.isin(post_index)]
    W = torch.zeros(len(post_ids), len(pre_ids))
    for pre, post, count in sub[["pre_pt_root_id", "post_pt_root_id", "count"]].itertuples(index=False):
        W[post_index[post], pre_index[pre]] = float(count)
    return W


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=200_000, help="synapses to download")
    p.add_argument("--n-in", type=int, default=16)
    p.add_argument("--n-hidden", type=int, default=128)
    p.add_argument("--n-out", type=int, default=16)
    p.add_argument("--out", default="fly_connectome.pt")
    args = p.parse_args()

    client = CAVEclient("flywire_fafb_public")
    print(f"Querying FlyWire for {args.limit:,} synapses (this can take a few minutes)...")
    syn = client.materialize.query_table("synapses_nt_v1", limit=args.limit)
    print(f"Downloaded {len(syn):,} synapses.")

    # One row per (pre, post) neuron pair with its synapse count; drop self-connections.
    edges = (syn.groupby(["pre_pt_root_id", "post_pt_root_id"]).size()
             .rename("count").reset_index())
    edges = edges[edges.pre_pt_root_id != edges.post_pt_root_id]

    inputs = (edges.groupby("pre_pt_root_id")["count"].sum()
              .nlargest(args.n_in).index.tolist())
    hidden = top_targets(edges, inputs, exclude=inputs, k=args.n_hidden)
    outputs = top_targets(edges, hidden, exclude=inputs + hidden, k=args.n_out)

    if len(hidden) < args.n_hidden or len(outputs) < args.n_out:
        raise SystemExit(
            f"Only found {len(hidden)} hidden and {len(outputs)} output neurons connected in this sample. "
            f"Re-run with a larger --limit."
        )

    W1 = to_matrix(edges, inputs, hidden)    # [128, 16]
    W2 = to_matrix(edges, hidden, outputs)   # [16, 128]

    stats = {
        "synapses_sampled": int(len(syn)),
        "W1_nonzero_fraction": round(float((W1 > 0).float().mean()), 4),
        "W2_nonzero_fraction": round(float((W2 > 0).float().mean()), 4),
        "W1_total_synapses": int(W1.sum()),
        "W2_total_synapses": int(W2.sum()),
    }

    torch.save({"W1": W1, "W2": W2}, args.out)
    with open("fly_connectome_meta.json", "w") as f:
        json.dump({
            "input_neuron_ids": [str(i) for i in inputs],
            "hidden_neuron_ids": [str(i) for i in hidden],
            "output_neuron_ids": [str(i) for i in outputs],
            **stats,
        }, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"\nSaved weights -> {args.out}")
    print("Saved neuron IDs + stats -> fly_connectome_meta.json")


if __name__ == "__main__":
    main()