import argparse
import itertools
import json
import random
from pathlib import Path

import chess
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from encoder.lc0_hf_bt5.hf_model import _encode_classical_112_planes, Lc0Bt4HFModel


# 13 classes: 0=empty, 1-6=white (pawn..king), 7-12=black (pawn..king)
NUM_CLASSES = 13
HIDDEN_SIZE = 1024
_PIECE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
CLASS_NAMES = ["empty"] + [f"W {p}" for p in _PIECE_NAMES] + [f"B {p}" for p in _PIECE_NAMES]


class PieceProbe(nn.Module):
    def __init__(self, layer_idx: int | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.linear = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# 25 classes: joint encoding of (white_attackers, black_attackers) capped at 4.
# class = min(n_white, 4) * 5 + min(n_black, 4)
ATTACK_NUM_CLASSES = 25
_BUCKET_NAMES = ["0", "1", "2", "3", "4+"]


class AttackProbe(nn.Module):
    def __init__(self, layer_idx: int | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.linear = nn.Linear(HIDDEN_SIZE, ATTACK_NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PositionDataset(Dataset):
    def __init__(self, triples: list):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        start_fen, move_list, end_fen = self.triples[idx]
        planes = _encode_classical_112_planes(start_fen, move_list)
        return planes, end_fen


def collate_fn(batch):
    planes = torch.stack([item[0] for item in batch])
    end_fens = [item[1] for item in batch]
    return planes, end_fens


def load_and_split(jsonl_path: str, n_positions: int | None, val_size: int, seed: int):
    with open(jsonl_path) as f:
        triples = [json.loads(line) for line in f]
    rng = random.Random(seed)
    rng.shuffle(triples)

    # Build a balanced val set: val_size//2 white-to-move, val_size//2 black-to-move.
    # Positions that don't fit the balance are discarded (not added to train).
    # Everything after the point where val is full becomes the training set.
    half = val_size // 2
    val_white, val_black = [], []
    cutoff = len(triples)
    for i, triple in enumerate(triples):
        is_white = triple[2].split()[1] == 'w'
        if is_white and len(val_white) < half:
            val_white.append(triple)
        elif not is_white and len(val_black) < half:
            val_black.append(triple)
        if len(val_white) == half and len(val_black) == half:
            cutoff = i + 1
            break

    val_triples = val_white + val_black
    train_triples = triples[cutoff:]
    if n_positions is not None:
        train_triples = train_triples[:n_positions]
    return train_triples, val_triples


def parse_args():
    parser = argparse.ArgumentParser(description="Train a linear probe on LC0 hidden states.")
    parser.add_argument("--jsonl", required=True, help="Path to positions JSONL file.")
    parser.add_argument("--lc0-weights", required=True, help="Path to LC0 HF checkpoint directory.")
    parser.add_argument("--layer-idx", type=int, required=True, help="Which hidden state layer to probe (0=embedding, 1-15=encoder layers).")
    parser.add_argument("--n-positions", type=int, default=None, help="Max training positions to use (default: all remaining after val).")
    parser.add_argument("--val-size", type=int, default=100, help="Val set size: val_size//2 white-to-move + val_size//2 black-to-move positions.")
    parser.add_argument("--train-k", type=int, default=8, help="Squares sampled per train position (default: 8).")
    parser.add_argument("--val-k", type=int, default=64, help="Squares sampled per val position (default: 64, i.e. all squares).")
    parser.add_argument("--train-bs", type=int, default=32, help="Positions per train step (default: 32, giving 32*8=256 probe examples).")
    parser.add_argument("--val-bs", type=int, default=4, help="Positions per val step (default: 4, giving 4*64=256 probe examples).")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100, help="Log train loss every N steps.")
    parser.add_argument("--eval-every", type=int, default=1000, help="Run val eval every K steps.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes for parallel plane encoding.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="probe_outputs", help="Root output directory. A subdirectory layer{idx}_epochs{n}/ is created automatically.")
    parser.add_argument("--probe-type", choices=["piece", "attack"], default="piece", help="Which probe to train: piece (13-class piece identity) or attack (25-class attack counts).")
    parser.add_argument("--run-name", required=True, help="Prefix for the output directory, e.g. 'piece_balanced'.")
    parser.add_argument("--no-class-weights", action="store_false", dest="class_weights", help="Disable inverse-frequency class weighting in the loss (default: enabled).")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience: stop after this many evals with no improvement (default: 5).")
    return parser.parse_args()

_FLIP_IDX = [sq ^ 56 for sq in range(64)]


def board_to_labels(board: chess.Board) -> torch.Tensor:
    labels = torch.zeros(64, dtype=torch.long)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            labels[sq] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
    return labels


def board_to_attack_labels(board: chess.Board) -> torch.Tensor:
    labels = torch.zeros(64, dtype=torch.long)
    for sq in chess.SQUARES:
        n_white = min(len(board.attackers(chess.WHITE, sq)), 4)
        n_black = min(len(board.attackers(chess.BLACK, sq)), 4)
        labels[sq] = n_white * 5 + n_black
    return labels


def canonicalize(hidden: torch.Tensor, end_fens: list[str]) -> torch.Tensor:
    """Flip hidden states for black-to-move positions so index 0 = absolute a1."""
    black = torch.tensor([fen.split()[1] == 'b' for fen in end_fens])
    if black.any():
        flip = torch.tensor(_FLIP_IDX, device=hidden.device)
        hidden = hidden.clone()
        hidden[black] = hidden[black][:, flip]
    return hidden


def run_val(val_loader: DataLoader, probe: nn.Module, encoder: Lc0Bt4HFModel, device: torch.device, args, label_fn):
    probe.eval()
    all_preds, all_labels, all_is_white = [], [], []

    with torch.no_grad():
        for planes, end_fens in val_loader:
            planes = planes.to(device)
            hidden = encoder(planes, output_hidden_states=True).all_hidden_states[args.layer_idx]
            hidden = canonicalize(hidden, end_fens)

            B = hidden.shape[0]
            sq_idx = torch.stack([torch.randperm(64, device=device)[:args.val_k] for _ in range(B)])
            x = hidden.gather(1, sq_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_SIZE)).reshape(B * args.val_k, HIDDEN_SIZE)
            labels = torch.stack([label_fn(chess.Board(f)) for f in end_fens]).to(device)
            y = labels.gather(1, sq_idx).reshape(B * args.val_k)

            is_white = torch.tensor([f.split()[1] == 'w' for f in end_fens], device=device)
            is_white = is_white.unsqueeze(1).expand(-1, args.val_k).reshape(-1)

            all_preds.append(probe(x).argmax(dim=-1).cpu())
            all_labels.append(y.cpu())
            all_is_white.append(is_white.cpu())

    return torch.cat(all_preds), torch.cat(all_labels), torch.cat(all_is_white)


def _safe_acc(preds, labels, c):
    mask = labels == c
    return (preds[mask] == c).float().mean().item() if mask.any() else float("nan")


def log_piece_val(log_print, log_metric, step, preds, labels, is_white):
    val_acc = (preds == labels).float().mean().item()
    log_print(f"\n--- step {step:6d} | val_acc {val_acc:.4f} ---")
    log_print(f"  {'class':<14s} {'W-to-move':>10} {'B-to-move':>10}")
    per_w, per_b = [], []
    for c, name in enumerate(CLASS_NAMES):
        acc_w = _safe_acc(preds[is_white], labels[is_white], c)
        acc_b = _safe_acc(preds[~is_white], labels[~is_white], c)
        per_w.append(acc_w); per_b.append(acc_b)
        log_print(f"  {name:<14s} {acc_w:>10.4f} {acc_b:>10.4f}")
    log_print("")
    log_metric({"step": step, "val_acc": val_acc, "per_class_w": per_w, "per_class_b": per_b})
    return val_acc


def log_attack_val(log_print, log_metric, step, preds, labels, is_white):
    joint_acc = (preds == labels).float().mean().item()
    pred_w, true_w = preds // 5, labels // 5
    pred_b, true_b = preds % 5, labels % 5
    w_marg = (pred_w == true_w).float().mean().item()
    b_marg = (pred_b == true_b).float().mean().item()
    log_print(f"\n--- step {step:6d} | joint_acc {joint_acc:.4f} | W-marginal {w_marg:.4f} | B-marginal {b_marg:.4f} ---")

    wm_w = is_white.bool()
    wm_b = ~wm_w
    pw_wm, tw_wm = pred_w[wm_w], true_w[wm_w]
    pb_wm, tb_wm = pred_b[wm_w], true_b[wm_w]
    pw_bm, tw_bm = pred_w[wm_b], true_w[wm_b]
    pb_bm, tb_bm = pred_b[wm_b], true_b[wm_b]

    log_print(f"  {'bucket':<14s} {'W-atk W-to-move':>16} {'W-atk B-to-move':>16}")
    wm_white_buckets, bm_white_buckets = [], []
    for i, name in enumerate(_BUCKET_NAMES):
        acc_wm = _safe_acc(pw_wm, tw_wm, i)
        acc_bm = _safe_acc(pw_bm, tw_bm, i)
        wm_white_buckets.append(acc_wm); bm_white_buckets.append(acc_bm)
        log_print(f"  {name:<14s} {acc_wm:>16.4f} {acc_bm:>16.4f}")

    log_print(f"  {'bucket':<14s} {'B-atk W-to-move':>16} {'B-atk B-to-move':>16}")
    wm_black_buckets, bm_black_buckets = [], []
    for i, name in enumerate(_BUCKET_NAMES):
        acc_wm = _safe_acc(pb_wm, tb_wm, i)
        acc_bm = _safe_acc(pb_bm, tb_bm, i)
        wm_black_buckets.append(acc_wm); bm_black_buckets.append(acc_bm)
        log_print(f"  {name:<14s} {acc_wm:>16.4f} {acc_bm:>16.4f}")

    log_print("")
    log_metric({
        "step": step, "joint_acc": joint_acc, "w_marginal": w_marg, "b_marginal": b_marg,
        "wm_white_buckets": wm_white_buckets, "wm_black_buckets": wm_black_buckets,
        "bm_white_buckets": bm_white_buckets, "bm_black_buckets": bm_black_buckets,
    })
    return joint_acc


def compute_class_weights(triples: list, label_fn, n_classes: int, n_scan: int = 10000) -> torch.Tensor:
    """Inverse-frequency weights from the first n_scan training positions (CPU, no encoder needed)."""
    counts = torch.zeros(n_classes)
    for triple in triples[:n_scan]:
        _, _, end_fen = triple
        for c in label_fn(chess.Board(end_fen)):
            counts[c] += 1
    # normalize so weights average to 1.0
    weights = counts.sum() / (n_classes * counts.clamp(min=1))
    return weights


def _infinite_loader(loader: DataLoader):
    while True:
        yield from loader


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    probe: nn.Module,
    encoder: Lc0Bt4HFModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
    log_path: Path,
    label_fn,
    log_val_fn,
    class_weights: torch.Tensor | None = None,
    patience: int = 5,
):
    encoder.eval()
    probe.train()
    best_val_acc = -1.0
    no_improve_count = 0
    running_loss, running_steps = 0.0, 0

    metrics_path = log_path.parent / "metrics.jsonl"
    with open(log_path, "w") as log, open(metrics_path, "w") as metrics:
        def log_print(line: str):
            print(line)
            log.write(line + "\n")
            log.flush()

        def log_metric(record: dict):
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()

        for step, (planes, end_fens) in enumerate(
            itertools.islice(_infinite_loader(train_loader), args.max_steps), start=1
        ):
            planes = planes.to(device)
            with torch.no_grad():
                hidden = encoder(planes, output_hidden_states=True).all_hidden_states[args.layer_idx]
            hidden = canonicalize(hidden, end_fens)

            B = hidden.shape[0]
            sq_idx = torch.randint(0, 64, (B, args.train_k), device=device)

            x = hidden.gather(1, sq_idx.unsqueeze(-1).expand(-1, -1, HIDDEN_SIZE)).reshape(B * args.train_k, HIDDEN_SIZE)
            labels = torch.stack([label_fn(chess.Board(f)) for f in end_fens]).to(device)
            y = labels.gather(1, sq_idx).reshape(B * args.train_k)

            loss = nn.functional.cross_entropy(probe(x), y, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_steps += 1

            if step % args.log_every == 0:
                avg_loss = running_loss / running_steps
                log_print(f"step {step:6d} | train_loss {avg_loss:.4f}")
                log_metric({"step": step, "train_loss": avg_loss})
                running_loss, running_steps = 0.0, 0

            if step % args.eval_every == 0:
                preds, labels_val, is_white = run_val(val_loader, probe, encoder, device, args, label_fn)
                val_acc = log_val_fn(log_print, log_metric, step, preds, labels_val, is_white)
                probe.train()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        log_print(f"Early stopping at step {step} (no improvement for {patience} evals, best val_acc {best_val_acc:.4f})")
                        break


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_triples, val_triples = load_and_split(args.jsonl, args.n_positions, args.val_size, args.seed)
    train_loader = DataLoader(PositionDataset(train_triples), batch_size=args.train_bs, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(PositionDataset(val_triples), batch_size=args.val_bs, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    encoder = Lc0Bt4HFModel.from_pretrained(args.lc0_weights).to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)

    if args.probe_type == "piece":
        probe = PieceProbe(layer_idx=args.layer_idx).to(device)
        label_fn = board_to_labels
        log_val_fn = log_piece_val
        n_classes = NUM_CLASSES
    else:
        probe = AttackProbe(layer_idx=args.layer_idx).to(device)
        label_fn = board_to_attack_labels
        log_val_fn = log_attack_val
        n_classes = ATTACK_NUM_CLASSES

    class_weights = compute_class_weights(train_triples, label_fn, n_classes).to(device) if args.class_weights else None

    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir) / f"{args.run_name}_steps{args.max_steps}_bs{args.train_bs}_lr{args.lr}" / f"layer_{args.layer_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train(train_loader, val_loader, probe, encoder, optimizer, device, args,
          log_path=out_dir / "train.log", label_fn=label_fn, log_val_fn=log_val_fn,
          class_weights=class_weights, patience=args.patience)

    torch.save(probe.state_dict(), out_dir / "probe.pt")
    print(f"Saved probe weights to {out_dir / 'probe.pt'}")


if __name__ == "__main__":
    main()
