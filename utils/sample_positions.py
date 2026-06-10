#!/usr/bin/env python3
"""
Sample random positions (with move history) from a Lichess PGN or PGN.zst file.

Interactive mode (--interactive):
    Reservoir-samples one random game from up to --max-games games, picks a
    random ply, and prints the board + position info for a quick sanity check.

Non-interactive mode (default):
    Streams through the entire file and writes a JSONL file where each line is
    a JSON array [start_fen, move_list, end_fen].
      - start_fen:  FEN at most HISTORY_PLIES (7) moves before the sampled position
      - move_list:  the ≤7 UCI moves from start_fen to end_fen
      - end_fen:    FEN of the randomly sampled position

    LC0 uses 8 history slots (start_fen board + up to 7 moves), so storing the
    full game move list is wasteful and unnecessary. start_fen is always at most
    7 plies before end_fen. Pass start_fen + move_list to
    model.input_planes_from_fen to get LC0 hidden states. Use end_fen to derive
    piece-location labels and the active color for canonicalization.
"""

import argparse
import io
import json
import random
import sys

import chess
import chess.pgn

# LC0 encodes up to 8 history boards: the position at start_fen plus one per move.
# Storing more than 7 moves provides no additional signal to the encoder.
HISTORY_PLIES = 7

# Skip positions from the opening — both players must have made at least 5 moves.
MIN_PLY = 10


def open_pgn(path: str):
    if path.endswith(".zst"):
        import zstandard
        f = open(path, "rb")
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def iter_games(path: str, max_games: int = None):
    """Yield valid games (non-empty, no parse errors)."""
    with open_pgn(path) as handle:
        count = 0
        while True:
            if max_games is not None and count >= max_games:
                break
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            if game.errors:
                continue
            moves = list(game.mainline_moves())
            if not moves:
                continue
            yield game
            count += 1


def sample_position(game: chess.pgn.Game, rng: random.Random):
    """Return (start_fen, moves_prefix, board_at_ply, ply), or None if game too short."""
    start_fen = game.headers.get("FEN", chess.STARTING_FEN)
    moves = [m.uci() for m in game.mainline_moves()]
    if len(moves) < MIN_PLY:
        return None
    ply = rng.randint(MIN_PLY, len(moves))
    board = chess.Board(start_fen)
    for uci in moves[:ply]:
        board.push(chess.Move.from_uci(uci))
    return start_fen, moves[:ply], board, ply


def run_interactive(pgn_path: str, max_games: int, seed: int) -> None:
    rng = random.Random(seed)

    chosen = None
    for k, game in enumerate(iter_games(pgn_path, max_games=max_games)):
        if rng.random() < 1.0 / (k + 1):
            chosen = game
    if chosen is None:
        print("No games found.", file=sys.stderr)
        sys.exit(1)

    result = sample_position(chosen, rng)
    if result is None:
        print("Sampled game is shorter than MIN_PLY; try again.", file=sys.stderr)
        sys.exit(1)
    start_fen, moves, board, ply = result

    h = chosen.headers
    print(f"Game   : {h.get('White', '?')} vs {h.get('Black', '?')}")
    print(f"Result : {h.get('Result', '?')}")
    print(f"Event  : {h.get('Event', '?')}")
    print(f"Total plies in game : {len([m for m in chosen.mainline_moves()])}")
    print(f"Sampled ply         : {ply}")
    print(f"Start FEN  : {start_fen}")
    print(f"Move prefix: {' '.join(moves) if moves else '(start of game)'}")
    print(f"Position FEN: {board.fen()}")
    print()
    print(board)
    print()

    turn = "White" if board.turn == chess.WHITE else "Black"
    print(f"Side to move: {turn}")
    print()
    print("Piece locations (absolute):")
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            color = "White" if piece.color == chess.WHITE else "Black"
            print(f"  {chess.square_name(sq):3s} : {color} {chess.piece_name(piece.piece_type)}")


def run_generate(
    pgn_path: str,
    output_path: str,
    n_per_game: int,
    seed: int,
    max_games,
) -> None:
    rng = random.Random(seed)
    written = 0

    with open(output_path, "w") as out:
        for game_idx, game in enumerate(iter_games(pgn_path, max_games=max_games)):
            moves_all = [m.uci() for m in game.mainline_moves()]
            if len(moves_all) < MIN_PLY:
                continue

            # Build FEN for every ply once per game so multiple samples are cheap.
            game_start_fen = game.headers.get("FEN", chess.STARTING_FEN)
            board = chess.Board(game_start_fen)
            game_fens = [board.fen()]
            for uci in moves_all:
                board.push(chess.Move.from_uci(uci))
                game_fens.append(board.fen())

            for _ in range(n_per_game):
                end_ply = rng.randint(MIN_PLY, len(moves_all))
                start_ply = max(0, end_ply - HISTORY_PLIES)

                start_fen = game_fens[start_ply]
                move_list = moves_all[start_ply:end_ply]
                end_fen = game_fens[end_ply]

                out.write(json.dumps([start_fen, move_list, end_fen]) + "\n")
                written += 1

            if (game_idx + 1) % 10_000 == 0:
                print(
                    f"  processed {game_idx + 1:,} games | {written:,} positions written",
                    file=sys.stderr,
                )

    print(f"Done. Wrote {written:,} positions to {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample random positions with move history from a PGN file."
    )
    parser.add_argument("pgn", help="Path to .pgn or .pgn.zst file")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Load one random game, display a random position, then exit.",
    )
    parser.add_argument(
        "--output",
        default="positions.jsonl",
        help="Output JSONL file path (non-interactive mode).",
    )
    parser.add_argument(
        "--n-per-game",
        type=int,
        default=1,
        help="Number of positions to sample per game (non-interactive mode).",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Stop after this many games. Default: process all games. "
             "In interactive mode defaults to 1000.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed. Defaults to None (random) in interactive mode, 42 in generate mode.")
    args = parser.parse_args()

    if args.interactive:
        max_games = args.max_games if args.max_games is not None else 1000
        seed = args.seed  # None → random each run
        run_interactive(args.pgn, max_games=max_games, seed=seed)
    else:
        run_generate(
            args.pgn,
            output_path=args.output,
            n_per_game=args.n_per_game,
            seed=args.seed if args.seed is not None else 42,
            max_games=args.max_games,
        )


if __name__ == "__main__":
    main()
