# Gomoku (五子棋)

15x15 pygame gomoku board. Human (black) clicks to place stones. Ghost (white) responds via Channel.

## Setup

```bash
cd .moss_ws/apps/games/gomoku
uv sync
```

Dependencies: `pygame>=2.5.0`, `ghoshell-moss[host]` (editable from workspace root). No extra models or services required.

## Commands

- `human_move(row, col)` — place black stone via voice/CTML (Ghost proxy for human)
- `move(row, col)` — place white stone
- `ai_move()` — run built-in alpha-beta and place best move
- `reset_board()` — new game
- `undo_move()` — undo last human+AI pair

## Interaction flow

Human clicks → stone placed → input signal sent to Ghost → Ghost sees board state via context_messages → Ghost calls `ai_move()` or `move(row,col)` via CTML → white stone placed → repeat.

## AI

Minimax + alpha-beta, depth 2, beam width 15. Moves limited to positions within 2 cells of existing stones.