"""Gomoku (五子棋) — 15x15 pygame board with AI via Channel commands.

Human plays black (clicks on grid), Ghost plays white via CTML:
  <apps.games_gomoku:move row="7" col="7" />
  <apps.games_gomoku:ai_move />   (built-in alpha-beta)
  <apps.games_gomoku:reset />
  <apps.games_gomoku:undo />

context_messages provides the board state as a compact text grid.
"""

from __future__ import annotations

import asyncio
import math
import sys
from typing import Optional

import pygame

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel

# ── Constants ──────────────────────────────────────────────────────────────

SIZE = 15
CELL = 38
MARGIN = 40
WINDOW = MARGIN * 2 + CELL * (SIZE - 1)
STONE_R = CELL // 2 - 2

EMPTY = 0
BLACK = 1  # human
WHITE = 2  # AI

# Colors
WOOD = (222, 184, 135)
LINE = (92, 64, 51)
BLACK_C = (30, 30, 30)
WHITE_C = (240, 240, 240)
HIGHLIGHT = (220, 50, 50)
TEXT_C = (255, 255, 255)
PANEL = (40, 40, 40)

# AI scoring
WIN_SCORE = 100_000_000
FIVE = 100_000
FOUR_OPEN = 50_000
FOUR_BLOCKED = 5_000
THREE_OPEN = 5_000
THREE_BLOCKED = 500
TWO_OPEN = 500
TWO_BLOCKED = 50
ONE = 10

DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]

# ── Game State ─────────────────────────────────────────────────────────────

board: list[list[int]] = [[EMPTY] * SIZE for _ in range(SIZE)]
current_player = BLACK
game_over = False
winner: int = EMPTY
move_history: list[tuple[int, int, int]] = []  # (row, col, player)
last_move: tuple[int, int] | None = None
human_turn = True
_ai_signaled = False
_matrix: Optional[Matrix] = None


def _pub_gomoku_state(state: str) -> None:
    """Publish gomoku state to stream for other apps (e.g. ai_eye) to react."""
    if _matrix is not None:
        _matrix.session.pub_stream_delta("gomoku/state", state.encode())


def _signal_ai() -> None:
    """Send input signal to Ghost to prompt an AI move."""
    global _ai_signaled
    if _matrix is None or _ai_signaled:
        return
    _ai_signaled = True
    _matrix.session.add_input_signal(
        "Your turn to play White on the Gomoku board. "
        "Please respond with <apps.games_gomoku:ai_move /> to make your move.",
        description="gomoku: AI turn",
    )


def reset() -> None:
    global board, current_player, game_over, winner, move_history, last_move, human_turn, _ai_signaled
    board = [[EMPTY] * SIZE for _ in range(SIZE)]
    current_player = BLACK
    game_over = False
    winner = EMPTY
    move_history.clear()
    last_move = None
    human_turn = True
    _ai_signaled = False


def undo() -> str:
    global board, current_player, game_over, winner, last_move, human_turn, _ai_signaled
    if not move_history:
        return "No moves to undo"
    # Undo two moves (AI + human)
    for _ in range(2):
        if not move_history:
            break
        r, c, _ = move_history.pop()
        board[r][c] = EMPTY
    if move_history:
        _, _, prev = move_history[-1]
        current_player = BLACK if prev == WHITE else WHITE
    else:
        current_player = BLACK
    game_over = False
    winner = EMPTY
    last_move = move_history[-1][:2] if move_history else None
    human_turn = True
    _ai_signaled = False
    return "Undone"


def place(r: int, c: int, player: int) -> bool:
    global game_over, winner, last_move, human_turn
    if not (0 <= r < SIZE and 0 <= c < SIZE):
        return False
    if board[r][c] != EMPTY:
        return False
    if game_over:
        return False
    board[r][c] = player
    move_history.append((r, c, player))
    last_move = (r, c)
    if check_win(r, c, player):
        game_over = True
        winner = player
    human_turn = not human_turn
    return True


def check_win(r: int, c: int, player: int) -> bool:
    for dr, dc in DIRS:
        cnt = 1
        for sign in (1, -1):
            rr, cc = r + dr * sign, c + dc * sign
            while 0 <= rr < SIZE and 0 <= cc < SIZE and board[rr][cc] == player:
                cnt += 1
                rr += dr * sign
                cc += dc * sign
        if cnt >= 5:
            return True
    return False


def is_draw() -> bool:
    return all(board[r][c] != EMPTY for r in range(SIZE) for c in range(SIZE))


def board_to_text() -> str:
    rows = []
    header = "   " + "".join(f"{i:2}" for i in range(SIZE))
    rows.append(header)
    symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
    for r in range(SIZE):
        line = f"{r:2} " + " ".join(symbols[board[r][c]] for c in range(SIZE))
        rows.append(line)
    return "\n".join(rows)


# ── AI ─────────────────────────────────────────────────────────────────────


def _is_line_start(r: int, c: int, dr: int, dc: int, player: int) -> bool:
    pr, pc = r - dr, c - dc
    return not (0 <= pr < SIZE and 0 <= pc < SIZE and board[pr][pc] == player)


def _eval_line(r: int, c: int, dr: int, dc: int, player: int) -> int:
    if not _is_line_start(r, c, dr, dc, player):
        return 0
    cnt = 0
    rr, cc = r, c
    while 0 <= rr < SIZE and 0 <= cc < SIZE and board[rr][cc] == player:
        cnt += 1
        rr += dr
        cc += dc
    open1 = 1 if (0 <= rr < SIZE and 0 <= cc < SIZE and board[rr][cc] == EMPTY) else 0
    pr, pc = r - dr, c - dc
    open2 = 1 if (0 <= pr < SIZE and 0 <= pc < SIZE and board[pr][pc] == EMPTY) else 0
    opens = open1 + open2

    if cnt >= 5:
        return FIVE
    if cnt == 4:
        return FOUR_OPEN if opens == 2 else FOUR_BLOCKED if opens == 1 else 0
    if cnt == 3:
        return THREE_OPEN if opens == 2 else THREE_BLOCKED if opens == 1 else 0
    if cnt == 2:
        return TWO_OPEN if opens == 2 else TWO_BLOCKED if opens == 1 else 0
    if cnt == 1:
        return ONE if opens > 0 else 0
    return 0


def evaluate(player: int) -> int:
    sc = 0
    opp = BLACK if player == WHITE else WHITE
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == player:
                for dr, dc in DIRS:
                    sc += _eval_line(r, c, dr, dc, player)
            elif board[r][c] == opp:
                for dr, dc in DIRS:
                    sc -= _eval_line(r, c, dr, dc, opp)
    return sc


def _candidates():
    pts = set()
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != EMPTY:
                for dr in (-2, -1, 0, 1, 2):
                    for dc in (-2, -1, 0, 1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc] == EMPTY:
                            pts.add((nr, nc))
    if not pts:
        pts.add((SIZE // 2, SIZE // 2))
    return [(evaluate_move(r, c, WHITE), r, c) for r, c in pts]


def evaluate_move(r: int, c: int, player: int) -> int:
    board[r][c] = player
    sc = 0
    for dr, dc in DIRS:
        sc += _eval_line(r, c, dr, dc, player)
    board[r][c] = EMPTY
    return sc


def minimax(depth: int, alpha: float, beta: float, maximizing: bool) -> float:
    if depth == 0:
        return float(evaluate(WHITE))

    if check_win(*move_history[-1][:2], move_history[-1][2]) if move_history else False:
        last_player = move_history[-1][2] if move_history else EMPTY
        if last_player == WHITE:
            return float(WIN_SCORE + depth)
        else:
            return float(-WIN_SCORE - depth)

    if is_draw():
        return 0.0

    cand = _candidates()
    if not cand:
        return float(evaluate(WHITE))

    cand.sort(key=lambda x: x[0], reverse=True)
    cand = cand[:15]  # beam search

    if maximizing:
        best = -float("inf")
        for _, r, c in cand:
            board[r][c] = WHITE
            move_history.append((r, c, WHITE))
            val = minimax(depth - 1, alpha, beta, False)
            move_history.pop()
            board[r][c] = EMPTY
            best = max(best, val)
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return best
    else:
        best = float("inf")
        for _, r, c in cand:
            board[r][c] = BLACK
            move_history.append((r, c, BLACK))
            val = minimax(depth - 1, alpha, beta, True)
            move_history.pop()
            board[r][c] = EMPTY
            best = min(best, val)
            beta = min(beta, val)
            if alpha >= beta:
                break
        return best


def ai_best_move(depth: int = 2) -> tuple[int, int] | None:
    cand = _candidates()
    if not cand:
        return None
    best_score = -float("inf")
    best = cand[0]
    for sc, r, c in cand[:20]:
        board[r][c] = WHITE
        move_history.append((r, c, WHITE))
        val = minimax(depth - 1, -float("inf"), float("inf"), False)
        move_history.pop()
        board[r][c] = EMPTY
        if val > best_score:
            best_score = val
            best = (sc, r, c)
    return best[1], best[2]


# ── Rendering ──────────────────────────────────────────────────────────────

_screen: pygame.Surface | None = None
_font: pygame.font.Font | None = None


def _init_display():
    global _screen, _font
    pygame.init()
    _screen = pygame.display.set_mode((WINDOW, WINDOW + 60))
    pygame.display.set_caption("Gomoku — MOSS")
    _font = pygame.font.SysFont("arial", 16)


def _pixel_to_grid(px: int, py: int) -> tuple[int, int]:
    col = round((px - MARGIN) / CELL)
    row = round((py - MARGIN) / CELL)
    return max(0, min(SIZE - 1, row)), max(0, min(SIZE - 1, col))


def _grid_to_pixel(r: int, c: int) -> tuple[int, int]:
    return MARGIN + c * CELL, MARGIN + r * CELL


def _draw_stone(r: int, c: int, player: int):
    x, y = _grid_to_pixel(r, c)
    color = BLACK_C if player == BLACK else WHITE_C
    outline = (80, 80, 80) if player == BLACK else (180, 180, 180)
    pygame.draw.circle(_screen, outline, (x, y), STONE_R)
    pygame.draw.circle(_screen, color, (x, y), STONE_R - 1)
    # subtle highlight
    light = (80, 80, 80) if player == BLACK else (255, 255, 255)
    pygame.draw.circle(_screen, light, (x - STONE_R // 4, y - STONE_R // 4), STONE_R // 3)


def _draw_last_move():
    if last_move is None or not move_history:
        return
    r, c = last_move
    player = move_history[-1][2]
    x, y = _grid_to_pixel(r, c)
    color = (255, 100, 100) if player == BLACK else (255, 80, 80)
    pygame.draw.circle(_screen, color, (x, y), 5)


def _draw_status():
    panel = pygame.Surface((WINDOW, 40))
    panel.fill(PANEL)
    _screen.blit(panel, (0, WINDOW))

    if game_over:
        if winner == BLACK:
            txt = "Black (Human) wins!"
        elif winner == WHITE:
            txt = "White (AI) wins!"
        else:
            txt = "Draw!"
    else:
        turn = "Black (Human)" if current_player == BLACK else "White (AI)"
        txt = f"Turn: {turn}"
    line = txt + "  |  Click to place  |  Ghost auto-responds"
    surf = _font.render(line, True, TEXT_C)
    _screen.blit(surf, (12, WINDOW + 10))


def render():
    _screen.fill(PANEL)
    # Board background
    board_surf = pygame.Surface((WINDOW, WINDOW))
    board_surf.fill(WOOD)
    _screen.blit(board_surf, (0, 0))

    # Grid
    for i in range(SIZE):
        x0, y0 = _grid_to_pixel(i, 0)
        x1, y1 = _grid_to_pixel(i, SIZE - 1)
        pygame.draw.line(_screen, LINE, (x0, y0), (x1, y1), 1)
        pygame.draw.line(_screen, LINE, (_grid_to_pixel(0, i)), (_grid_to_pixel(SIZE - 1, i)), 1)

    # Star points (traditional gomoku markers)
    stars = [(3, 3), (3, 7), (3, 11), (7, 3), (7, 7), (7, 11), (11, 3), (11, 7), (11, 11)]
    for r, c in stars:
        x, y = _grid_to_pixel(r, c)
        pygame.draw.circle(_screen, LINE, (x, y), 3)

    # Stones
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != EMPTY:
                _draw_stone(r, c, board[r][c])

    _draw_last_move()
    _draw_status()
    pygame.display.flip()


# ── Channel ────────────────────────────────────────────────────────────────

channel = new_channel(
    name="games_gomoku",
    description="15x15 Gomoku board. Human plays black (click or voice via human_move). Ghost plays white via commands.",
)


@channel.build.command()
async def move(row: int, col: int) -> str:
    """Place white stone at (row, col)."""
    global _ai_signaled, current_player
    if current_player != WHITE:
        return f"Not AI's turn. Current: {'Black' if current_player == BLACK else 'White'}"
    if game_over:
        winner_name = "Black" if winner == BLACK else "White" if winner == WHITE else "None"
        return f"Game over. Winner: {winner_name}"
    if not place(row, col, WHITE):
        return f"Invalid move at ({row}, {col})"
    _ai_signaled = False
    current_player = BLACK
    _pub_gomoku_state("game_over" if game_over else "ai_moved")
    if game_over:
        return f"White wins! Move: ({row}, {col})"
    if is_draw():
        return f"Draw. Move: ({row}, {col})"
    return f"White placed at ({row}, {col})"


@channel.build.command()
async def ai_move() -> str:
    """Run built-in alpha-beta AI and place the best white move."""
    global _ai_signaled, current_player
    if current_player != WHITE:
        return f"Not AI's turn."
    if game_over:
        return "Game already over."
    result = ai_best_move(depth=2)
    if result is None:
        return "No valid moves"
    r, c = result
    place(r, c, WHITE)
    _ai_signaled = False
    current_player = BLACK
    _pub_gomoku_state("game_over" if game_over else "ai_moved")
    if game_over:
        return f"AI wins! Move: ({r}, {c})"
    return f"AI played at ({r}, {c})"


@channel.build.command()
async def human_move(row: int, col: int) -> str:
    """Place a black stone for the human via voice/CTML command.
    Allows Ghost to play on behalf of the human when receiving voice commands."""
    global _ai_signaled, current_player
    if not human_turn:
        return "Not human's turn. Wait for AI to move."
    if game_over:
        winner_name = "Black" if winner == BLACK else "White" if winner == WHITE else "None"
        return f"Game over. Winner: {winner_name}"
    if not place(row, col, BLACK):
        return f"Invalid move at ({row}, {col})"
    current_player = WHITE
    _signal_ai()
    _pub_gomoku_state("human_moved")
    return f"Black placed at ({row}, {col})"


@channel.build.command()
async def reset_board() -> str:
    """Reset the board to start a new game."""
    reset()
    return "Board reset. Black (human) goes first."


@channel.build.command()
async def undo_move() -> str:
    """Undo the last moves (human + AI pair)."""
    return undo()


@channel.build.context_messages
async def context() -> list:
    parts = []
    parts.append(f"[games/gomoku] Board state ({SIZE}x{SIZE}, X=Black/Human, O=White/AI):")
    parts.append(board_to_text())
    if game_over:
        w = "Black (Human)" if winner == BLACK else "White (AI)" if winner == WHITE else "None"
        parts.append(f"Game over. Winner: {w}. Use reset_board() to start new game.")
    else:
        turn = "Black (Human)" if current_player == BLACK else "White (AI)"
        if human_turn:
            parts.append(f"Current turn: {turn}. Awaiting human click.")
        else:
            parts.append(f"Current turn: {turn}. It is YOUR turn! Respond with <apps.games_gomoku:ai_move />.")
    return parts


# ── Game Loop ──────────────────────────────────────────────────────────────

async def game_loop():
    global human_turn, current_player, game_over

    clock = pygame.time.Clock()
    running = True

    try:
        while running:
            # Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if human_turn and not game_over:
                        px, py = event.pos
                        if py < WINDOW:  # click is on board
                            r, c = _pixel_to_grid(px, py)
                            if board[r][c] == EMPTY:
                                place(r, c, BLACK)
                                current_player = WHITE
                                _signal_ai()
                                _pub_gomoku_state("human_moved")

            render()
            clock.tick(60)
            await asyncio.sleep(0)
    finally:
        pygame.quit()


async def main(matrix: Matrix):
    global _matrix
    _matrix = matrix
    loop = asyncio.get_running_loop()
    game_task = loop.create_task(game_loop())
    await matrix.provide_channel(channel)
    # Channel cleared (ghost session ended) — cancel game loop and clean up
    game_task.cancel()
    try:
        await game_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    import signal
    _init_display()
    pygame.event.pump()

    def _sigterm_handler(signum, frame):
        pygame.quit()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)
    Matrix.discover().run(main)