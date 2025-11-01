# Chess Bot - AI Coding Agent Instructions

## Project Overview
AlphaZero-style chess engine using Monte Carlo Tree Search (MCTS) + deep neural networks. Self-play generates training data; neural network learns both move policy and position evaluation.

## Architecture

### Core Components (5 modules)
1. **`src/board_encoding.py`** - Converts chess positions → 106-plane tensors (8×8 each)
2. **`src/move_encoding.py`** - Maps chess moves ↔ policy indices [0-4863] (64 squares × 76 move types)
3. **`src/mcts.py`** - Functional-style MCTS with mutable `Node` dataclass for performance
4. **`src/nn.py`** - `AlphaZeroNet` (ResNet-style CNN) outputs policy logits [4864] + value [-1,1]
5. **`src/training.py`** - Full pipeline: self-play → data collection → neural network training

### Data Flow
```
chess.Board + history → encode_board() → [106,8,8] tensor
                                            ↓
                                       AlphaZeroNet
                                            ↓
                              policy_logits [4864] + value
                                            ↓
                         MCTS search() refines policy via simulations
                                            ↓
                              sample_move() → make move
                                            ↓
                           collect (state, policy, outcome) tuples
                                            ↓
                              train network on self-play data
```

## Critical Patterns

### Board State Always Includes History
**Every** function operating on chess positions takes `board: chess.Board, history: List[chess.Board]`:
- `encode_board(board, history)` - encodes last 7 positions for repetition detection
- `create_root_node(board, history)` - MCTS needs history for encoding
- History starts with `[board.copy()]` at game start, append after each move

### Mutable Nodes, Functional Operations
`Node` dataclass uses in-place updates (`update_node()`) but operators are pure functions:
- `search(root, model, num_simulations, device)` → returns updated root
- Tree reuse: after move, reuse `root.children[move]` as new root (see `self_play.py:35-38`)
- Never modify `board` in Node - always `.copy()` first

### Move Encoding is Bidirectional
- `encode_move(move)` → policy index [0-4863]
- `decode_move(index, board)` → chess.Move (validates legality)
- Policy tensor size is **4864** (not 4672 - check comments for discrepancies)

### Temperature-Based Move Sampling
Early game (moves < 30): `temperature=1.0` (exploration)
Late game: `temperature=0.1` (near-greedy, deterministic play)
```python
move = sample_move(root, board, temperature=1.0 if move_count < 30 else 0.1)
```

## Development Workflows

### Running Scripts
- Activate venv: `source .venv/bin/python` or use VS Code's Python interpreter
- Python path must include workspace root: `PYTHONPATH=/home/brandon/projects/chess-bot` (set in `.vscode/launch.json`)
- Import style: `from src.module import function` (absolute imports from project root)

### Testing Individual Modules
Each module has `if __name__ == "__main__":` blocks with runnable examples:
- `python src/board_encoding.py` - encode sample positions
- `python src/nn.py` - train on dummy data (1000 samples)
- `python src/training.py` - run `quick_test()` (2 iterations, 5 games, CPU)

### Debugging MCTS
Key inspection points:
- `Node.visit_count` - total simulations through this node
- `get_value(node)` - average value (Q-value)
- `extract_tensor_policy(root)` - visit count distribution as policy tensor

## Code Conventions

### Type Hints Everywhere
All function signatures use full type hints per Pyright `basic` mode:
```python
def encode_board(board: chess.Board, history: List[chess.Board]) -> torch.Tensor:
```

### Formatting via Ruff
- Max line length: 120 chars
- Format on save enabled
- Imports auto-organized (no trailing commas in lists)
- Run manually: `ruff format` or `ruff check --fix`

### Tensor Shapes in Comments
Document expected shapes in docstrings and inline:
```python
policy_logits: torch.Tensor  # [batch_size, 4864]
state: torch.Tensor         # [106, 8, 8]
```

## Common Gotchas

### Import Paths
- Use `from src.module import function` (NOT relative imports like `from .module`)
- `board_encoding.py` example code imports `from nn import AlphaZeroNet` without `src.` prefix (runs as script)

### Training Pipeline Incomplete
- `self_play.py:47` calls undefined `create_training_examples(examples, board)` function
- `src/training.py` has full `AlphaZeroTrainer` class with complete pipeline
- When implementing self-play, check if reusing existing `SelfPlayGame` class or creating new `play_automated_game()`

### Policy Size Discrepancy
Comments/docstrings inconsistently reference 4672 vs 4864:
- **Correct size: 4864** (64 squares × 76 move types, confirmed in `move_encoding.py`)
- Update outdated comments showing [4672]

### Device Handling
Default device is `"cuda"` in many functions - ensure model and tensors on same device:
```python
state_batch = state.unsqueeze(0).to(device)
policy_logits, value = model(state_batch)
```

## Key Files for Understanding Patterns
- `src/mcts.py:151-184` - Core `simulate()` recursion with tree updates
- `self_play.py:24-45` - Tree reuse pattern in self-play loop
- `src/nn.py:77-96` - AlphaZero loss function (policy + value)
- `src/training.py:299-338` - Complete training iteration loop
