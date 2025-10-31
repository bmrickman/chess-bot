import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import chess
import torch
import torch.optim as optim

from board_encoding import encode_board
from mcts import MCTS
from nn import AlphaZeroNet, alphazero_loss
from src.move_encoding import decode_move, encode_move


@dataclass
class TrainingConfig:
    """Configuration for training"""

    # Self-play
    num_iterations: int = 100
    games_per_iteration: int = 100
    num_simulations: int = 800

    # Training
    epochs_per_iteration: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # Model
    num_res_blocks: int = 20
    num_channels: int = 256

    # Other
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    max_game_length: int = 200  # Max moves before draw


# Type alias for training examples
TrainingExample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
# (state [119,8,8], mcts_policy [4672], outcome [1,1])


class SelfPlayGame:
    """Plays one self-play game and collects training data"""

    def __init__(self, model: AlphaZeroNet, config: TrainingConfig):
        self.model = model
        self.config = config
        self.mcts = MCTS(model=model, num_simulations=config.num_simulations, device=config.device)

    def play(self) -> List[TrainingExample]:
        """
        Play one complete game

        Returns:
            List of training examples from the game
        """
        board = chess.Board()
        history: List[chess.Board] = [board.copy()]
        examples: List[Dict] = []

        move_count = 0

        while not board.is_game_over() and move_count < self.config.max_game_length:
            # Encode current position
            state = encode_board(board, history)

            # Run MCTS to get improved policy
            mcts_policy, mcts_move = self.mcts.search(board, history)

            # Store example (outcome will be filled in later)
            examples.append({"state": state, "policy": mcts_policy, "player": 1 if board.turn == chess.WHITE else -1})

            # Select move with temperature
            # Early game: temperature=1.0 (exploration)
            # Late game: temperature→0 (exploitation)
            temperature = 1.0 if move_count < 30 else 0.1
            move = self._sample_move(mcts_policy, board, temperature)

            # Make move
            board.push(move)
            history.append(board.copy())
            move_count += 1

        # Game over - determine outcome
        game_outcome = self._get_game_outcome(board)

        # Convert to training examples with outcomes
        training_examples: List[TrainingExample] = []
        for example in examples:
            # Outcome from current player's perspective
            outcome = game_outcome * example["player"]
            outcome_tensor = torch.tensor([[outcome]], dtype=torch.float32)

            training_examples.append(
                (
                    example["state"],  # [119, 8, 8]
                    example["policy"],  # [4672]
                    outcome_tensor,  # [1, 1]
                )
            )

        return training_examples

    def _sample_move(self, policy: torch.Tensor, board: chess.Board, temperature: float) -> chess.Move:
        """
        Sample a move from the MCTS policy

        Args:
            policy: MCTS policy [4672]
            board: Current board
            temperature: Sampling temperature (0 = greedy, 1 = stochastic)

        Returns:
            Selected chess move
        """
        legal_moves = list(board.legal_moves)

        # Get policy values for legal moves
        legal_move_probs: List[float] = []
        for move in legal_moves:
            move_idx = self.mcts._move_to_index(move)
            prob = policy[move_idx].item()
            legal_move_probs.append(prob)

        # Apply temperature
        if temperature == 0:
            # Greedy: pick best move
            best_idx = legal_move_probs.index(max(legal_move_probs))
            return legal_moves[best_idx]
        else:
            # Stochastic: sample with temperature
            legal_move_probs_tensor = torch.tensor(legal_move_probs)
            tempered_probs = legal_move_probs_tensor ** (1 / temperature)
            tempered_probs = tempered_probs / tempered_probs.sum()

            move_idx = torch.multinomial(tempered_probs, 1).item()
            return legal_moves[int(move_idx)]

    def _get_game_outcome(self, board: chess.Board) -> float:
        """
        Get game outcome from White's perspective

        Returns:
            1.0 if White won, -1.0 if Black won, 0.0 if draw
        """
        if not board.is_game_over():
            return 0.0  # Game not over yet (shouldn't happen)

        result = board.result()
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0


class SelfPlayDataGenerator:
    """Generates self-play training data"""

    def __init__(self, model: AlphaZeroNet, config: TrainingConfig):
        self.model = model
        self.config = config

    def generate(self, num_games: int) -> List[TrainingExample]:
        """
        Generate training data from self-play games

        Args:
            num_games: Number of games to play

        Returns:
            List of all training examples from all games
        """
        all_examples: List[TrainingExample] = []

        print(f"\nGenerating self-play data ({num_games} games)...")
        print("-" * 60)

        for game_num in range(num_games):
            game = SelfPlayGame(self.model, self.config)
            examples = game.play()
            all_examples.extend(examples)

            print(f"Game {game_num + 1}/{num_games}: {len(examples)} positions")

        print(f"\nTotal training examples collected: {len(all_examples)}")
        return all_examples


class Trainer:
    """Trains the neural network on self-play data"""

    def __init__(self, model: AlphaZeroNet, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = config.device

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

    def train(self, training_data: List[TrainingExample], epochs: int) -> Dict[str, List[float]]:
        """
        Train on a dataset

        Args:
            training_data: List of (state, policy, value) tuples
            epochs: Number of training epochs

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        metrics: Dict[str, List[float]] = {"total_loss": [], "policy_loss": [], "value_loss": []}

        print(f"\nTraining on {len(training_data)} examples...")
        print("-" * 60)

        for epoch in range(epochs):
            # Shuffle data
            random.shuffle(training_data)

            epoch_losses = {"total": 0.0, "policy": 0.0, "value": 0.0}
            num_batches = 0

            # Process in batches
            for batch_start in range(0, len(training_data), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(training_data))
                batch = training_data[batch_start:batch_end]

                # Train on batch
                losses = self._train_batch(batch)

                epoch_losses["total"] += losses["total_loss"]
                epoch_losses["policy"] += losses["policy_loss"]
                epoch_losses["value"] += losses["value_loss"]
                num_batches += 1

            # Update learning rate
            self.scheduler.step()

            # Record metrics
            avg_total = epoch_losses["total"] / num_batches
            avg_policy = epoch_losses["policy"] / num_batches
            avg_value = epoch_losses["value"] / num_batches

            metrics["total_loss"].append(avg_total)
            metrics["policy_loss"].append(avg_policy)
            metrics["value_loss"].append(avg_value)

            # Print progress
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Loss: {avg_total:.4f} "
                f"(Policy: {avg_policy:.4f}, Value: {avg_value:.4f}) - "
                f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
            )

        self.model.eval()
        return metrics

    def _train_batch(self, batch: List[TrainingExample]) -> Dict[str, float]:
        """
        Train on a single batch

        Args:
            batch: List of (state, policy, value) tuples

        Returns:
            Dictionary with loss values
        """
        # Stack batch tensors
        states = torch.stack([item[0] for item in batch]).to(self.device)
        policies = torch.stack([item[1] for item in batch]).to(self.device)
        values = torch.stack([item[2] for item in batch]).to(self.device).squeeze(1)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass
        policy_logits, value_preds = self.model(states)

        # Compute loss
        total_loss, policy_loss, value_loss = alphazero_loss(policy_logits, policies, value_preds, values)

        # Backward pass
        total_loss.backward()

        # Update weights
        self.optimizer.step()

        return {"total_loss": total_loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()}


class AlphaZeroTrainer:
    """Main training pipeline for AlphaZero"""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = AlphaZeroNet(num_res_blocks=config.num_res_blocks, num_channels=config.num_channels)

        # Initialize components
        self.data_generator = SelfPlayDataGenerator(self.model, config)
        self.trainer = Trainer(self.model, config)

    def train(self) -> None:
        """Run the complete training pipeline"""
        print("=" * 60)
        print("AlphaZero Training Pipeline")
        print("=" * 60)
        print(f"Device: {self.config.device}")
        print(f"Model: {self.config.num_res_blocks} blocks, {self.config.num_channels} channels")
        print(f"Iterations: {self.config.num_iterations}")
        print(f"Games per iteration: {self.config.games_per_iteration}")
        print(f"MCTS simulations: {self.config.num_simulations}")
        print("=" * 60)

        for iteration in range(self.config.num_iterations):
            print(f"\n{'=' * 60}")
            print(f"ITERATION {iteration + 1}/{self.config.num_iterations}")
            print(f"{'=' * 60}")

            # Phase 1: Generate self-play data
            training_data = self.data_generator.generate(self.config.games_per_iteration)

            # Phase 2: Train on self-play data
            metrics = self.trainer.train(training_data, self.config.epochs_per_iteration)

            # Phase 3: Save checkpoint
            checkpoint_path = Path(self.config.checkpoint_dir) / f"iteration_{iteration + 1}.pth"
            self.save_checkpoint(checkpoint_path, iteration, metrics)

            print(f"\n✓ Iteration {iteration + 1} complete!")
            print(f"Checkpoint saved: {checkpoint_path}")

    def save_checkpoint(self, path: Path, iteration: int, metrics: Dict[str, List[float]]) -> None:
        """
        Save model checkpoint

        Args:
            path: Path to save checkpoint
            iteration: Current iteration number
            metrics: Training metrics
        """
        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "scheduler_state_dict": self.trainer.scheduler.state_dict(),
            "config": self.config,
            "metrics": metrics,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> int:
        """
        Load model checkpoint

        Args:
            path: Path to checkpoint file

        Returns:
            Iteration number of loaded checkpoint
        """
        checkpoint = torch.load(path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        iteration = checkpoint["iteration"]
        print(f"Loaded checkpoint from iteration {iteration}")

        return iteration


def main():
    """Main entry point for training"""
    # Configure training
    config = TrainingConfig(
        num_iterations=100,
        games_per_iteration=100,
        num_simulations=800,
        epochs_per_iteration=10,
        batch_size=32,
        learning_rate=0.001,
        num_res_blocks=20,
        num_channels=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Create trainer
    trainer = AlphaZeroTrainer(config)

    # Optional: Load from checkpoint
    # checkpoint_path = Path('checkpoints/iteration_10.pth')
    # if checkpoint_path.exists():
    #     trainer.load_checkpoint(checkpoint_path)

    # Run training
    trainer.train()


def quick_test():
    """Quick test with minimal settings"""
    print("Running quick test...")

    config = TrainingConfig(
        num_iterations=2,
        games_per_iteration=5,
        num_simulations=50,
        epochs_per_iteration=2,
        batch_size=16,
        learning_rate=0.001,
        num_res_blocks=5,
        num_channels=64,
        device="cpu",
    )

    trainer = AlphaZeroTrainer(config)
    trainer.train()

    print("\n✓ Quick test complete!")


if __name__ == "__main__":
    # For testing
    quick_test()

    # For full training
    # main()
