import random
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ResidualBlock(nn.Module):
    """A residual block with two convolutional layers"""

    def __init__(self, channels: int = 256) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """AlphaZero-style neural network for chess"""

    def __init__(self, num_res_blocks: int = 20, num_channels: int = 256, input_planes: int = 106) -> None:
        super(AlphaZeroNet, self).__init__()

        # Initial convolutional block
        self.conv_input = nn.Conv2d(input_planes, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4672)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial conv block
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


def alphazero_loss(
    policy_logits: torch.Tensor,
    policy_targets: torch.Tensor,
    value_preds: torch.Tensor,
    value_targets: torch.Tensor,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss function for AlphaZero

    Args:
        policy_logits: Raw policy outputs from network [batch_size, 4672]
        policy_targets: Target policy from MCTS [batch_size, 4672]
        value_preds: Predicted values from network [batch_size, 1]
        value_targets: Actual game outcomes [batch_size, 1]
        policy_weight: Weight for policy loss
        value_weight: Weight for value loss

    Returns:
        Tuple of (total_loss, policy_loss, value_loss)
    """
    # Policy loss: Cross-entropy between predicted and target policy
    # We use log_softmax + NLLLoss, which is equivalent to CrossEntropyLoss
    # but allows us to work with target distributions (not just class indices)
    policy_loss = -torch.sum(policy_targets * F.log_softmax(policy_logits, dim=1)) / policy_logits.size(0)

    # Value loss: Mean squared error between predicted and actual outcome
    value_loss = F.mse_loss(value_preds, value_targets)

    # Combined loss
    total_loss = policy_weight * policy_loss + value_weight * value_loss

    return total_loss, policy_loss, value_loss


def train_step(
    model: AlphaZeroNet,
    optimizer: optim.Optimizer,
    states: torch.Tensor,
    policy_targets: torch.Tensor,
    value_targets: torch.Tensor,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, float]:
    """
    Perform one training step

    Args:
        model: The AlphaZero neural network
        optimizer: PyTorch optimizer
        states: Board positions [batch_size, input_planes, 8, 8]
        policy_targets: Target policies from MCTS [batch_size, 4672]
        value_targets: Game outcomes [batch_size, 1]
        device: 'cpu' or 'cuda'

    Returns:
        Dictionary with losses
    """
    model.train()  # Set to training mode

    # Move data to device
    states = states.to(device)
    policy_targets = policy_targets.to(device)
    value_targets = value_targets.to(device)

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    policy_logits, value_preds = model(states)

    # Calculate loss
    total_loss, policy_loss, value_loss = alphazero_loss(policy_logits, policy_targets, value_preds, value_targets)

    # Backward pass
    total_loss.backward()

    # Update weights
    optimizer.step()

    return {"total_loss": total_loss.item(), "policy_loss": policy_loss.item(), "value_loss": value_loss.item()}


def training_loop(
    model: AlphaZeroNet,
    train_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    device: Union[str, torch.device] = "cpu",
) -> None:
    """
    Full training loop example

    Args:
        model: The AlphaZero neural network
        train_data: List of (state, policy_target, value_target) tuples
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: 'cpu' or 'cuda'
    """
    model = model.to(device)

    # Optimizer: Adam is commonly used for AlphaZero
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler (optional but recommended)
    scheduler: optim.lr_scheduler.StepLR = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(epochs):
        epoch_losses: Dict[str, float] = {"total": 0.0, "policy": 0.0, "value": 0.0}
        num_batches: int = 0

        # Shuffle data each epoch
        random.shuffle(train_data)

        # Process in batches
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]

            # Unpack batch
            states: torch.Tensor = torch.stack([item[0] for item in batch])
            policy_targets: torch.Tensor = torch.stack([item[1] for item in batch])
            value_targets: torch.Tensor = torch.stack([item[2] for item in batch])

            # Training step
            losses: Dict[str, float] = train_step(model, optimizer, states, policy_targets, value_targets, device)

            # Accumulate losses
            epoch_losses["total"] += losses["total_loss"]
            epoch_losses["policy"] += losses["policy_loss"]
            epoch_losses["value"] += losses["value_loss"]
            num_batches += 1

        # Update learning rate
        scheduler.step()

        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Total Loss:  {epoch_losses['total'] / num_batches:.4f}")
        print(f"  Policy Loss: {epoch_losses['policy'] / num_batches:.4f}")
        print(f"  Value Loss:  {epoch_losses['value'] / num_batches:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print()


# Example usage
if __name__ == "__main__":
    # Set device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    model: AlphaZeroNet = AlphaZeroNet(num_res_blocks=10, num_channels=128, input_planes=106)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate dummy training data
    # In real AlphaZero, this comes from self-play games + MCTS
    num_samples: int = 1000
    train_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for _ in range(num_samples):
        # Random board position
        state: torch.Tensor = torch.randn(106, 8, 8)

        # Random policy (would come from MCTS in real training)
        # Normalized to sum to 1
        policy: torch.Tensor = torch.rand(4672)
        policy = policy / policy.sum()

        # Random game outcome: -1 (loss), 0 (draw), or 1 (win)
        value: torch.Tensor = torch.tensor([random.choice([-1.0, 0.0, 1.0])]).unsqueeze(0)

        train_data.append((state, policy, value))

    # Train the model
    training_loop(model=model, train_data=train_data, epochs=5, batch_size=32, lr=0.001, device=device)

    # Save the trained model
    torch.save(model.state_dict(), "alphazero_checkpoint.pth")
    print("Model saved!")

    # To load later:
    # model.load_state_dict(torch.load('alphazero_checkpoint.pth'))
