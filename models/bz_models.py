"""
Model architectures for IMF Bz forecasting using Surya.

Three variants:
1. Full fine-tuning: Train entire Surya model
2. Frozen encoder: Freeze Surya, train only prediction head
3. LoRA: Parameter-efficient fine-tuning with low-rank adaptation

Author: Vineet Vora
Date: 2025-11-27
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from peft import LoraConfig, get_peft_model, TaskType
import sys
import os

# Add parent directory to path for Surya imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from surya.models.helio_spectformer import HelioSpectFormer


class BzPredictionHead(nn.Module):
    """
    Prediction head for multi-horizon Bz forecasting.

    Takes encoded features from Surya and predicts Bz at multiple future time horizons.
    Default: Predicts Bz at T+24h, T+48h, T+72h (1-3 days ahead) to account for
    CME/solar wind propagation time from Sun to Earth.
    """

    def __init__(
        self,
        input_dim: int = 768,  # Typical transformer hidden dim
        num_horizons: int = 3,  # Number of forecast horizons (e.g., 24h, 48h, 72h)
        hidden_dims: list = [512, 256],
        dropout: float = 0.2,
    ):
        """
        Initialize Bz prediction head.

        Args:
            input_dim: Dimension of input features from encoder
            num_horizons: Number of forecast horizons (default: 3 for 24h, 48h, 72h)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.num_horizons = num_horizons

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer (one value per forecast horizon)
        layers.append(nn.Linear(prev_dim, num_horizons))

        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Encoded features, shape (B, D) or (B, T, D)

        Returns:
            Bz predictions, shape (B, num_horizons)
            e.g., [Bz(T+24h), Bz(T+48h), Bz(T+72h)] for each sample in batch
        """
        # Handle both 2D and 3D inputs
        if features.dim() == 3:
            # Take mean over temporal dimension or last timestep
            features = features[:, -1, :]  # Use last timestep

        return self.mlp(features)


class SuryaBzModel(nn.Module):
    """
    Complete model combining Surya encoder with Bz prediction head.
    Predicts IMF Bz at multiple future time horizons (default: 24h, 48h, 72h).
    """

    def __init__(
        self,
        surya_config: Dict[str, Any],
        num_horizons: int = 3,  # Number of forecast horizons (default: 3 for 24h/48h/72h)
        freeze_encoder: bool = False,
        prediction_head_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Surya-Bz model.

        Args:
            surya_config: Configuration dict for Surya model
            num_horizons: Number of forecast horizons (default: 3 for 24h/48h/72h prediction)
            freeze_encoder: Whether to freeze Surya encoder
            prediction_head_config: Configuration for prediction head
        """
        super().__init__()

        self.num_horizons = num_horizons
        self.freeze_encoder = freeze_encoder

        # Initialize Surya encoder
        self.encoder = HelioSpectFormer(**surya_config)

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Surya encoder frozen")

        # Get encoder output dimension
        # Create a dummy input to determine actual encoder output shape
        with torch.no_grad():
            dummy_input = torch.randn(1, surya_config['in_chans'], 1,
                                     surya_config['img_size'], surya_config['img_size'])
            dummy_batch = {
                "ts": dummy_input,
                "time_delta_input": torch.zeros(1, 1)
            }
            dummy_output = self.encoder(dummy_batch)
            if isinstance(dummy_output, tuple):
                dummy_output = dummy_output[0]

            # Apply same pooling as in forward
            while dummy_output.dim() > 2:
                dummy_output = dummy_output.mean(dim=-1)

            encoder_output_dim = dummy_output.shape[-1]
            print(f"Detected encoder output dimension: {encoder_output_dim}")

        # Initialize prediction head
        if prediction_head_config is None:
            prediction_head_config = {}

        self.prediction_head = BzPredictionHead(
            input_dim=encoder_output_dim,
            num_horizons=num_horizons,
            **prediction_head_config
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input SDO images, shape (B, C, H, W) or (B, C, T, H, W)
            return_features: Whether to return encoder features

        Returns:
            Bz predictions, shape (B, num_horizons)
            e.g., [Bz(T+24h), Bz(T+48h), Bz(T+72h)] for each sample
            If return_features=True, returns (predictions, features)
        """
        # Prepare input for Surya encoder
        # Surya expects: batch dict with "ts" (B, C, T, H, W) and "time_delta_input" (B, T)
        if x.dim() == 4:  # (B, C, H, W)
            # Add temporal dimension
            x = x.unsqueeze(2)  # (B, C, 1, H, W)

        B, C, T, H, W = x.shape
        batch = {
            "ts": x,
            "time_delta_input": torch.zeros(B, T, device=x.device)  # Dummy time deltas
        }

        # Get encoder features
        features = self.encoder(batch)

        # Handle different encoder output formats
        # Adjust based on actual Surya output
        if isinstance(features, tuple):
            features = features[0]  # Take first output if multiple

        # Global pooling if needed (spatial dimensions)
        if features.dim() > 2:
            # Assuming features are (B, C, H, W) or (B, T, C, H, W)
            # Apply global average pooling
            while features.dim() > 2:
                features = features.mean(dim=-1)

        # Predict Bz
        predictions = self.prediction_head(features)

        if return_features:
            return predictions, features
        return predictions

    def load_surya_weights(self, checkpoint_path: str):
        """
        Load pretrained Surya weights using filtering approach.
        Only loads weights that match in shape (handles channel and size mismatches).
        Based on solar wind forecasting pipeline approach.
        """
        print(f"Loading Surya weights from {checkpoint_path}")
        checkpoint_state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Handle different checkpoint formats
        if 'model' in checkpoint_state:
            checkpoint_state = checkpoint_state['model']
        elif 'state_dict' in checkpoint_state:
            checkpoint_state = checkpoint_state['state_dict']

        # Get current model state
        model_state = self.encoder.state_dict()

        # Filter: only load weights that exist in model AND have matching shapes
        filtered_checkpoint_state = {
            k: v
            for k, v in checkpoint_state.items()
            if k in model_state and v.shape == model_state[k].shape
        }

        # Count what we're loading vs skipping
        total_checkpoint_keys = len(checkpoint_state)
        loaded_keys = len(filtered_checkpoint_state)
        skipped_keys = total_checkpoint_keys - loaded_keys

        print(f"Loading {loaded_keys}/{total_checkpoint_keys} pretrained weights")
        print(f"Skipping {skipped_keys} weights due to shape mismatch or missing keys")

        # Update model state with filtered weights
        model_state.update(filtered_checkpoint_state)
        self.encoder.load_state_dict(model_state, strict=True)

        print("Surya weights loaded successfully")

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = sum(p.numel() for p in self.prediction_head.parameters())

        return {
            'total': total_params,
            'trainable': trainable_params,
            'encoder': encoder_params,
            'prediction_head': head_params,
            'trainable_percentage': 100 * trainable_params / total_params,
        }


class SuryaBzLoRA(nn.Module):
    """
    Surya-Bz model with LoRA (Low-Rank Adaptation).

    Uses PEFT library for parameter-efficient fine-tuning.
    """

    def __init__(
        self,
        surya_config: Dict[str, Any],
        num_horizons: int = 3,  # Number of forecast horizons (default: 3 for 24h/48h/72h)
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[list] = None,
        prediction_head_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Surya-Bz with LoRA.

        Args:
            surya_config: Configuration for Surya model
            num_horizons: Number of forecast horizons (default: 3 for 24h/48h/72h prediction)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout
            target_modules: List of module names to apply LoRA to
            prediction_head_config: Configuration for prediction head
        """
        super().__init__()

        self.num_horizons = num_horizons

        # Initialize base Surya model
        base_model = HelioSpectFormer(**surya_config)

        # Configure LoRA
        if target_modules is None:
            # Find linear layers in the model to apply LoRA
            # Inspect model structure to find attention-related linear layers
            target_modules = []
            for name, module in base_model.named_modules():
                # Look for attention-related modules (qkv, proj, etc.)
                if any(x in name.lower() for x in ['attn', 'attention']):
                    for sub_name, sub_module in module.named_modules():
                        if isinstance(sub_module, torch.nn.Linear):
                            full_name = f"{name}.{sub_name}" if sub_name else name
                            # Get just the last part of the name
                            layer_name = sub_name if sub_name else name.split('.')[-1]
                            if layer_name and layer_name not in target_modules:
                                target_modules.append(layer_name)

            # If no attention layers found, apply to all linear layers (less common)
            if not target_modules:
                print("Warning: No attention layers found, applying LoRA to all Linear layers")
                target_modules = [".*"]  # Regex to match all

        print(f"LoRA target modules: {target_modules}")

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        # Apply LoRA
        self.encoder = get_peft_model(base_model, lora_config)
        print(f"LoRA applied with r={lora_r}, alpha={lora_alpha}")
        self.encoder.print_trainable_parameters()

        # Get encoder output dimension
        # Create a dummy input to determine actual encoder output shape
        with torch.no_grad():
            dummy_input = torch.randn(1, surya_config['in_chans'], 1,
                                     surya_config['img_size'], surya_config['img_size'])
            dummy_batch = {
                "ts": dummy_input,
                "time_delta_input": torch.zeros(1, 1)
            }
            dummy_output = self.encoder.base_model.model(dummy_batch)
            if isinstance(dummy_output, tuple):
                dummy_output = dummy_output[0]

            # Apply same pooling as in forward
            while dummy_output.dim() > 2:
                dummy_output = dummy_output.mean(dim=-1)

            encoder_output_dim = dummy_output.shape[-1]
            print(f"Detected encoder output dimension: {encoder_output_dim}")

        # Initialize prediction head
        if prediction_head_config is None:
            prediction_head_config = {}

        self.prediction_head = BzPredictionHead(
            input_dim=encoder_output_dim,
            num_horizons=num_horizons,
            **prediction_head_config
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """Forward pass."""
        # Prepare input for Surya encoder
        # Surya expects: batch dict with "ts" (B, C, T, H, W) and "time_delta_input" (B, T)

        # Handle different input shapes
        if x.dim() == 3:  # (B, H, W) - grayscale or single channel
            x = x.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H, W)
        elif x.dim() == 4:  # (B, C, H, W)
            x = x.unsqueeze(2)  # (B, C, 1, H, W)
        elif x.dim() != 5:
            raise ValueError(f"Expected input with 3, 4, or 5 dimensions, got {x.dim()} dimensions with shape {x.shape}")

        B, C, T, H, W = x.shape
        batch = {
            "ts": x,
            "time_delta_input": torch.zeros(B, T, device=x.device)  # Dummy time deltas
        }

        # Get encoder features (PEFT-wrapped encoder)
        # Access the base model directly to avoid PEFT kwargs issues
        features = self.encoder.base_model.model(batch)

        if isinstance(features, tuple):
            features = features[0]

        # Global pooling if needed
        if features.dim() > 2:
            while features.dim() > 2:
                features = features.mean(dim=-1)

        predictions = self.prediction_head(features)

        if return_features:
            return predictions, features
        return predictions

    def load_surya_weights(self, checkpoint_path: str):
        """
        Load pretrained Surya weights into base model using filtering approach.
        Only loads weights that match in shape (handles channel and size mismatches).
        Based on solar wind forecasting pipeline approach.
        """
        print(f"Loading Surya weights from {checkpoint_path}")
        checkpoint_state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # Handle different checkpoint formats
        if 'model' in checkpoint_state:
            checkpoint_state = checkpoint_state['model']
        elif 'state_dict' in checkpoint_state:
            checkpoint_state = checkpoint_state['state_dict']

        # Get current model state (base model for LoRA)
        model_state = self.encoder.base_model.model.state_dict()

        # Filter: only load weights that exist in model AND have matching shapes
        filtered_checkpoint_state = {
            k: v
            for k, v in checkpoint_state.items()
            if k in model_state and v.shape == model_state[k].shape
        }

        # Count what we're loading vs skipping
        total_checkpoint_keys = len(checkpoint_state)
        loaded_keys = len(filtered_checkpoint_state)
        skipped_keys = total_checkpoint_keys - loaded_keys

        print(f"Loading {loaded_keys}/{total_checkpoint_keys} pretrained weights")
        print(f"Skipping {skipped_keys} weights due to shape mismatch or missing keys")

        # Update model state with filtered weights
        model_state.update(filtered_checkpoint_state)
        self.encoder.base_model.model.load_state_dict(model_state, strict=True)

        print("Surya weights loaded successfully")

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'trainable_percentage': 100 * trainable_params / total_params,
        }


def create_bz_model(
    strategy: str,
    surya_config: Dict[str, Any],
    num_horizons: int = 3,  # Number of forecast horizons (default: 3 for 24h/48h/72h)
    surya_checkpoint: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create Bz forecasting model.

    Args:
        strategy: One of ['full', 'frozen', 'lora']
        surya_config: Configuration for Surya model
        num_horizons: Number of forecast horizons (default: 3 for 24h/48h/72h prediction)
        surya_checkpoint: Path to pretrained Surya weights
        **kwargs: Additional arguments for model initialization

    Returns:
        Initialized model
    """
    if strategy == 'full':
        # Full fine-tuning
        # Filter out LoRA-specific kwargs
        non_lora_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ['lora_r', 'lora_alpha', 'lora_dropout']}
        model = SuryaBzModel(
            surya_config=surya_config,
            num_horizons=num_horizons,
            freeze_encoder=False,
            **non_lora_kwargs
        )
        print("Created model with FULL fine-tuning strategy")

    elif strategy == 'frozen':
        # Frozen encoder
        # Filter out LoRA-specific kwargs
        non_lora_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ['lora_r', 'lora_alpha', 'lora_dropout']}
        model = SuryaBzModel(
            surya_config=surya_config,
            num_horizons=num_horizons,
            freeze_encoder=True,
            **non_lora_kwargs
        )
        print("Created model with FROZEN encoder strategy")

    elif strategy == 'lora':
        # LoRA fine-tuning
        model = SuryaBzLoRA(
            surya_config=surya_config,
            num_horizons=num_horizons,
            **kwargs
        )
        print("Created model with LoRA strategy")

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from ['full', 'frozen', 'lora']")

    # Load pretrained weights if provided
    if surya_checkpoint is not None:
        model.load_surya_weights(surya_checkpoint)

    # Print parameter counts
    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for key, value in param_counts.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:,}")

    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing Bz model variants...")

    # Minimal Surya config for testing
    surya_config = {
        'img_size': 512,
        'patch_size': 16,
        'in_chans': 13,
        'embed_dim': 768,
        'depth': 8,
        'num_heads': 12,
    }

    strategies = ['full', 'frozen', 'lora']

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing {strategy.upper()} strategy")
        print('='*60)

        model = create_bz_model(
            strategy=strategy,
            surya_config=surya_config,
            num_horizons=3  # Test with 3 horizons (24h, 48h, 72h)
        )

        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 13, 512, 512)

        print(f"\nInput shape: {x.shape}")

        try:
            output = model(x)
            print(f"Output shape: {output.shape}")
            print(f"Output sample: {output[0].detach().numpy()}")
            print(f"✓ {strategy} model works!")
        except Exception as e:
            print(f"✗ Error with {strategy} model: {e}")

    print("\n" + "="*60)
    print("Model testing complete!")
