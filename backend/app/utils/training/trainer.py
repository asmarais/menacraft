"""
Trainer module for DeepFakeDetector model training.
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import wandb
import numpy as np

from data.dataset import FaceDataset
from data.dataloader import DataLoaderWrapper
from core.model import DeepFakeDetector
from config import PROJECT_ROOT, MODEL_DIR



@dataclass
class DatasetConfig:
    """Configuration for datasets."""
    train_csv: str = str(PROJECT_ROOT / "data" / "preprocessed_dataset" / "train.csv")
    test_csv: str = str(PROJECT_ROOT / "data" / "preprocessed_dataset" / "test.csv")
    eval_csv: str = str(PROJECT_ROOT / "data" / "preprocessed_dataset" / "eval.csv")
    
@dataclass
class TestDatasetConfig:
    """Configuration for test datasets."""
    train_csv: str = str(PROJECT_ROOT / "data" / "test_preprocessed_dataset" / "train.csv")
    test_csv: str = str(PROJECT_ROOT / "data" / "test_preprocessed_dataset" / "test.csv")
    eval_csv: str = str(PROJECT_ROOT / "data" / "test_preprocessed_dataset" / "eval.csv")


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Device
    device: torch.device = field(default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    test: bool = False

    # Dataset paths (set in __post_init__ based on test flag)
    dataset_config: Optional[Union[DatasetConfig, TestDatasetConfig]] = None

    # Training hyperparameters
    batch_size: int = 64 # Optimal for GPU 16GB (use 16 if GPU 8GB)
    learning_rate: float = 1e-4  # Maximum learning rate
    min_learning_rate: float = 1e-6  # Minimum learning rate for scheduler
    weight_decay: float = 1e-4  # Standard L2 regularization
    num_epochs: int = 25  # Full training
    num_workers: int = 8
    warmup_epochs: int = 3  # Number of warmup epochs
    # utils
    compile_model: bool = False
    seed : int = 42
    use_automixed_precision: bool = False

    # Logging
    log_interval: int = 10
    eval_interval: int = 1  # Evaluate every N epochs
    log_gradients: bool = True  # Log gradient histograms to wandb
    log_weights: bool = True  # Log weight histograms to wandb

    # Model saving
    save_dir: str = str(MODEL_DIR)
    model_name: str = "deepfake_detector"

    # Early stopping
    early_stopping_patience: int = 5

    # Wandb
    wandb_project: str = "deepfake-detector"
    wandb_entity: Optional[str] = None
    use_wandb: bool = True

    def __post_init__(self):
        if self.dataset_config is None:
            self.dataset_config = TestDatasetConfig() if self.test else DatasetConfig()


class Trainer:
    """
    Trainer for DeepFakeDetector model.

    Handles training loop, validation, logging, and model checkpointing.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize Trainer.

        Args:
            config: Training configuration. Uses defaults if None.
        """
        self.config = config or TrainingConfig()
        self.logger = logging.getLogger("training.Trainer")
        self.device = self.config.device

        # Initialize wandb (will be configured later after model/dataloaders are created)
        self.wandb_run = None

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(self.config.seed)

        torch.set_float32_matmul_precision('high')

        # Load datasets
        self.logger.info("Loading datasets...")
        self.train_dataset = FaceDataset(
            csv_path=self.config.dataset_config.train_csv,
            augment=True
        )
        self.test_dataset = FaceDataset(
            csv_path=self.config.dataset_config.test_csv,
            augment=False
        )
        self.eval_dataset = FaceDataset(
            csv_path=self.config.dataset_config.eval_csv,
            augment=False
        )

        # Create dataloaders
        self.train_loader = DataLoaderWrapper(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            device=self.device,
            drop_last=True
        )
        self.test_loader = DataLoaderWrapper(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            device=self.device
        )
        self.eval_loader = DataLoaderWrapper(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            device=self.device
        )

        # Initialize model
        self.logger.info("Initializing model...")
        self.model = DeepFakeDetector().to(self.device)
        
        if self.config.compile_model:
            self.logger.info("Compiling model for optimized performance...")
            self.model = torch.compile(self.model)
            self.logger.info("Model compilation complete.")

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # GradScaler for mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if self.config.use_automixed_precision and self.device.type == 'cuda' else None

        # Learning rate scheduler with warmup + cosine annealing
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0

        # Create save directory
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)

        # Initialize wandb now that model and dataloaders are ready
        if self.config.use_wandb:
            self._init_wandb()

        self.logger.info(
            f"Trainer initialized:\n"
            f"  - Device: {self.device}\n"
            f"  - Train samples: {len(self.train_dataset)}\n"
            f"  - Test samples: {len(self.test_dataset)}\n"
            f"  - Eval samples: {len(self.eval_dataset)}\n"
            f"  - Batch size: {self.config.batch_size}\n"
            f"  - Learning rate: {self.config.learning_rate}\n"
            f"  - Epochs: {self.config.num_epochs}"
        )

    def _create_scheduler(self):
        """
        Create learning rate scheduler with warmup + cosine annealing.

        Returns:
            Learning rate scheduler
        """
        def lr_lambda(epoch):
            # Warmup phase: linear increase from 0 to max_lr
            if epoch < self.config.warmup_epochs:
                return (epoch + 1) / self.config.warmup_epochs
            # Cosine annealing phase: smooth decrease from max_lr to min_lr
            else:
                progress = (epoch - self.config.warmup_epochs) / max(1, self.config.num_epochs - self.config.warmup_epochs)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                lr_range = 1.0 - (self.config.min_learning_rate / self.config.learning_rate)
                return (self.config.min_learning_rate / self.config.learning_rate) + lr_range * cosine_decay

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        run_name = f"veridisquo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.wandb_run = wandb.init(
            entity=self.config.wandb_entity,
            project=self.config.wandb_project,
            name=run_name,
            config={
                "learning_rate": self.config.learning_rate,
                "min_learning_rate": self.config.min_learning_rate,
                "batch_size": self.config.batch_size,
                "num_epochs": self.config.num_epochs,
                "warmup_epochs": self.config.warmup_epochs,
                "weight_decay": self.config.weight_decay,
                "architecture": "DeepFakeDetector",
                "optimizer": "AdamW",
                "scheduler": "Warmup + Cosine Annealing",
                "use_automixed_precision": self.config.use_automixed_precision,
            },
            reinit="finish_previous"
        )
        # Watch model for gradient and parameter tracking
        if self.config.log_gradients or self.config.log_weights:
            log_freq = max(100, len(self.train_loader) // 10)  # Log ~10 times per epoch
            wandb.watch(
                self.model,
                log="all" if self.config.log_gradients and self.config.log_weights else "gradients" if self.config.log_gradients else "parameters",
                log_freq=log_freq
            )
        self.logger.info(f"Wandb initialized: {self.wandb_run.name}")

    def train(self) -> Dict[str, float]:
        """
        Run full training loop.

        Returns:
            Dictionary with final metrics
        """
        self.logger.info("Starting training...")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Training epoch
            train_metrics = self._train_epoch()

            # Validation
            if (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self._validate()

                # Log metrics
                self._log_metrics(train_metrics, val_metrics, epoch)

                # Check for improvement and save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.best_val_acc = val_metrics['accuracy']
                    self.epochs_without_improvement = 0
                    self._save_model(is_best=True)
                else:
                    self.epochs_without_improvement += 1

                # Early stopping
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    self.logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs"
                    )
                    break

            # Learning rate scheduling (step after each epoch)
            self.scheduler.step()

            # Save latest checkpoint
            self._save_model(is_latest=True)

        # Final evaluation on test set
        self.logger.info("Running final evaluation on test set...")
        test_metrics = self._evaluate(self.test_loader, "test")

        # Log final test metrics to wandb
        if self.wandb_run:
            wandb.log({
                "test/loss": test_metrics['loss'],
                "test/accuracy": test_metrics['accuracy'],
                "test/precision": test_metrics['precision'],
                "test/recall": test_metrics['recall'],
                "test/f1": test_metrics['f1'],
            })
            # Log confusion matrix
            self._log_confusion_matrix(test_metrics)
            self.wandb_run.finish()

        self.logger.info(
            f"Training complete!\n"
            f"  - Best val loss: {self.best_val_loss:.4f}\n"
            f"  - Best val accuracy: {self.best_val_acc:.4f}\n"
            f"  - Test accuracy: {test_metrics['accuracy']:.4f}"
        )

        return test_metrics

    def _train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        # Accumulate tensors on GPU
        batch_losses = []
        batch_corrects = []
        total_samples = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.num_epochs}",
            leave=True
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            # Forward pass
            loss, correct = self._train_step(images, labels)

            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(
                    f"NaN or Inf loss detected at batch {batch_idx}!\n"
                    f"  - Loss value: {loss.item()}\n"
                    f"  - Images shape: {images.shape}\n"
                    f"  - Images min/max: {images.min().item():.4f}/{images.max().item():.4f}\n"
                    f"  - Labels: {labels}\n"
                )
                raise ValueError("Training stopped due to NaN/Inf loss")

            # Accumulate tensors (keep on GPU)
            batch_losses.append(loss * images.size(0))
            batch_corrects.append(correct)
            total_samples += images.size(0)

            # Update progress bar and log every 10 batches (periodic sync to CPU)
            if batch_idx % self.config.log_interval == 0:
                # Calculate running metrics from recent batches
                recent_start = max(0, len(batch_losses) - self.config.log_interval)
                recent_losses = torch.stack(batch_losses[recent_start:])
                recent_corrects = torch.stack(batch_corrects[recent_start:])
                recent_samples = min(total_samples, self.config.log_interval * images.size(0))

                running_loss = recent_losses.sum().item() / recent_samples
                running_acc = recent_corrects.sum().item() / recent_samples

                pbar.set_postfix({
                    'loss': f'{running_loss:.4f}',
                    'acc': f'{running_acc:.4f}'
                })

                # Log to wandb
                if self.wandb_run:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/batch_acc": (correct.float() / images.size(0)).item(),
                        "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                        "global_step": self.global_step
                    })

            self.global_step += 1

        # End of epoch: single GPU aggregation then sync to CPU
        total_loss_tensor = torch.stack(batch_losses).sum()
        total_correct_tensor = torch.stack(batch_corrects).sum()

        return {
            'loss': (total_loss_tensor / total_samples).item(),
            'accuracy': (total_correct_tensor.float() / total_samples).item()
        }

    def _train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform single training step.

        Args:
            images: Batch of images [B, 3, 224, 224]
            labels: Batch of labels [B]

        Returns:
            Tuple of (loss tensor, correct count tensor)
        """
        self.optimizer.zero_grad()

        # Forward pass with mixed precision
        if self.config.use_automixed_precision and self.scaler is not None:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first for accurate clipping)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

        # Calculate accuracy (keep as tensor)
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == labels).sum()

        return loss.detach(), correct

    def _validate(self) -> Dict[str, float]:
        """
        Run validation on eval set.

        Returns:
            Dictionary with validation metrics
        """
        return self._evaluate(self.eval_loader, "val")

    def _evaluate(
        self,
        dataloader: DataLoaderWrapper,
        split_name: str
    ) -> Dict[str, float]:
        """
        Evaluate model on given dataloader.

        Args:
            dataloader: DataLoader to evaluate on
            split_name: Name of split for logging

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        total_samples = 0

        # Accumulate tensors on GPU
        batch_losses = []
        all_predictions = []  # List of tensors
        all_labels = []       # List of tensors

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)

                # Keep on GPU
                batch_losses.append(loss * images.size(0))
                all_predictions.append(predictions)
                all_labels.append(labels)
                total_samples += images.size(0)

        # Single GPU concatenation
        all_predictions_gpu = torch.cat(all_predictions, dim=0)
        all_labels_gpu = torch.cat(all_labels, dim=0)
        total_loss_tensor = torch.stack(batch_losses).sum()
        total_correct_tensor = (all_predictions_gpu == all_labels_gpu).sum()

        # Calculate basic metrics on GPU
        avg_loss = (total_loss_tensor / total_samples).item()
        accuracy = (total_correct_tensor.float() / total_samples).item()

        # Compute precision/recall/F1 on GPU
        metrics = self._compute_metrics_gpu(all_predictions_gpu, all_labels_gpu)
        metrics['loss'] = avg_loss
        metrics['accuracy'] = accuracy

        # Store predictions and labels for confusion matrix
        metrics['predictions'] = all_predictions_gpu
        metrics['labels'] = all_labels_gpu

        self.logger.info(
            f"{split_name.upper()} - Loss: {avg_loss:.4f}, "
            f"Acc: {accuracy:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}"
        )

        return metrics

    def _compute_metrics(
        self,
        predictions: List[int],
        labels: List[int]
    ) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 score.

        Args:
            predictions: List of predicted labels
            labels: List of true labels

        Returns:
            Dictionary with precision, recall, F1
        """
        # For FAKE detection (class 0)
        tp = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _compute_metrics_gpu(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 score on GPU.

        Args:
            predictions (torch.Tensor): predicted labels [N]
            labels (torch.Tensor): true labels [N]

        Returns:
            Dict[str, float]: metrics dictionary
        """
        # Compute TP, FP, FN, TN for both classes
        # Class 0 (FAKE)
        tp_fake = ((predictions == 0) & (labels == 0)).sum().float()
        fp_fake = ((predictions == 0) & (labels == 1)).sum().float()
        fn_fake = ((predictions == 1) & (labels == 0)).sum().float()
        tn_fake = ((predictions == 1) & (labels == 1)).sum().float()

        # Class 1 (REAL)
        tp_real = ((predictions == 1) & (labels == 1)).sum().float()
        fp_real = ((predictions == 1) & (labels == 0)).sum().float()
        fn_real = ((predictions == 0) & (labels == 1)).sum().float()
        tn_real = ((predictions == 0) & (labels == 0)).sum().float()

        # Compute metrics for FAKE class (primary task)
        precision = tp_fake / (tp_fake + fp_fake + 1e-10)
        recall = tp_fake / (tp_fake + fn_fake + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        # Compute per-class metrics
        precision_real = tp_real / (tp_real + fp_real + 1e-10)
        recall_real = tp_real / (tp_real + fn_real + 1e-10)
        f1_real = 2 * precision_real * recall_real / (precision_real + recall_real + 1e-10)

        # Single sync to CPU
        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'precision_fake': precision.item(),
            'recall_fake': recall.item(),
            'f1_fake': f1.item(),
            'precision_real': precision_real.item(),
            'recall_real': recall_real.item(),
            'f1_real': f1_real.item(),
            'tp_fake': tp_fake.item(),
            'fp_fake': fp_fake.item(),
            'fn_fake': fn_fake.item(),
            'tn_fake': tn_fake.item(),
        }

    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int
    ) -> None:
        """Log metrics to wandb and console."""
        self.logger.info(
            f"Epoch {epoch + 1}/{self.config.num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        if self.wandb_run:
            # Basic metrics
            metrics_dict = {
                "epoch": epoch + 1,
                "train/loss": train_metrics['loss'],
                "train/accuracy": train_metrics['accuracy'],
                "val/loss": val_metrics['loss'],
                "val/accuracy": val_metrics['accuracy'],
                "val/precision": val_metrics['precision'],
                "val/recall": val_metrics['recall'],
                "val/f1": val_metrics['f1'],
                "learning_rate": self.optimizer.param_groups[0]['lr'],
            }

            # Per-class metrics
            if 'precision_fake' in val_metrics:
                metrics_dict.update({
                    "val/precision_fake": val_metrics['precision_fake'],
                    "val/recall_fake": val_metrics['recall_fake'],
                    "val/f1_fake": val_metrics['f1_fake'],
                    "val/precision_real": val_metrics['precision_real'],
                    "val/recall_real": val_metrics['recall_real'],
                    "val/f1_real": val_metrics['f1_real'],
                })

            # Confusion matrix elements
            if 'tp_fake' in val_metrics:
                metrics_dict.update({
                    "val/tp_fake": val_metrics['tp_fake'],
                    "val/fp_fake": val_metrics['fp_fake'],
                    "val/fn_fake": val_metrics['fn_fake'],
                    "val/tn_fake": val_metrics['tn_fake'],
                })

            wandb.log(metrics_dict)

            # Log confusion matrix visualization
            self._log_confusion_matrix(val_metrics, prefix="val")

    def _log_confusion_matrix(self, metrics: Dict, prefix: str = "test") -> None:
        """
        Log confusion matrix to wandb.

        Args:
            metrics: Dictionary containing predictions and labels
            prefix: Prefix for wandb logging (e.g., "val" or "test")
        """
        if not self.wandb_run or 'predictions' not in metrics or 'labels' not in metrics:
            return

        # Move to CPU for wandb logging
        predictions = metrics['predictions'].cpu().numpy()
        labels = metrics['labels'].cpu().numpy()

        # Create confusion matrix
        wandb.log({
            f"{prefix}/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=labels,
                preds=predictions,
                class_names=["FAKE", "REAL"]
            )
        })

    def _save_model(self, is_best: bool = False, is_latest: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far (lowest validation loss)
            is_latest: Whether this is the latest model (most recent epoch)
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }

        if is_best:
            path = Path(self.config.save_dir) / f"{self.config.model_name}_best.pth"
            torch.save(checkpoint, path)
            self.logger.info(f"💾 Saved BEST model to {path} (val_loss: {self.best_val_loss:.4f})")

        if is_latest:
            path = Path(self.config.save_dir) / f"{self.config.model_name}_latest.pth"
            torch.save(checkpoint, path)
            # Only log latest save every 10 epochs to reduce noise
            if (self.current_epoch + 1) % 10 == 0:
                self.logger.info(f"💾 Saved LATEST model to {path} (epoch {self.current_epoch + 1})")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']

        self.logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(epoch {self.current_epoch + 1})"
        )
