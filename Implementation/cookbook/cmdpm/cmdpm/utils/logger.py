import os
import datetime

class TrainingLogger:
    def __init__(self, log_dir="logs/"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir,
            f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        self._initialize_log_file()

    def _initialize_log_file(self):
        """Initialize log file with headers."""
        with open(self.log_file, "w") as f:
            f.write("Epoch\tTrain Loss\tVal Loss\tVal Accuracy\n")

    def log_step(self, loss):
        """Log training step (optional)."""
        pass  # Can be implemented for per-step logging

    def log_epoch(self, epoch, train_loss, val_loss, val_accuracy):
        """Log epoch-level metrics."""
        with open(self.log_file, "a") as f:
            f.write(f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_accuracy:.4f}\n")