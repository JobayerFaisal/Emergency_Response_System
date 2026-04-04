"""Training components"""

try:
    from training.train_depth import DepthTrainer, quick_train
    from training.dataset_generator import DepthDatasetGenerator
except ImportError:
    from .train_depth import DepthTrainer, quick_train
    from .dataset_generator import DepthDatasetGenerator

__all__ = ['DepthTrainer', 'quick_train', 'DepthDatasetGenerator']