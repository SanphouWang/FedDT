from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
from src.utils.generation_models import CycleGAN
from src.utils.classification_models import ResNet50, MobileNet


def get_model_arch(model_name):
    # static means the model arch is fixed.
    static = {"cyclegan": CycleGAN, "resnet50": ResNet50, "mobilenet": MobileNet}
    if model_name in static:
        return static[model_name]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
