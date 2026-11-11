import warnings
import logging

# ------- Log and warning control ---------
warnings.filterwarnings("ignore") # hide general warnings
logging.getLogger("timm").setLevel(logging.ERROR) # hide timm internal logs
logging.getLogger("torch").setLevel(logging.ERROR) # hide torch logs

from .models import ms_eff_vit, ms_eff_gcvit
from .models.ms_eff_vit import (
    MultiScaleEffViT,
    ms_eff_vit_b0,
    ms_eff_vit_b5,
)
from .models.ms_eff_gcvit import (
    MultiScaleEffGCViT,
    ms_eff_gcvit_b0,
    ms_eff_gcvit_b5,
)

__all__ = [
    "MultiScaleEffViT",
    "ms_eff_vit_b0",
    "ms_eff_vit_b5",
    "MultiScaleEffGCViT",
    "ms_eff_gcvit_b0",
    "ms_eff_gcvit_b5",
]