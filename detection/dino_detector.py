import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from config import DET
#from ..config import DET
try:
    from groundingdino.util.inference import load_model, predict
except Exception as e:
    raise ImportError(
        "groundingdino is not installed or not importable. "
        "Install with `pip install groundingdino`."
    ) from e
class ObstacleDetector:
    def __init__(self, cfg=DET, device=None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if not self.cfg.config_path or not self.cfg.weights_path:
            raise ValueError(
                "Set DET.config_path and DET.weights_path in config.py to your GroundingDINO files."
            )
        self.model = load_model(self.cfg.config_path, self.cfg.weights_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.caption = " . ".join(self.cfg.obstacle_labels) + " ."
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    @torch.no_grad()
    def __call__(self, rgb_np: np.ndarray) -> bool:
        if rgb_np.dtype != np.uint8:
            rgb_np = rgb_np.astype(np.uint8)
        pil = Image.fromarray(rgb_np, mode="RGB")
        image = self.tf(pil)
        boxes, logits, phrases = predict(model=self.model,image=image,caption=self.caption,box_threshold=self.cfg.box_threshold,text_threshold=self.cfg.text_threshold,device=self.device,)
        has_obstacle = boxes is not None and len(boxes) > 0
        return bool(has_obstacle)
