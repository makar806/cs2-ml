"""Verify pretrained inference and a training backward pass, not CS2 accuracy."""
from pathlib import Path
import json
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    model = YOLO(str(ROOT / 'models/yolo11n.pt'))
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    result = model.predict(image, device=0, verbose=False)[0]
    result.save(filename=str(ROOT / 'runs/smoke-detection.jpg'))
    net = YOLO(str(ROOT / 'models/yolo11n.pt')).model.to('cuda').train()
    net.args = deepcopy(DEFAULT_CFG)
    for parameter in net.parameters():
        parameter.requires_grad_(True)
    batch = {
        'img': torch.rand(2, 3, 320, 320, device='cuda'),
        'batch_idx': torch.tensor([0, 1], device='cuda'),
        'cls': torch.tensor([[0.], [0.]], device='cuda'),
        'bboxes': torch.tensor([[.5, .5, .2, .3], [.5, .5, .2, .3]], device='cuda'),
    }
    loss, _ = net(batch)
    loss.sum().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss).all().item()
    assert any(p.grad is not None for p in net.parameters())
    print(json.dumps({'inference': 'passed', 'training_backward': 'passed',
                      'loss': loss.detach().cpu().tolist(), 'device': 'cuda'}, indent=2))
