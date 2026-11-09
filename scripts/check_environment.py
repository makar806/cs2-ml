import json
import platform
import cv2
import torch
import ultralytics

if __name__ == '__main__':
    available = torch.cuda.is_available()
    result = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'cuda_available': available,
        'opencv': cv2.__version__,
        'ultralytics': ultralytics.__version__,
    }
    if available:
        result['gpu'] = torch.cuda.get_device_name(0)
        x = torch.randn(256, 256, device='cuda')
        y = x @ x.T
        torch.cuda.synchronize()
        assert torch.isfinite(y).all().item()
        result['gpu_calculation'] = 'passed'
    print(json.dumps(result, indent=2))
    if not available:
        raise SystemExit('CUDA is not available; GPU setup is incomplete.')
