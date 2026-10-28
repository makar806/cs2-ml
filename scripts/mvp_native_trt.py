"""Small TensorRT/CUDA runtime. No torch, Ultralytics, training or image history."""

import ctypes as C
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from mvp_core import Detection

ROOT = Path(__file__).resolve().parents[1]


def decode_pose(output, original, ratio, pad, confidence=0.35):
    rows = np.asarray(output)[0].T
    if rows.shape[1] != 9:
        raise ValueError(f"Expected two classes and one Head, got {rows.shape}")
    scores = rows[:, 4:6]
    classes = scores.argmax(axis=1)
    conf = scores.max(axis=1)
    indices = np.flatnonzero((conf >= confidence) & np.isfinite(rows).all(axis=1))
    if not len(indices):
        return []
    indices = indices[np.argsort(conf[indices])[::-1][:30000]]
    xywh = rows[indices, :4].copy()
    xywh[:, :2] -= xywh[:, 2:] / 2

    offset = classes[indices] * 7680
    nms_boxes = xywh.copy()
    nms_boxes[:, 0] += offset
    nms_boxes[:, 1] += offset
    keep = cv2.dnn.NMSBoxes(nms_boxes.tolist(), conf[indices].tolist(), confidence, 0.7)
    width, height = original
    px, py = pad
    detections = []
    for local in np.asarray(keep).reshape(-1)[:300]:
        i = int(indices[int(local)])
        cx, cy, w, h = rows[i, :4]
        hx, hy, hc = rows[i, 6:9]
        box = (
            (cx - w / 2 - px) / ratio,
            (cy - h / 2 - py) / ratio,
            (cx + w / 2 - px) / ratio,
            (cy + h / 2 - py) / ratio,
        )
        box = tuple(
            float(np.clip(v, 0, width if j % 2 == 0 else height)) for j, v in enumerate(box)
        )
        detections.append(
            Detection(
                int(classes[i]),
                float(conf[i]),
                box,
                (
                    float(np.clip((hx - px) / ratio, 0, width)),
                    float(np.clip((hy - py) / ratio, 0, height)),
                ),
                float(hc),
            )
        )
    return detections


class NativeTensorRT:
    def __init__(self, path, size=1280, blocking_wait=False):
        import tensorrt as trt

        self.trt = trt
        self.size = size
        self.buffers = {}
        self.stream = C.c_void_p()
        self.shape = None
        cv2.setNumThreads(1)

        cuda_path = Path(sys.prefix) / "Lib/site-packages/torch/lib/cudart64_12.dll"
        if not cuda_path.exists():
            raise FileNotFoundError(cuda_path)
        self.cuda = C.WinDLL(str(cuda_path))
        self._bind()
        self.check(self.cuda.cudaSetDevice(0))
        least = C.c_int()
        greatest = C.c_int()
        self.check(self.cuda.cudaDeviceGetStreamPriorityRange(C.byref(least), C.byref(greatest)))
        self.check(self.cuda.cudaStreamCreateWithPriority(C.byref(self.stream), 1, least.value))
        self.completion = C.c_void_p()
        self.blocking_wait = blocking_wait

        if blocking_wait:
            self.check(self.cuda.cudaEventCreateWithFlags(C.byref(self.completion), 1 | 2))
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        data = Path(path).read_bytes()
        metadata_length = int.from_bytes(data[:4], "little")
        if 0 < metadata_length < 100000:
            json.loads(data[4 : 4 + metadata_length])
            data = data[4 + metadata_length :]
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError("TensorRT deserialization failed")
        self.context = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [
            name
            for name in self.names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        outputs = [
            name
            for name in self.names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one image and one pose output")
        self.input_name = inputs[0]
        self.output_name = outputs[0]

    def _bind(self):
        declarations = {
            "cudaSetDevice": [C.c_int],
            "cudaDeviceGetStreamPriorityRange": [C.POINTER(C.c_int), C.POINTER(C.c_int)],
            "cudaStreamCreateWithPriority": [C.POINTER(C.c_void_p), C.c_uint, C.c_int],
            "cudaStreamSynchronize": [C.c_void_p],
            "cudaStreamDestroy": [C.c_void_p],
            "cudaEventCreateWithFlags": [C.POINTER(C.c_void_p), C.c_uint],
            "cudaEventRecord": [C.c_void_p, C.c_void_p],
            "cudaEventSynchronize": [C.c_void_p],
            "cudaEventDestroy": [C.c_void_p],
            "cudaMalloc": [C.POINTER(C.c_void_p), C.c_size_t],
            "cudaFree": [C.c_void_p],
            "cudaHostAlloc": [C.POINTER(C.c_void_p), C.c_size_t, C.c_uint],
            "cudaFreeHost": [C.c_void_p],
            "cudaMemcpyAsync": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p],
        }
        for name, args in declarations.items():
            fn = getattr(self.cuda, name)
            fn.argtypes = args
            fn.restype = C.c_int

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"CUDA runtime error {code}")

    def _release_buffers(self):
        for device, host, array in self.buffers.values():
            self.check(self.cuda.cudaFree(device))
            self.check(self.cuda.cudaFreeHost(host))
        self.buffers.clear()

    def _configure(self, shape):
        if shape == self.shape:
            return
        self.check(self.cuda.cudaStreamSynchronize(self.stream))
        self._release_buffers()
        if not self.context.set_input_shape(self.input_name, shape):
            raise ValueError(f"Unsupported input shape {shape}")
        for name in self.names:
            dims = tuple(self.context.get_tensor_shape(name))
            if any(v <= 0 for v in dims):
                raise ValueError(f"Unresolved shape {name}: {dims}")
            dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
            nbytes = math.prod(dims) * dtype.itemsize
            device = C.c_void_p()
            host = C.c_void_p()
            self.check(self.cuda.cudaMalloc(C.byref(device), nbytes))
            self.check(self.cuda.cudaHostAlloc(C.byref(host), nbytes, 0))
            array = np.frombuffer(
                (C.c_ubyte * nbytes).from_address(host.value), dtype=dtype
            ).reshape(dims)
            self.buffers[name] = (device, host, array)
            if not self.context.set_tensor_address(name, device.value):
                raise RuntimeError(f"Bind failed: {name}")
        self.shape = shape

    def predict(self, image):
        started = time.perf_counter()
        height, width = image.shape[:2]
        ratio = min(self.size / height, self.size / width)
        rw, rh = round(width * ratio), round(height * ratio)
        padded_w = math.ceil(rw / 32) * 32
        padded_h = math.ceil(rh / 32) * 32
        left = round((padded_w - rw) / 2 - 0.1)
        top = round((padded_h - rh) / 2 - 0.1)
        self._configure((1, 3, padded_h, padded_w))
        resized = (
            cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
            if (rw, rh) != (width, height)
            else image
        )
        tensor = self.buffers[self.input_name][2]
        tensor.fill(114 / 255)
        np.multiply(
            resized[:, :, ::-1].transpose(2, 0, 1),
            1 / 255,
            out=tensor[0, :, top : top + rh, left : left + rw],
            casting="unsafe",
        )
        device, host, array = self.buffers[self.input_name]
        prepared = time.perf_counter()
        self.check(self.cuda.cudaMemcpyAsync(device, host, array.nbytes, 1, self.stream))
        if not self.context.execute_async_v3(stream_handle=self.stream.value):
            raise RuntimeError("TensorRT enqueue failed")
        device, host, array = self.buffers[self.output_name]
        self.check(self.cuda.cudaMemcpyAsync(host, device, array.nbytes, 2, self.stream))
        submitted = time.perf_counter()
        if self.blocking_wait:
            self.check(self.cuda.cudaEventRecord(self.completion, self.stream))
            self.check(self.cuda.cudaEventSynchronize(self.completion))
        else:
            self.check(self.cuda.cudaStreamSynchronize(self.stream))
        synchronized = time.perf_counter()
        result = decode_pose(array, (width, height), ratio, (left, top))
        self.last_timings = dict(
            preprocess_ms=(prepared - started) * 1000,
            submit_ms=(submitted - prepared) * 1000,
            wait_ms=(synchronized - submitted) * 1000,
            decode_ms=(time.perf_counter() - synchronized) * 1000,
        )
        return result

    def close(self):
        if self.stream:
            self.check(self.cuda.cudaStreamSynchronize(self.stream))
            self._release_buffers()
            self.context = None
            self.engine = None
            self.runtime = None
            if self.completion:
                self.check(self.cuda.cudaEventDestroy(self.completion))
                self.completion = C.c_void_p()
            self.check(self.cuda.cudaStreamDestroy(self.stream))
            self.stream = C.c_void_p()
