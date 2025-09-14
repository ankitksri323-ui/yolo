import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import platform

if platform.system() == 'Windows':
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

import torch.serialization
from models.yolo import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])

# Load model
weights = str(ROOT / 'best.pt') 
device = select_device('0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = (640, 640)

# Warmup
model.warmup(imgsz=(1, 3, *imgsz))

# 🔹 Use webcam (0) OR sample video
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(str(ROOT / 'sample.webm'))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    im = cv2.resize(frame, imgsz)
    img = im[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB, HWC → CHW
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # normalize
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Draw results
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                              (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show output
    cv2.imshow('Helmet Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
