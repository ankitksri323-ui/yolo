import platform
import sys
import winsound
from pathlib import Path

import cv2
import numpy as np
import torch

if platform.system() == "Windows":
    import pathlib

    pathlib.PosixPath = pathlib.WindowsPath

# Set YOLOv5 root path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# YOLOv5 imports
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Load model
weights_path = ROOT / "f_mod.pt"
device = select_device("0" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(str(weights_path), device=device)
stride, class_names, is_pt = model.stride, model.names, model.pt
img_size = (640, 640)
model.warmup(imgsz=(1, 3, *img_size))

# Load video
cap = cv2.VideoCapture(str(ROOT / "fire.mp4"))
# cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, img_size)
    img = resized[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    preds = model(img_tensor)
    preds = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)

    fire_detected = False

    for det in preds:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                cls_name = class_names[int(cls)]
                label = f"{cls_name} {conf:.2f}"
                x1, y1, x2, y2 = map(int, xyxy)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if "fire" in cls_name.lower():
                    fire_detected = True

    if fire_detected:
        cv2.putText(frame, "Fire Detected!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        winsound.Beep(1500, 500)  # Play beep if fire is detected

    cv2.imshow("Fire Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
