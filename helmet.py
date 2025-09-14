import torch
import cv2
import numpy as np
import sys
import platform
import winsound
from pathlib import Path

if platform.system() == 'Windows':
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath

#YOLOv5 path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# YOLOv5 imports
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

#model
weights_path = ROOT / 'best.pt'
device = select_device('0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(str(weights_path), device=device)
stride, class_names, is_pt = model.stride, model.names, model.pt
img_size = (640, 640)
model.warmup(imgsz=(1, 3, *img_size))

#input
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(str(ROOT / 'Safety Meets Comfort.mp4'))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare image
    resized = cv2.resize(frame, img_size)
    img = resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    preds = model(img_tensor)
    preds = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)

    helmet_found = False

    # Process detections
    for det in preds:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                cls_name = class_names[int(cls)]
                label = f"{cls_name} {conf:.2f}"
                x1, y1, x2, y2 = map(int, xyxy)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if 'helmet' in cls_name.lower():
                    helmet_found = True

    # Alert if no helmet
    if not helmet_found:
        cv2.putText(frame, "No Helmet Detected!", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        winsound.Beep(1000, 500)

    # Show output
    cv2.imshow('Helmet Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
