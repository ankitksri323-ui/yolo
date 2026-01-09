import argparse
import sys
import platform
from pathlib import Path

import cv2
import numpy as np
import torch

if platform.system() == "Windows":
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath

# Paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

#  YOLOv5 Imports 
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Argument Parser 
def parse_args():
    parser = argparse.ArgumentParser(description="Fire Detection using YOLOv5")
    parser.add_argument("--weights", type=str, required=True, help="Path to fire model weights (.pt)")
    parser.add_argument("--source", type=str, default="0", help="Video path or webcam index")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    return parser.parse_args()

# main
def main():
    args = parse_args()

    device = select_device("0" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(args.weights, device=device)
    stride, names = model.stride, model.names
    imgsz = (args.imgsz, args.imgsz)

    model.warmup(imgsz=(1, 3, *imgsz))

    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(" Error opening video source")
        return

    if platform.system() == "Windows":
        import winsound

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, imgsz)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)
        pred = non_max_suppression(pred, args.conf, args.iou)

        fire_detected = False

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    cls_name = names[int(cls)]
                    if "fire" in cls_name.lower():
                        fire_detected = True

                    x1, y1, x2, y2 = map(int, xyxy)
                    label = f"{cls_name} {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if fire_detected:
            cv2.putText(frame, "🔥 FIRE DETECTED!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            if platform.system() == "Windows":
                winsound.Beep(1500, 400)

        cv2.imshow("Fire Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
