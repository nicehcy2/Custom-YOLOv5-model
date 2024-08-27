import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys
import os
import torch
import cv2
import numpy as np

# YOLOv5 디렉토리를 PYTHONPATH에 추가

yolov5_path = os.path.join(os.getcwd(), 'yolov5')
sys.path.append(yolov5_path)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# CUDA 사용 가능 여부 확인
print("CUDA Available: ", torch.cuda.is_available())

# YOLOv5 모델 로드 (로컬 파일 사용)
model_path = os.path.join(os.getcwd(), 'best.pt')
device = select_device('')
model = attempt_load(model_path)
model.eval()

# 클래스 이름 로드
class_names = model.module.names if hasattr(model, 'module') else model.names

# 웹캠 설정
cap = cv2.VideoCapture(0)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def detect_objects():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 전처리
        img, ratio, dwdh = letterbox(frame, new_shape=640)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # YOLOv5 모델로 객체 검출
        with torch.no_grad():
            pred = model(img, augment=False)[0]

        pred = non_max_suppression(pred, 0.4, 0.5)

        # 결과를 이미지에 표시
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    bgr = (0, 255, 0)  # 상자 색상 (초록색)
                    class_name = class_names[int(cls)]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                    cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        cv2.imshow('YOLOv5 Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_objects()
