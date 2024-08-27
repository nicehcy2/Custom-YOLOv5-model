import os
import sys
from pathlib import Path

import cv2
import torch

from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.torch_utils import select_device

from models.common import DetectMultiBackend
from utils.dataloaders import VID_FORMATS, LoadStreams
from utils.plots import plot_one_box

# 경로 설정
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 루트 디렉터리
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # PATH에 ROOT 추가
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 상대 경로

from models.common import DetectMultiBackend
from utils.dataloaders import VID_FORMATS, LoadStreams
from ultralytics.utils.plotting import colors, save_one_box

def run(
    weights=ROOT / "best.pt",         # 모델 경로
    source=ROOT / "pothole.mp4",             # 입력 소스
    imgsz=(640, 640),                    # 이미지 사이즈
    conf_thres=0.25,                     # 신뢰 임계값
    iou_thres=0.45,                      # NMS IOU 임계값
    max_det=1000,                        # 이미지 당 최대 검출 수
    device="",                           # CUDA 디바이스
    save_img=True,                       # 결과 이미지 저장 여부
    save_crop=False,                     # 검출된 객체 잘라내어 저장 여부
    save_dir=ROOT / "inference_output",  # 저장할 디렉터리
):
    # 입력 소스를 문자열로 변환
    source = str(source)
    is_file = Path(source).suffix[1:] in VID_FORMATS  # 입력 소스가 비디오 파일인지 확인
    dataset = LoadStreams(source, img_size=imgsz) if not is_file else None  # 데이터셋 로드 (스트림 또는 비디오)
    save_dir.mkdir(parents=True, exist_ok=True)  # 결과 저장 디렉터리 생성

    # 장치 선택
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride = model.stride

    # 추론 시작
    model.warmup(imgsz=(1, 3, *imgsz))  # 모델 웜업
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 객체 검출
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # 결과 처리
        for i, det in enumerate(pred):
            p, im0, frame = path, im0s.copy(), dataset.count

            # 객체 검출된 경우
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

                # 결과를 이미지에 추가
                for *xyxy, conf, cls in det:
                    label = f"{model.names[int(cls)]} {conf:.2f}"
                    plot_one_box(xyxy, im0, label=label, color=colors(int(cls), True), line_thickness=3)

                    # 객체 잘라내어 저장
                    if save_crop:
                        save_one_box(xyxy, im0, file=f"{save_dir}/crop_{frame}.jpg", BGR=True)

            # 이미지 저장
            if save_img:
                cv2.imwrite(f"{save_dir}/result_{frame}.jpg", im0)

if __name__ == "__main__":
    run()