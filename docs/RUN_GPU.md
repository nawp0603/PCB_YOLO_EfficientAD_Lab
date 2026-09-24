# Chạy thật Bước 3 (Baseline YOLO) trên GPU

Bước 3 huấn luyện YOLO11n (B01) và YOLO11s (B02) trên GPU. Môi trường local (worktree
`PCB_Lab_impl3`) chỉ là **CPU dev** để phát triển/smoke; huấn luyện nặng 100 epoch chạy
trên **GPU đám mây** (theo PLAN §5 Bước 0 / Gate 0). Tài liệu này ghi đúng lệnh cho hai
trường hợp: (a) CUDA local, (b) Colab/Kaggle.

> Quy tắc: dataset chỉ đọc. Ultralytics **chỉ** thấy view do `build_yolo_view` dựng
> (hard link + label + `data.yaml`); không bao giờ ghi vào `DATASET_ROOT`. Artifact và
> prediction được đưa về máy local để nghiệm thu.

## 0. Chuẩn bị dataset (cả hai môi trường)

Dataset PCB nằm ở `D:\FPTU\KLTN\DatasetVer4_Public` (Windows local) hoặc được tải lên
thư mục làm việc trên cloud. Đặt biến `DATASET_ROOT` trỏ vào thư mục gốc chứa
`benchmarks/deeppcb/...`.

- Windows local: `$env:DATASET_ROOT = 'D:\FPTU\KLTN\DatasetVer4_Public'`
- Linux/Colab: `export DATASET_ROOT=/path/to/DatasetVer4_Public`
- Hoặc truyền `--dataset-root` khi chạy script.

## 1a. CUDA local (Windows/Linux, có GPU)

```powershell
# Tạo venv Python 3.14.7 và cài đúng requirements (CUDA wheels)
uv --cache-dir .cache/uv venv --python 3.14.7 .venv
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe `
    --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple `
    -r requirements/step3.txt
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe --no-build-isolation --no-deps -e .
```
> Với torch CUDA: đổi `--index-url` thành bản cu128 (hoặc cu126 tuỳ driver) thay vì
> `.../cpu`. Các gói còn lại giữ nguyên `requirements/step3.txt` (khóa phiên bản).

Tải pretrained YOLO11 (YOLO11n/YOLO11s) từ Ultralytics về `artifacts/pretrained/`:

```powershell
New-Item -ItemType Directory -Force artifacts/pretrained | Out-Null
# YOLO11n
Invoke-WebRequest -Uri https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt `
    -OutFile artifacts/pretrained/yolo11n.pt
Invoke-WebRequest -Uri https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt `
    -OutFile artifacts/pretrained/yolo11s.pt
```
> Ghi nguồn + phiên bản vào `run_manifest.pretrained.source`. Hash được tự động tính và
> ghi `pretrained.sha256`. Nếu không tải thủ công, runner từ chối chạy (phải dùng
> `--allow-download` và ghi rõ nguồn).

Huấn luyện (device 0 = GPU đầu tiên):

```powershell
$env:DATASET_ROOT = 'D:\FPTU\KLTN\DatasetVer4_Public'
python scripts/train_yolo.py --config configs/models/yolo11n.yaml --seed 42 --device 0
python scripts/train_yolo.py --config configs/models/yolo11s.yaml --seed 42 --device 0
python scripts/extract_yolo_predictions.py --artifact artifacts/yolo/B01_yolo11n/seed42
python scripts/extract_yolo_predictions.py --artifact artifacts/yolo/B02_yolo11s/seed42
```

## 1b. Colab / Kaggle (đám mây)

Tải repo lên hoặc `git clone` branch `step3/impl`, rồi:

```python
import os, subprocess, sys
# 1) Dataset: mount Drive hoặc upload thư mục DatasetVer4_Public, đặt DATASET_ROOT
os.environ["DATASET_ROOT"] = "/content/DatasetVer4_Public"
# 2) Cài requirements (CUDA đã có sẵn trên Colab; Kaggle có P100/V100)
subprocess.run(["pip", "install", "-r", "requirements/step3.txt"], check=True)
# 3) Tải pretrained
!mkdir -p artifacts/pretrained
!wget -O artifacts/pretrained/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
!wget -O artifacts/pretrained/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
```

Huấn luyện (dùng device mặc định = GPU 0 trên Colab/Kaggle):

```python
!python scripts/train_yolo.py --config configs/models/yolo11n.yaml --seed 42 --device 0
!python scripts/train_yolo.py --config configs/models/yolo11s.yaml --seed 42 --device 0
!python scripts/extract_yolo_predictions.py --artifact artifacts/yolo/B01_yolo11n/seed42
!python scripts/extract_yolo_predictions.py --artifact artifacts/yolo/B02_yolo11s/seed42
```

Trên GPU, runner tự bật AMP (đổi `amp=False` trong `train.py` overrides thành `True`
cho cloud, hoặc truyền qua biến môi trường `PCB_AMP=1`) và ghi `peak_vram_mb`.

## 2. Mang artifacts/preds về máy local để nghiệm thu

Chỉ mang về những file sau (không mang `runs/` hay dataset):

- `artifacts/yolo/B01_yolo11n/seed42/{best.pt,last.pt,artifact.json,model_card.md,
  calibration_per_class.json,preds_calibration.jsonl,preds_fusion.jsonl}`
- Tương tự cho `B02_yolo11s/seed42`.

Trên Colab: `from google.colab import files; files.download('artifacts/yolo/B01_yolo11n/seed42/best.pt')`
Hoặc nén rồi tải: `!zip -r yolo_artifacts.zip artifacts/`.

Đưa về máy local vào đúng đường dẫn `artifacts/yolo/...` trong worktree `PCB_Lab_impl3`
để VERIFIER nghiệm thu (adapter load lại, kiểm class order + hash, extract validation).

## 3. Tham số đã khoá (đọc từ configs/models/*.yaml)

- `imgsz=640`, `epochs=100`, `patience=20`, `optimizer=AdamW`, `lr0=0.001`, `batch=16`.
- Augmentation `yolo_aug_v1`: hình học thuần (xoay 90°×k + lật ngang), **tắt mọi**
  photometric/mosaic (ghi đè hsv/scale/mosaic=0 trong runner).
- `infer`: `conf_floor=0.001`, `iou(NMS)=0.7`, `max_det=300`, `agnostic_nms=False`, `half=False`.
- Selection: validation trên `calibration`; không dùng test chọn epoch/ngưỡng.

## 4. Lưu ý

- `workers` mặc định 8 trên GPU; giảm nếu thiếu RAM. Nếu B02 (yolo11s) hết VRAM, giảm
  `batch` và ghi lý do vào handoff (hợp đồng cho phép khác batch chỉ khi hết VRAM).
- Không commit checkpoint/preds/ảnh. `.gitignore` đã loại `artifacts/**/*.pt`,
  `artifacts/**/preds_*.jsonl`, `runs/`, `data_refs/yolo_view*`.
- Telemetry Ultralytics bị tắt bằng `ULTRALYTICS_NO_ANALYTICS=1` trong runner.
