# Bàn giao IMPLEMENTER — Step 3 (Baseline YOLO: B01 YOLO11n, B02 YOLO11s)

## Phase 0

### Căn cứ và mục tiêu

- Làm việc tại worktree `D:/FPTU/KLTN/PCB_Lab_impl3`, branch `step3/impl` (base `2a1545e`). WORKFLOW.md quy định mỗi agent một branch/worktree; worktree này đã được người điều phối tạo sẵn cho `step3/impl`.
- Đã đọc `docs/handoff/WORKFLOW.md` (quy tắc chung), `docs/handoff/step3-contract.md` (hợp đồng Bước 3), `PLAN.md` §1.2–1.3, §3.1, §5 Bước 3, §6, §7.3, §8, §11; `docs/handoff/step2-contract.md`, `step2-impl.md`; code Step 1–2 (`data/samples.py`, `inference/preprocessing.py`, `data/manifest.py`). Không dùng docs/archive.
- **Không** nạp checkpoint cũ, không chạy runner cũ, không đụng branch khác.
- Mục tiêu Bước 3 (PLAN §5): train baseline YOLO11n (B01) + YOLO11s đối chứng (B02) trên 1.792 ảnh train, 6 lớp; trích dự đoán calibration/fusion để Bước 5 hiệu chỉnh ngưỡng. Đầu ra: artifact, model card, báo cáo validation, `run_manifest.json`.
- **Hoàn thành:** checkpoint nạp lại được (có `allow_smoke=True`), class order đúng (0=open_circuit…5=pin_hole), không dùng test chọn epoch/ngưỡng, artifact đầy đủ B01+B02.

### Metadata thật đã xác minh (môi trường đã cài)

Môi trường phát triển CPU local tại worktree, cài mới vào `.venv` riêng (Python 3.14.7, Windows x64, **CUDA unavailable**). Khóa phiên bản theo bản đã cài:

- **torch 2.14.0+cpu**, **torchvision 0.29.0+cpu**, **ultralytics 8.4.161**.
- Step 2: NumPy 2.4.3, Pillow 12.1.1, PyYAML 6.0.3, setuptools 84.0.0.
- Dataset `D:\FPTU\KLTN\DatasetVer4_Public` **tồn tại** (xác nhận bằng ls). Trong process tool hiện tại biến `DATASET_ROOT` chưa có giá trị; adapter vẫn phân giải đúng từ `configs/dataset.yaml`.
- **Quan trọng — network/telemetry:** import `ultralytics` tự tạo file user-level `C:\Users\ASUS\AppData\Roaming\Ultralytics\settings.json` (ngoài DATASET_ROOT, ngoài repo). Không ghi vào dataset. Đã ghi chú đoạn sau để vô hiệu hóa telemetry (`ULTRALYTICS_NO_ANALYTICS=1`) trong train/extract.

### Đọc mã ultralytics 8.4.161 — xác nhận default thực và điểm khác hợp đồng

Đọc trực tiếp `ultralytics/cfg/default.yaml` và `engine/predictor.py`, `data/dataset.py`, `nn/tasks.py` của bản đã cài (không dùng trí nhớ). Bảng so sánh với hợp đồng:

| Tham số | Default thực (8.4.161) | Hợp đồng step3 | Quyết định |
|---|---|---|---|
| `epochs` | 100 | 100 | dùng 100 |
| `patience` | **100** | **20** | override = 20 (ghi args_used) |
| `batch` | 16 | null → duyệt 16 | dùng 16 (user duyệt) |
| `imgsz` | 640 | 640 | dùng 640 |
| `optimizer` | `auto` | **AdamW** | override = AdamW |
| `lr0` | 0.01, comment "SGD=1e-2, Adam/AdamW=1e-3" | null → duyệt 0.001 | dùng **0.001** (đúng default AdamW thực) |
| `deterministic` | True | True | dùng True |
| `seed` | **0** | CLI mặc định 42 | truyền `seed=42` qua CLI/runner |
| `workers` | 8 | null → phải đặt | đặt (CPU local=0/2; cloud GPU theo VRAM) |
| `cache` | False | False | False (tránh ghi cache) |
| `plots` | True | (muốn lưu plots) | True |
| `conf` | predict 0.25 / val 0.001 | `infer.conf_floor`=0.001 | val/infer dùng 0.001 |
| `iou` (NMS) | 0.7 | 0.7 | dùng 0.7 |
| `max_det` | 300 | 300 | dùng 300 |
| `agnostic_nms` | False | False | False |
| `amp` | True | (không nói) | CPU: tắt (False) để tránh warning/path sinh artifact; ghi args_used. Cloud GPU: bật. |
| `warmup_epochs` | 3.0 | — | giữ mặc định |
| `cos_lr` | False | — | giữ mặc định |
| `close_mosaic` | 10 | — | giữ mặc định (ảnh PCB không mosaic-friendly nhưng đây là mặc định; ablation sau) |
| augmentation photometric | `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, mosaic=1.0, scale=0.5` | `augment=null` → duyệt sau | **Tất cả photometric/mosaic tắt (0.0)** cho recipe `yolo_aug_v1`; chỉ giữ hình học thuần của Step 2 (xoay 90°×k + flip). Lý do: giữ nguyên lỗi nhỏ ở 640px. |

**Các điểm khác hợp đồng / cần lưu ý (ghi vào handoff theo yêu cầu Phase 0.3):**

1. **Kênh màu (rất quan trọng):** `engine/predictor.py:165–179` — khi truyền **mảng numpy** vào `model.predict()`, ultralytics kỳ vọng **BGR uint8** (docstring rõ: "[(H,W,3)xN] for list of BGR uint8 arrays"), và dòng 179 `im = im.flip(1)  # BGR to RGB` đảo kênh. Còn khi truyền **đường dẫn ảnh**, ultralytics dùng `cv2.imread` → cũng BGR. Nghĩa là: cả hai đường đều vào BGR rồi đảo thành RGB nội bộ. Adapter của Step 3 làm việc trên **RGB canonical** (theo Step 2) → phải chuyển `RGB → BGR` (`arr[..., ::-1]`) trước khi đưa mảng cho ultralytics, để kết quả khớp với đường truyền path. Tôi sẽ viết kiểm chứng tại chỗ chứng minh `predict(RGB_array_bgr_converted)` ≈ `predict(image_path)` (cùng box, cùng conf) — chi tiết Phase 1.
2. **Ghi cache nhãn:** `data/dataset.py:109` `cache_labels(path = Path("./labels.cache"))` — đường dẫn **tương đối với thư mục chạy (CWD)**, không phải DATASET_ROOT. Để "không cho ultralytics ghi bất cứ thứ gì vào DATASET_ROOT", runner sẽ (a) đặt `cache=False`, và (b) chdir vào thư mục `runs/...` được kiểm soát trước khi gọi train, đồng thời guard kiểm tra không có file `.cache`/`labels.cache` nào xuất hiện dưới DATASET_ROOT.
3. **Truy cập mạng:** `nn/tasks.py:1827,1891` `attempt_download_asset(weight)` — chỉ kích hoạt khi file trọng số **không tồn tại local**, tải từ `github.com/ultralytics/assets/releases`. Hợp đồng yêu cầu pretrained phải có sẵn (`artifacts/pretrained/*.pt`) và chỉ tải khi `allow_download=True`. Runner sẽ từ chối (`ConfigError`) nếu pretrained thiếu và `allow_download=False`. Khi `allow_download=True` sẽ ghi nguồn+phiên bản.
4. **Kiểm tra AMP:** không tải mạng; chỉ chạy một forward nhỏ trên device. Trên CPU để `amp=False` nên bước này bỏ qua/nhanh. Ghi rõ method đo peak memory.
5. **Font/telemetry:** `utils/__init__.py:1008, SETTINGS_FILE` — import tạo settings.json user-level (đã nêu). Không tải font từ mạng trong flow detect thông thường. Thiết lập `ULTRALYTICS_NO_ANALYTICS=1`.

### Quyết định ảnh hưởng phạm vi (3 đề xuất đã duyệt)

**(a) Recipe `yolo_aug_v1`** — hình học thuần, tái dùng recipe `aug_v1` đã duyệt của Step 2 (xoay 0/90/180/270° + lật ngang p=0.5, RNG cục bộ theo seed, **không** photometric/scale/resample). Lý do giữ lỗi nhỏ: theo `reports/data_profile.md` §"Kích thước box", lỗi nhỏ nhất pin_hole short_side min 20px (P5/50/95 = 24/30/42), mouse_bite min 20px, spur min 21px — ở 640px, biến đổi hình học giữ nguyên pixel → **không làm mất/đổi bản chất lỗi nhỏ**. Mọi photometric (hsv/brightness/contrast), scale, mosaic đẩy sang ablation có tên (Bước 9). Hợp đồng truyền augmentation qua tham số ultralytics `"yolo_aug_v1"` → tôi sẽ set toàn bộ hyp photometric = 0.0 và bật augmentation pipeline hình học riêng (dùng `apply_augmentation` Step 2 trên mỗi ảnh trước letterbox, qua `augment.py` của dataset loader hoặc augment custom). Quyết định cụ thể cơ chế áp augmentation vào YOLO DataLoader ghi rõ ở Phase 1.

**(b) lr0 & batch cho AdamW** — `lr0=0.001`, `batch=16`. Đã duyệt. Khớp default thực AdamW của bản 8.4.161 (`lr0` comment: "Adam/AdamW=1e-3"). `batch` tinh chỉnh theo VRAM trên cloud; nếu B02 (yolo11s) hết VRAM sẽ ghi lý do (hợp đồng cho phép khác batch chỉ khi hết VRAM).

**(c) Tham số `infer{}`** — khoá: `conf_floor=0.001` (lọc cực thấp, không rụng box trước bước chọn `tau_yolo`, đúng PLAN §5 Bước 3), `iou(NMS)=0.7`, `max_det=300`, `imgsz=640`, `agnostic_nms=False`, `half=False`. Giữ nguyên hợp đồng.

### Giả định nhỏ

- Không dùng test để chọn epoch/ngưỡng; checkpoint chọn bằng metric validation trên calibration (`ultralytics_fitness` = weighted combination mAP50/mAP50-95/...) như hợp đồng `selection.metric`.
- `classes_ref` = `benchmarks/deeppcb/configs/classes.json` (tên 0..5). `data.yaml` chỉ có `train`/`val`, không có `test`. names theo thứ tự classes.json.
- `seed` mặc định 42 qua CLI; hợp đồng config không chứa seed (config_hash không gồm seed).
- ARTIFACT: smoke không tạo artifact chính thức (chỉ `runs/smoke/...`); `from_artifact` mặc định `allow_smoke=False` sẽ từ chối artifact smoke.
- Peak memory: CPU RAM đo bằng `tracemalloc`/RSS trước-sau (method ghi rõ); VRAM = null trên CPU (method = "torch.cuda.* không khả dụng trên CPU; ghi null").

### Chỗ mơ hồ / câu hỏi

- **Không có mơ hộng chặn triển khai.** Hợp đồng đã chốt rõ phạm vi, schema, bất biến. Các giá trị null (lr0/batch/workers/augment) đã được duyệt ở trên.
- Một điểm nhỏ chưa chốt bằng văn bản: cơ chế áp augmentation hình học vào YOLO DataLoader (custom `Albumentations`-style transform vs áp trước letterbox). Tôi sẽ dùng transform tùy chỉnh đọc từ `apply_augmentation` Step 2, áp trên ảnh gốc rồi letterbox — không đổi training code ultralytics. Ghi rõ ở Phase 1, không cần dừng hỏi.

### File dự định tạo/sửa (Phase 1)

- Tạo `src/pcb_lab/models/__init__.py`, `src/pcb_lab/models/yolo/{__init__,view,train,adapter,extract}.py`.
- Tạo `configs/models/yolo_common.yaml`, `configs/models/yolo11n.yaml`, `configs/models/yolo11s.yaml`.
- Tạo `scripts/train_yolo.py`, `scripts/extract_yolo_predictions.py`.
- Tạo `requirements/step3.txt` (torch cpu / ultralytics / step2 deps, khóa phiên bản).
- Tạo `docs/RUN_GPU.md` (lệnh CUDA local + Colab/Kaggle: đặt DATASET_ROOT, cài requirements, tải pretrained, mang artifacts về local).
- Cập nhật chính handoff này (Phase 1/2); ghi `docs/handoff/questions-impl.md` nếu phát sinh mâu thuẫn.
- Không sửa tests/, PLAN/DECISIONS/TRIAGE/WORKFLOW/contractor; không commit dataset/checkpoint/preds/ảnh. Mỗi commit kiểm `git status`.

### Trạng thái duyệt

- Đề xuất (a) recipe hình học, (b) lr0=0.001/batch=16, (c) infer{} — **đã được người dùng duyệt** qua AskUserQuestion.
- **DỪNG chờ người dùng duyệt trước khi chạy train thật trên GPU.** Phase 0 chỉ thiết lập môi trường + xác nhận default + ghi handoff này.
