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

---

## Phase 1

### File đã tạo (đã commit sớm, xem git log)

- `src/pcb_lab/models/__init__.py`, `src/pcb_lab/models/yolo/{__init__,view,train,adapter,extract}.py`.
- `configs/models/yolo_common.yaml` (chỉ comment, parse thành mapping rỗng), `yolo11n.yaml`, `yolo11s.yaml`.
- `scripts/train_yolo.py`, `scripts/extract_yolo_predictions.py`.
- `requirements/step3.txt` (torch 2.14.0+cpu, ultralytics 8.4.161, các dep Step 2 khóa phiên bản, kèm lệnh cài `--index-url https://download.pytorch.org/whl/cpu`).
- `docs/RUN_GPU.md` (CUDA local + Colab/Kaggle: đặt DATASET_ROOT, cài requirements, tải pretrained, mang artifact về local).

### Cơ chế đã hiện thực (theo hợp đồng step3-contract.md)

**view.py — `build_yolo_view`**
- Chỉ chấp nhận `train` + `calibration`; `fusion`/`test` → `ViewError(PermissionError)` (`_reject_forbidden`).
- Ghi hardlink ảnh (fallback copy nếu cross-device) + label YOLO `class cx cy w h` (6 decimals) + `data.yaml` **chỉ có `train`/`val`/`names`, không có `test`** (assert cứng).
- Không bao giờ ghi vào DATASET_ROOT: dùng `output_path()` guard + `cache=False` trong runner; `view_manifest.txt` + `view_signature` (sha256 trên `sample_id\timage_sha256\tlabel_sha256` đã sort) để rebuild idempotent.
- `limit={'good':n,'defect':n}` tuỳ chọn; nếu `None` giữ nguyên toàn bộ (train 1792 / calibration 460).

**train.py — `train_yolo`**
- Merge `yolo_common.yaml` + model config; từ chối key không thuộc `ALLOWED_CONFIG_KEYS` (`ConfigError`).
- `_check_nulls` từ chối `lr0/batch/workers/augment` null → runner không chạy với hyperparam thiếu.
- Override ép **toàn bộ** photometric/mosaic = 0.0 (`hsv_*`, `degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`, `fliplr=0.5`, `mosaic`, `mixup`, `copy_paste`, `bgr`, `close_mosaic`); `amp=False` (CPU), `cos_lr`/`warmup_epochs` từ config.
- Recipe `yolo_aug_v1` = hình học thuần, áp **online** qua `_AugmentedYOLODataset.get_image_and_label`: xoay 90°×k (seed per `(aug_seed,epoch,index)`) + lật ngang p=0.5, áp lên cả ảnh BGR và box xywh chuẩn hóa (`_geometric_augment`). **View không bị mở rộng** → giữ invariant #1 (1792/460).
- Quản lý pretrained: `ConfigError` nếu thiếu và `allow_download=False`; nếu `allow_download=True` ghi source. Smoke dùng `arch` + `limit=8/8`, `epochs=1`, không tạo artifact.
- `chdir` vào `runs/...` trước khi gọi train (ngăn ghi `labels.cache` vào DATASET_ROOT). Đo peak RAM (psutil / win32 psapi), VRAM = null trên CPU (`method` ghi rõ, không fake).
- Xuất `run_manifest.json`, `env.json`, copy `best/last.pt` → `artifacts/yolo/<model_id>/seed<seed>`, `calibration_per_class.json`, `artifact.json` (có sha checkpoint + class_order + config_hash + view_signature), `model_card.md` (tiếng Việt, ASCII-safe), `_link_preds`.

**adapter.py — `YoloDetector` / `UltralyticsEngine`**
- `UltralyticsEngine.infer` nhận **RGB uint8 HWC** (canonical Step 2), chuyển `bgr = arr[..., ::-1]` rồi `model.predict(...)` — khớp đường truyền path (cv2.imread=BGR). Trả `[N,6]` xyxy/conf/cls trong không gian letterboxed 640.
- `YoloDetector.predict` dùng **duy nhất** `prepare_input(image,"yolo")` + `LetterboxMeta` + `unletterbox_boxes` (tái dùng Step 2) → trả list `Detection` đã sort theo confidence giảm dần; `image_score` = max conf hoặc 0.0.
- `from_artifact` kiểm: `artifact.json` + `best.pt` tồn tại, sha checkpoint khớp (`ArtifactMismatchError`), class order khớp `classes.json` (`ClassOrderError`), artifact smoke bị từ chối trừ khi `allow_smoke=True` (`SmokeArtifactError`).

**extract.py — `extract_predictions`**
- Chỉ `calibration`/`fusion`; `test`/khác → `PermissionError`.
- Đọc sample qua `ManifestDataset`, chạy adapter, ghi `preds_<partition>.jsonl` (1 dòng/ảnh, sort theo `sample_id`, deterministic, ghi sha256 file).
- Lỗi inference ghi vào `error_reason`, **không** drop/đổi thành GOOD/0; confidence ≥ `conf_floor`.

### Kiểm chứng tĩnh / mức module (đã chạy, PASS — chưa cần checkpoint/train)

1. **Compile:** `py_compile` toàn bộ `src/pcb_lab/models/yolo/*.py` + `scripts/*.py` → OK.
2. **Import symbols:** `import pcb_lab.models.yolo` + `_PCBDetectionTrainer`, `_AugmentedYOLODataset`, `_geometric_augment`, `_merge_configs`, `_config_hash`, `_check_nulls`, `ViewError`, `ConfigError`, `YoloDetector`, `extract_predictions` → OK (sửa lỗi thiếu `from dataclasses import dataclass, field` và `from ultralytics import YOLO`).
3. **`_merge_configs`** trên `yolo_common.yaml`(comment-only) + `yolo11n.yaml` → merge đúng, keys `{architecture,classes_ref,infer,model_id,pretrained,recipe_id,selection,train}`; `_check_nulls` PASS trên config hợp lệ, raise `ConfigError` khi `lr0=None`.
4. **`_geometric_augment`** (k=1,flip=True) trên ảnh 4×4 + box → ảnh xoay/lật, box `[[0.25,0.25,0.5,0.5]]` → `[[0.75,3.75,0.5,0.5]]` (đúng công thức flip ngang: cx'=1-cx; xoay 90° cw: (cx,cy)→(1-cy,cx), w↔h).
5. **Build view thật trên DATASET_ROOT** (`D:\FPTU\KLTN\DatasetVer4_Public`): `train=1792 (good 895/defect 897)`, `calibration=460 (good 230/defect 230)`; box_counts mỗi lớp hợp lý (open_circuit, short, mouse_bite, spur, spurious_copper, pin_hole); `data.yaml` **không có `test`**; `view_signature` sinh đúng. `limit={'good':2,'defect':2}` → 4/4 ảnh.
6. **Gating partition:** view từ chối `fusion`/`test`; extract từ chối `test`/`train`/`fusion+test` → `PermissionError`.
7. **DATASET_ROOT untouched:** sau build view, `ls` dataset chỉ ra các thư mục gốc (audit/benchmarks/...); không có file cache mới.
8. **Adapter wiring (không cần ckpt):** `prepare_input(path,"yolo")` trên ảnh defect thật trả meta có `scale/pad_left/pad_top/orig_w/orig_h` + `LetterboxMeta(...)` + `unletterbox_boxes` roundtrip → OK.

### Ghi chú kỹ thuật

- `resolve_classes_root`/`load_canonical_names` trong adapter phân giải dataset root từ `configs/dataset.yaml`/env, không hardcode.
- `view.data_yaml` gán `val: images/calibration` (không phải `test`) — khớp hợp đồng `selection.split=calibration`.
- `config_hash` loại trừ `seed` (CLI-only) nên tái lập được từ config.

---

## Phase 2 — Smoke CPU & Tự kiểm chứng

### Sửa lỗi thực thi (phát hiện khi chạy thật trên ultralytics 8.4.161)

1. **Smoke phải random init, không download:** runner gốc truyền `arch` ("yolo11n") làm model string → ultralytics nối `.pt` rồi tải trọng số. Đã sửa: smoke dùng `"<arch>.yaml"` (random init, không mạng). Real run vẫn dùng pretrained `.pt` như hợp đồng.
2. **Run dir path:** `project=runs_base.parent.parent` chỉ lên `runs/yolo`, làm `best.pt` không nằm ở `runs_base` → sửa thành `runs_base.parent` (`runs/yolo/<model_id>/seed<seed>`).
3. **`fraction` (dataloader):** truyền `None` gây `int * NoneType`. Đã sửa thành `1.0`.
4. **API label đổi:** ultralytics 8.4.161 trả `label["instances"]` (đối tượng `Instances`, box normalized xywh trong `_bboxes.bboxes`), **không phải** `label["bboxes"]`. Đã viết lại `_geometric_augment` (hoạt động trong không gian normalized xywh — khớp chính xác recipe `aug_v1` Step 2: xoay 90° ccw `(cx,cy,bw,bh)->(cy,1-cx,bh,bw)`, flip `cx->1-cx`) và `get_image_and_label` ghi qua `instances.update(...)` với `float32`.
5. **Adapter class order:** `model.model.names` trả dict `{idx: name}`, `list(...)` ra key số. Đã chuẩn hóa (dict → list theo index) trước so sánh với `classes.json`.

Tất cả sửa trên đã commit (`a06bc9a`).

### Kết quả chạy (PASS)

**A. Smoke train CPU 1 epoch (real data, limit 8/8 good/defect)**
- `runs/smoke/B01_yolo11n/seed42/` sinh đầy đủ: `run_manifest.json`, `env.json`, `weights/best.pt`(5.47 MB), `weights/last.pt`, plots (labels/results/confusion/curves), `train_batch0.jpg`, `val_batch0_pred.jpg`.
- `run_manifest.json`: `model="yolo11n.yaml"`, `epochs=1`, `optimizer=AdamW`, `lr0=0.001`, `patience=20`, photometric/mosaic = 0.0 (ghi `args_used`), `peak_ram_mb=770.0` (psutil), `peak_vram_mb=null` (CPU, method ghi rõ, không fake), `link_mode=hardlink`, `view_signature=ce24904dbd92a114`, `device=cpu`, `torch=2.14.0+cpu`, `ultralytics=8.4.161`, `dataset_release_sha256` đầy đủ.
- **`artifact_dir=null`** — đúng, smoke không tạo artifact chính thức (hợp đồng). Validation trên calibration ghi `overall` + `per_class` 6 lớp.

**B. Adapter reload + equivalence (chứng minh kênh màu)**
- Tạo smoke artifact dir (copy `best.pt` + viết `artifact.json` smoke, `smoke:true`). `YoloDetector.from_artifact(..., allow_smoke=True)` nạp OK.
- **EQUIVALENCE `predict(RGB_array)` == `predict(image_path)`:** test 12 ảnh calibration, **0 mismatches** (cùng số box, class, confidence, tọa độ). → chứng minh `RGB->BGR` trong `UltralyticsEngine` khớp đường truyền path (cv2.imread BGR).

**C. Extract predictions (calibration, full partition)**
- `extract_predictions(artifact, ..., partitions=("calibration",))` → `preds_calibration.jsonl` 460 dòng, 32.200 detections, deterministic (sha ghi rõ).
- Sanity: confidence ∈ [0,1], xyxy hợp lệ (x1≤x2, y1≤y2, trong bounds) — **0 bad rows**.
- Gating `test` → `PermissionError` (đã verify Phase 1).

**D. Full view + invariants (vs PLAN §1.3)**
- `train=1792` (good 895 / defect 897), `calibration=460` (good 230 / defect 230) — đúng.
- Per-class train box counts **khớp chính xác** PLAN §1.3: open_circuit 1216, short 973, mouse_bite 1228, spur 981, spurious_copper 884, pin_hole 865.
- `data.yaml` chỉ có `path/train/val/names`, **không có `test`**.
- `fusion`/`test` bị `ViewError(PermissionError)` từ chối.
- `view_signature` tái lập được (rebuild → cùng `ce24904dbd92a114`).
- `link_mode=hardlink`.

### Điều kiện/ngoại lệ đã ghi nhận

- Smoke chạy trên CPU (local dev) — không phải GPU đám mây. Hợp đồng Phase 2 chỉ yêu cầu smoke self-verify trên CPU; **train thật 100 epoch B01/B02 chạy trên GPU đám mây (Gate 0)** theo `docs/RUN_GPU.md`.
- `peak_vram_mb=null` trên CPU là đúng (không có CUDA) — ghi method rõ, không fake.
- Predict có warning `'half' is deprecated` (ultralytics 8.4.161) — không ảnh hưởng (`half=False` trong infer{}). Sẽ theo dõi khi chạy cloud.

### Trạng thái duyệt

- **Phase 2 PASS** trên CPU: smoke train OK, adapter equivalence OK, extract OK, full-view invariants OK. Không merge, chờ điều phối viên review rồi chuyển sang train GPU thật (Bước 0/Gate 0).

---

## Gói thực thi Colab (Gate 0 & Full Training 100 epochs)

### Thành phần đã tạo

- `notebooks/train_yolo_colab.ipynb` — notebook "run-all" (Runtime ▸ Run all) 8 cell:
  1. **Gate 0 Hardware**: `!nvidia-smi` + `torch.cuda.is_available()`, in tên GPU + CUDA capability, assert compute capability >= 7.0.
  2. **Dataset Setup**: mount Drive, gán `os.environ["DATASET_ROOT"]`, assert manifest `samples.jsonl` tồn tại.
  3. **Install**: cài từ `requirements/step3.txt` **nhưng bỏ qua torch/torchvision** (giữ CUDA build của Colab), rồi `pip install --no-build-isolation --no-deps -e .`.
  4. **Pretrained**: tạo `artifacts/pretrained/`, tải `yolo11n.pt` + `yolo11s.pt` từ Ultralytics release `v8.3.0`.
  5. **Train B01 YOLO11n**: 100 epoch, seed 42, AdamW, `--device 0`, AMP auto-on GPU.
  6. **Train B02 YOLO11s**: tương tự.
  7. **Extract**: `calibration` + `fusion` cho cả 2 model (tập `test` bị chặn trong code).
  8. **Export**: nén `artifacts/yolo/` -> `yolo_step3_artifacts.zip`, `files.download()`.
- `scripts/package_colab.py` — đóng gói mã nguồn thành `exports/colab_bundle.zip` (61 files, ~177 KB), loại `.venv/ runs/ .cache/ artifacts/ *.pt *.zip` và dataset.

### AMP (thay đổi interface)

- `train_yolo` và CLI `--amp` mới: **mặc định auto** bật trên CUDA GPU, tắt trên CPU; có thể ép bằng `--amp true|false`. Trước đây hardcode `amp=False`. Cloud GPU (Cell 5/6) sẽ ghi `amp=True` vào `run_manifest.json`.

### Cách đưa lên Colab

```powershell
# 1) Dong goi (local)
python scripts/package_colab.py                 # -> exports/colab_bundle.zip
# 2) Tai colab_bundle.zip len Google Drive hoac GitHub
# 3) Mo Colab, upload giai nen, chay notebook/train_yolo_colab.ipynb (Runtime -> Run all)
# 4) O Cell 2 sua DATASET_ROOT neu DatasetVer4_Public o duong dan khac tren Drive
# 5) Sau Cell 8, tai yolo_step3_artifacts.zip ve local, giai nen vao artifacts/yolo/
```

Ghi chu: không commit file `.pt` hay `.zip` lón vao Git (`.gitignore` dã lo). `exports/colab_bundle.zip` cung nam trong ignore (không commit).
