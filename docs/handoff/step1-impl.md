# Bàn giao IMPLEMENTER — Step 1

## Phase 0

### Căn cứ và mục tiêu

- Branch đã kiểm tra: `step1/impl`. Không có AGENTS.md trong workspace hoặc các thư mục cha; AGENTS.md người dùng đọc được nhưng rỗng.
- Đã đọc `PLAN.md` §1, §4.1–4.2, §5 Bước 0–1, §6, §11; `docs/DECISIONS.md`, `docs/TRIAGE.md`, `docs/handoff/step1-contract.md`. Không dùng archive, không cần đọc Review.
- Sai khác đường dẫn nhỏ: repo không có `docs/plan.md`; `PLAN.md` tại gốc là bản v3 có đủ các mục được yêu cầu. TRIAGE xác nhận tài liệu gốc ở root. Dùng file này làm căn cứ, giữ nguyên nội dung. Hợp đồng hiện có escape Markdown (`\\_`, `&#x20;`); đọc theo nghĩa hiển thị, không sửa hợp đồng.
- Bước 1 xác minh bản canonical có thể dùng cho thí nghiệm: ảnh/nhãn/hash đúng, số lượng và membership khớp manifest/holdout, không giao nhóm nguồn/cặp/hash giữa bất kỳ hai partition. EDA mô tả phân phối lớp, nhóm, số lỗi, kích thước box và rủi ro shortcut; quan sát overlay train/calibration đủ sáu lớp.
- Đầu vào: dataset chỉ đọc; manifest, holdout, classes, protocol, release đã khóa, các đường dẫn ảnh/nhãn được manifest chỉ định. Không tự ghép cặp hoặc chia lại tập.
- Đầu ra: `reports/data_audit.json`, `reports/data_profile.md` tiếng Việt, `reports/overlays/<partition>/<class_name>/*.png`, API và CLI đúng hợp đồng.
- Hoàn thành triển khai khi đã chạy thật, báo mọi FAIL/WARN và mọi sai khác với PLAN §1.2–1.3, kiểm tra tính tất định, xem overlay đủ sáu lớp. Điều kiện nghiệm thu dữ liệu của plan còn yêu cầu giải quyết lỗi nghiêm trọng và không rò rỉ; nếu audit FAIL phải báo nguyên trạng, không sửa dataset để làm PASS.

### Metadata thật đã đọc

`DATASET_ROOT` chưa được đặt trong process/user/machine. Dataset tồn tại ở mặc định hợp đồng `D:\FPTU\KLTN\DatasetVer4_Public`. Git có tại `C:\Program Files\Git\cmd\git.exe` nhưng không nằm trên PATH. Python cũng chưa có trên PATH; môi trường cũ được README nhắc tới không chạy được trong sandbox, không sửa môi trường đó.

Đã đọc README.md, DATASET_CARD.md và các metadata canonical bên dưới `benchmarks/deeppcb/`:

- `manifests/samples.jsonl`: mỗi dòng một object; `sample_id`, `split`, `source_group`, `pair_ids` (list), `is_defect` (bool), `image`, `label`, `width`, `height`, `sha256`, `label_sha256`, `pixel_sha256`, `boxes`. Box gồm `source_class_id`, `class_id`, `class_name`, `xyxy` pixel. Có metadata provenance/quality/transform và danh tính bo/design để null.
- `configs/classes.json`: `names` là list theo ID YOLO 0-based; `source_to_class_id` ánh xạ nguồn 1-based. Sáu tên: open_circuit, short, mouse_bite, spur, spurious_copper, pin_hole. Không gộp pin_hole với missing_hole.
- Nhãn `.txt` thực tế: mỗi dòng `class_id cx cy w h`, chuẩn hóa theo kích thước ảnh, 10 chữ số thập phân. Good dùng file rỗng. Đối chiếu với box manifest, không chỉ tin một nguồn.
- `splits/holdout.json`: `partitions`, `groups`, `samples`, `pair_ids`, `actual_pair_counts`, thông tin lựa chọn và component groups. Danh sách sample là chuẩn đối chiếu membership, không chỉ đối chiếu tổng. Bốn partition train/calibration/fusion/test; số lượng metadata lần lượt 1792/460/306/440 và 897/230/153/220 pair; đây chưa phải kết quả audit.
- Một canonical good có thể đại diện nhiều pair. Dataset card giải thích ba template trắng trùng pixel được giữ một đại diện, hai alias; template lẻ 90100034 không xuất canonical. Không giả định mỗi pair phải có đúng hai canonical image.
- `configs/protocol.json`: good train cho AD; calibration chọn ngưỡng; fusion của nguồn dành cho fit head; test cuối; không pixel mask; source_group không phải design/bo đã xác minh; giữ good nền trắng.
- `release.json`: `files` là map đường dẫn tương đối dataset root → SHA-256 (18 entry lúc đọc); có dataset_signature, tham chiếu inventory/manifest. Sẽ tính n_locked từ map, không hard-code 18.
- Đã đọc đoạn định nghĩa hash trong script kiểm chứng nguồn chỉ để xác định schema: pixel hash = SHA-256 của byte RGB liên tiếp theo hàng, không gồm header/kích thước. Không chạy script nguồn hay ghi dataset.

### Quyết định ảnh hưởng phạm vi

- R1: giữ split source-group, không LOGO-CV; báo số nhóm, giới hạn độc lập thống kê; không gọi source_group là bo đã xác minh. PLAN v3 nói bootstrap không giải quyết domain shift.
- R7/S2/V2: đo box nhỏ thực tế, báo tác động hình học của resize 640→256 (hệ số 0.4), không khẳng định lỗi biến mất chỉ từ bbox. Tile native 256 chưa được implement ở Step 1, không mô phỏng model.
- R2 và PLAN §6: giữ calibration riêng; giữ tên partition fusion trong báo cáo dù dự án sau dùng vai trò development_selection. Không viết protocol.yaml ở bước này.
- S5 và hợp đồng: chỉ CPU; không torch/ultralytics/anomalib, không Gate 0 hoặc training. R3/V1, R4/H2, R5/R6/R8–R10, S1/S3/S4 thuộc bước sau, không chặn audit.
- Không có pixel ground truth: không tạo mask hay metric segmentation. Good trắng vẫn được đếm; báo dấu hiệu shortcut tên file, template đã làm sạch, nguồn grayscale→RGB và lỗi tổng hợp mà không suy ra model đã học shortcut.

### Giả định nhỏ đã chốt trước code

- CLI ưu tiên `--dataset-root` > biến môi trường DATASET_ROOT > YAML mặc định. Đường dẫn YAML tương đối repo; đường dẫn trong manifest/release tương đối dataset root. Từ chối output nằm trong dataset và đường dẫn input thoát dataset.
- File metadata bắt buộc thiếu/JSON không đọc được hoặc JSONL sai cú pháp là lỗi không khởi chạy (CLI 2). Bản ghi mẫu hỏng nhưng JSON còn đọc được và ảnh/nhãn thiếu/hỏng: ghi errors, FAIL check liên quan và tiếp tục (CLI 1).
- Hash leakage tính lại từ file/pixel; thiếu khả năng tính thì check leakage tương ứng FAIL do chưa thể chứng minh tách biệt. Giao pair là giao tập tất cả pair_ids, không phải sample_id.
- EDA chi tiết kích thước và lựa chọn overlay chỉ train/calibration. Các bảng số đếm/membership lớp của test phục vụ kiểm toán, không dùng chọn ngưỡng. Ngưỡng cạnh nhỏ là `min(width,height) < 16` và `< 32` px tại ảnh gốc: tương ứng <6.4/<12.8 px sau resize-256; chỉ là dấu hiệu rủi ro hình học.
- `per_class=5` nghĩa là tối đa 5 ảnh mỗi lớp mỗi partition được cho phép, thứ tự lựa chọn cố định ưu tiên box nhỏ/vùng biên; báo thiếu mẫu hoặc lỗi render, không bỏ lỗi im lặng.
- Số lượng ảnh/lớp/nhóm và số khóa đều tính từ dữ liệu. Kích thước kỳ vọng 640 là yêu cầu check `image_size_640` trong hợp đồng.
- Không có mơ hồ đang chặn Phase 1. Nếu phát hiện xung đột contract sẽ ghi questions-impl.md và dừng phần bị ảnh hưởng.

### File dự định tạo/sửa và dependency

- Tạo `pyproject.toml`, `requirements/step1.txt`, `configs/dataset.yaml`, `src/pcb_lab/__init__.py`, `src/pcb_lab/data/{__init__,manifest,audit,eda}.py`, `scripts/audit_dataset.py`.
- Cập nhật chính handoff này; tạo báo cáo JSON/Markdown; overlay và môi trường riêng không commit. Bổ sung `.gitignore` để loại môi trường/cache/output tạm (file này đã tồn tại untracked trước phiên).
- Không sửa tests/, PLAN/DECISIONS/TRIAGE/contract; không cấu hình pytest; không thao tác branch khác.
- Dự kiến dependency runtime chỉ Pillow (decode RGB/render overlay) và PyYAML (đọc cấu hình); dùng thư viện chuẩn cho hash, bảng, percentile để tránh thêm numpy/matplotlib. Setuptools chỉ phục vụ đóng gói. Phiên bản sẽ khóa theo môi trường riêng đã cài và chạy thật ở Phase 2.

## Phase 1

Chưa viết code tại thời điểm ghi Phase 0.

## Phase 2

Chưa chạy audit; chưa tuyên bố hoàn thành.
