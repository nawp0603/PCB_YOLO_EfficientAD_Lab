# Step 3 — câu hỏi giao diện của VERIFIER

Ngày: 2026-09-25. Chỉ ghi vấn đề; không sửa `step3-contract.md` hay `WORKFLOW.md`.

## Các câu hỏi đã được giám sát giải đáp trong hội thoại Giai đoạn A

1. **Đường dẫn plan:** `docs/plan.md` không tồn tại. Giám sát xác nhận **`PLAN.md` tại gốc repo**
   là kế hoạch chính thức. Đã đọc §1.2–1.3, §3.1, §5 Bước 3, §6, §7.3, §8, §11.
2. **Config API:** dùng `pcb_lab.models.yolo.train._merge_configs(paths)`, `_config_hash(cfg)`,
   `_check_nulls(cfg)`, `ConfigError`. Tám khóa cấp cao hợp lệ: `model_id`, `recipe_id`,
   `architecture`, `pretrained`, `train`, `selection`, `infer`, `classes_ref`.
   Hash là SHA-256 của JSON canonical sắp khóa, không chứa seed.
3. **Detector metadata:** `model_id`, `checkpoint_sha256`, `class_order`, `infer_params`,
   `input_color="rgb"`. `describe()` bảo toàn các trường này.
4. **Fake engine:** tiêm trực tiếp `YoloDetector(engine, meta)`; engine có `class_names` và
   `infer(RGB uint8 HWC) -> ndarray[N,6]` trong tọa độ ảnh đầu vào.
   `from_artifact` kiểm `FileNotFoundError`, `SmokeArtifactError`, `ArtifactMismatchError`,
   `ClassOrderError` theo các điều kiện đã nêu trong hợp đồng/giải đáp.

Các điểm trên đã đóng; không cần thêm phê duyệt để viết test.

## Điểm schema còn chưa đặt tên chính xác, không chặn phần độc lập

- `partitions_used{train:n, val:"calibration", n}` chưa nêu tên khóa đếm calibration.
  Test chốt `train` là số nguyên, `val="calibration"`, không có fusion/test; tên khóa đếm
  còn lại cần đối chiếu schema được bàn giao ở B. Không tự sửa hợp đồng.
- `schema_version` chưa chốt kiểu string hay integer; test chấp nhận cả hai, loại bool.
- `peak_ram_mb` chưa chốt scalar hay object, hoặc tên trường method khi null. Test chấp nhận
  số không âm, object `{value, method}`, hoặc null. Ở C phải kiểm method được ghi khi null.
- Report calibration chỉ viết “infer params”, chưa chốt spelling. Test yêu cầu ít nhất một trong
  `infer_params` / `inference_params` / `infer` và giá trị bằng artifact, không tự bỏ qua trường này.
- Constructor `UltralyticsEngine` không được chốt. Test chặn nạp trọng số giả bằng cách thay
  **chỉ constructor này** bằng callable `*args, **kwargs -> FakeEngine(class_names=...)`.
  Không mock `YoloDetector.from_artifact`, hash hay kiểm class order trong các test guard/extraction.
  Nếu implementation dùng factory khác, chỉ sửa binding test khi chứng minh không phải lỗi hành vi.

Trong snapshot dataset, ảnh test (và ảnh ngoài danh sách manifest cho phép) chỉ lấy stat;
không mở byte để hash, không decode. Các file còn lại so SHA-256 + mtime + size, đồng thời
kiểm kê mọi đường dẫn/cache mới. Đây là cách giữ lệnh cấm mở ảnh test trong WORKFLOW khi kiểm dataset chỉ đọc.
