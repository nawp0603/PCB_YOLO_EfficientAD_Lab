# Bàn giao IMPLEMENTER — Step 2

## Phase 0

### Tóm tắt, căn cứ và tiêu chí hoàn thành

- Làm việc ở `D:\FPTU\KLTN\PCB_Lab_impl`, branch `step2/impl`, base `84e2c02`. Workspace mặc định của phiên vẫn là worktree `main`; không sửa ở đó hoặc worktree `step2/tests`.
- Không có AGENTS.md trong worktree/cha. Đã đọc `PLAN.md` §1.2, §3, §4.4, §5 Bước 2, §6, §9, §11; DECISIONS/TRIAGE; handoff Step 1, `reports/data_profile.md`, code `data/manifest.py`. Không dùng archive. Repo vẫn có `PLAN.md` ở gốc, không có `docs/plan.md`.
- Contract chưa có trong worktree impl; đọc bản người điều phối tại `D:\FPTU\KLTN\PCB_YOLO_EfficientAD_Lab\docs\handoff\step2-contract.md`, SHA-256 ban đầu `3e1fd25a0f07377ef1dc05d2091cc9dc15210b64e5650ad53c3cc167c7c5ce9a`. Không sao chép/sửa hoặc commit contract. Đọc các escape Markdown theo nghĩa hiển thị.
- Mục tiêu Bước 2: chuẩn hóa Sample/loader và một đường tiền xử lý CPU dùng chung cho API/CLI; bảo toàn split, màu, tọa độ, provenance; hỗ trợ letterbox YOLO, resize AD và tile native 256. TileManager phải ghép map có tính chất toán học đúng và gộp box theo Global NMS.
- Đầu vào: metadata canonical qua adapter Step 1, pixel RGB của train/phân vùng phát triển được phép, cấu hình có phiên bản và seed cho augment. Metadata test được đọc/đếm; không decode, xem hoặc xuất ảnh test trong công việc này.
- Đầu ra: các module data/inference được giao; `configs/preprocessing.yaml`; CLI kiểm một batch mỗi loader và preview metadata/hash; `reports/loader_check.json` tất định, bằng chứng kiểm chứng và handoff này.
- Hoàn thành khi loader giữ đúng good/train cho AD, canonical RGB/uint8/HWC và chuẩn hóa đúng một lần; plan/split/stitch/box/NMS đúng trên 640×640 và kích thước lạ; cùng ảnh qua API/CLI có pixel/hash/tọa độ nhất quán; augment đã được người dùng duyệt và kiểm tra; đã chạy thật, ghi cả lỗi nếu có. Không có model, threshold, benchmark/UI, protocol mới hoặc Gate 0.

### Schema và bằng chứng Step 1

- Tái dùng `load_dataset`, `load_manifest`, `dataset_path`, `resolve_dataset_root` và guard output của Step 1; không parse JSONL lần nữa trong module mới.
- Manifest dùng `split`, `pair_ids` list, `is_defect` bool, width/height, image, sha256, mask_status và `boxes[{class_id,class_name,xyxy,...}]`; metadata không chứa danh tính bo/design đã xác minh. Canonical đã là RGB PNG.
- Đọc metadata bằng adapter: mọi ảnh có kích thước 640×640. `deeppcb_20085159_good` có ba pair alias `20085159`, `20085167`, `20085168`; không được mất quan hệ pair khi tạo Sample.
- Step 1 EDA train/calibration: source mode L ở 2224 ảnh, RGB ở 28 ảnh; transform canonical là RGB, không resize. L phải nhân bản kênh, không đảo RGB/BGR.
- Pin_hole có cạnh ngắn min 20 px, P5/50/95 = 24/30/42; mouse_bite min 20 px, P5/50/95 = 24/28/37; spur min cạnh 21 px. Đây là kích thước bbox, không phải kích thước cấu trúc lỗi thật. Resize có rủi ro làm mất chi tiết dù bbox còn tồn tại.
- Biến môi trường DATASET_ROOT trong các process tool hiện chưa có giá trị (dù người dùng đã đặt ở terminal). Adapter vẫn xác định đúng `D:\FPTU\KLTN\DatasetVer4_Public` từ YAML. Khi chạy kiểm chứng sẽ đặt biến này rõ trong process lệnh; không sửa thiết lập hệ thống hoặc dataset.

### Ba quyết định mở và rủi ro

| Quyết định | Hướng áp dụng / trạng thái | Lý do và rủi ro |
| --- | --- | --- |
| pad_mode mặc định reflect | Dùng mặc định reflect đúng contract; hỗ trợ replicate và constant. Chỉ pad phải/dưới, không đổi tọa độ gốc. | Tránh dải hằng nhân tạo lớn ở biên, nhưng reflect có thể phản chiếu đường đồng thành cấu trúc không có thật. Phần pad không thuộc kết quả stitch; ảnh có chiều 1 cần xử lý biên rõ ràng. Chưa chứng minh mode này tốt nhất cho AD. |
| AD train dùng mọi tile, 9 tile/ảnh 640×640 | Tính số tile bằng TileManager.plan theo công thức ceil đã đính chính, không hard-code 9. Chỉ good train; mỗi tile giữ Sample/partition của ảnh nguồn. | Bao phủ native pixel giúp giữ lỗi nhỏ khi suy luận; 9 tile tăng chi phí và tương quan giữa mẫu, tile biên có pad. Num tiles không phải số ảnh/bo độc lập. |
| recipe aug_v1 | Đề xuất bên dưới; **chờ người dùng duyệt, chưa viết augment.py hoặc áp augmentation**. | Cần recipe nhẹ, seed tái lập, giữ cấu trúc lỗi nhỏ và không trộn biến dạng stress test vào train. |

### Recipe aug_v1 đề xuất để duyệt

1. Chỉ partition `train`; seed tạo RNG cục bộ, không đổi RNG toàn cục.
2. Chọn đều `k ∈ {0,1,2,3}`, xoay `k×90°` ngược chiều kim đồng hồ bằng hoán vị pixel; áp cùng phép lên box.
3. Sau xoay, lật ngang với xác suất 0.5; ảnh/bbox cùng biến đổi.
4. Không nội suy, crop, resize hoặc đổi scale; không brightness/contrast/color jitter, blur, noise, cutout, mosaic, MixUp, elastic, morphology. Toàn bộ pixel/diện tích box được giữ; W/H có thể hoán đổi với ảnh không vuông. Không sửa mảng/box input tại chỗ.
5. Chọn 8 phép đối xứng hình học đồng xác suất; giữ chính xác hình thái pin_hole/mouse_bite/spur. Rủi ro: phân phối hướng/handedness layout thay đổi; không bảo đảm tốt hơn train không augment. Các thay đổi photometric/alignment thuộc recipe/ablation khác có tên, không tự thêm.

Đã gửi yêu cầu duyệt qua câu hỏi bất đồng bộ. API bổ sung dự kiến: `apply_augmentation(image, boxes, *, partition, seed, recipe_id="aug_v1")`; tên callable này chưa do contract khóa, sẽ ghi rõ kiểu trả về sau khi recipe được duyệt.

### Mơ hồ và giả định

- **Chặn Sample/loader:** contract có `Sample.pair_id` số ít nhưng nguồn có nhiều `pair_ids`, chưa chốt kiểu/ý nghĩa. Đề xuất `pair_id: str` là alias nhỏ nhất theo từ điển và thêm `pair_ids: tuple[str,...]` đầy đủ; chờ người dùng chốt trước khi làm phần phụ thuộc. Ghi `questions-impl.md`.
- Contract đính chính công thức ceil cho tile được ưu tiên hơn phép tính sai trong PLAN; 640 có origins 0/224/448, canvas704, pad64 phía phải/dưới. `tile_size !=256` phải raise ValueError.
- Box dùng tọa độ biên liên tục xyxy (x2/y2 có thể bằng W/H), không cộng/trừ một pixel. NMS class-agnostic, giữ tie theo thứ tự input; chỉ suppress khi IoU > ngưỡng.
- Stitch dự kiến feather tuyến tính từ rìa overlap về phía tâm, chuẩn hóa tổng trọng số; rìa canvas không có láng giềng giữ trọng số dương. Kiểm cả map hằng, hàm tuyến tính theo tọa độ toàn ảnh, pixel chỉ thuộc một tile và crossfade giữa map tile khác nhau để phát hiện bước nhảy.
- `to_canonical` nhận uint8 array/PIL/path; L nhân ba kênh; LA/RGBA bỏ alpha như Pillow convert RGB, không composite lên nền mới; không tự xoay EXIF. Mảng float bị từ chối để không suy đoán đã normalize hay chưa.
- Normalize dùng `(uint8/255 - mean)/std` float32, chỉ nhận uint8; dữ liệu normalize lần đầu là float32 nên lần hai raise ValueError. Mean/std và trạng thái normalized được ghi meta; không tự normalize trong prepare_input (contract yêu cầu pixels uint8).
- Meta hình học giữ các field contract và thêm version/hash; PreparedInput.meta là dict JSON được. Hash cấu hình từ JSON chuẩn hóa có defaults; hash PreparedInput sẽ ràng buộc pixel/shape/dtype/meta, không phụ thuộc đường dẫn input/out hay thời gian.
- Letterbox/resize dùng bilinear, letterbox pad114 cân hai phía (nếu lẻ phần dư về phải/dưới). Upsample map bilinear float32, không overshoot. Các tham số thực dùng được ghi meta.
- Metadata test lấy được qua Dataset nhưng Sample.load_image bị chặn mặc định; check_loaders chỉ kiểm metadata/gating cho test, không bật allow_test để decode. Preview đường dẫn canonical sẽ kiểm membership trước khi mở ảnh; ảnh ngoài dataset dùng canonical conversion thông thường.
- `AdTrainSet.tiles_per_image` suy từ kế hoạch của metadata mọi good train. Nếu dataset khác có số tile/ảnh không đồng nhất, không báo num_tiles bằng công thức sai: raise ValueError với thông báo rõ, giữ contract num_tiles=n_samples×tiles_per_image. Loader canonical hiện đồng nhất.

### File dự định tạo/sửa

- Tạo `src/pcb_lab/data/{samples,augment}.py`, `src/pcb_lab/inference/{__init__,preprocessing,tiling}.py`, `configs/preprocessing.yaml`, `scripts/{check_loaders,prep_preview}.py`.
- Thêm `requirements/step2.txt`; cập nhật dependencies `pyproject.toml` để khai báo NumPy cho API; không sửa manifest adapter trừ khi bắt buộc. NumPy dùng array, tile/map/vectorized blending/NMS, không framework nặng; khóa phiên bản sau khi cài/chạy thật.
- Ghi `docs/handoff/{step2-impl,questions-impl}.md`, `reports/loader_check.json` và bằng chứng kiểm chứng text. Dùng `reports/verification/` đã ignored cho fixture, preview, script tự kiểm chứng tạm; không sửa tests hoặc cấu hình pytest.
- Không sửa PLAN, DECISIONS, TRIAGE hoặc contract. Mỗi commit kiểm git status/staged paths, không stage dataset/ảnh/key/token; không merge/push/rebase/đụng branch khác.

## Phase 1

Chưa viết code tại thời điểm ghi Phase 0. Augment và ánh xạ pair_id đang chờ phản hồi; tiếp tục được TileManager/preprocessing độc lập.

## Phase 2

Chưa chạy loader/preview hoặc kiểm tính chất TileManager; chưa tuyên bố hoàn thành.
