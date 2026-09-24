# Xác minh VERIFIER — Step 3: Baseline YOLO

## Giai đoạn A — viết test từ spec

Ngày: 2026-09-25. Branch đích: `step3/tests`, worktree `D:/FPTU/KLTN/PCB_Lab_verify3`.
Base: `2a1545e`. Workspace mở ban đầu là `main` ở `PCB_YOLO_EfficientAD_Lab`; không sửa main.

Đã đọc `docs/handoff/WORKFLOW.md`, `step3-contract.md`, `step2-verify.md`, hợp đồng Step 2
để tái dùng API, fixture/test Step 1–2, `docs/DECISIONS.md`, `docs/TRIAGE.md`, và các mục plan
được yêu cầu. Giám sát xác nhận plan chính thức là `PLAN.md` tại gốc; `docs/plan.md` không tồn tại.
Handoff Step 2 hiện có chỉ ghi Giai đoạn A, **không suy diễn Step 2 đã PASS Giai đoạn B**.

Không đọc source/handoff Step 3 implementation; chưa merge `step3/impl`. Chỉ dùng spec và
giải đáp giao diện của giám sát trong hội thoại, được chép tại `questions-verify.md`.

### Phạm vi và oracle

| File | Điều kiện được bảo vệ |
|---|---|
| `tests/conftest.py` | Đăng ký `smoke`, `artifacts`; thiếu `STEP3_ARTIFACT_DIR` tự skip, đường dẫn đã cấp nhưng sai thì fail. Smoke thiếu dependency skip rõ lý do. Chặn truy cập mạng Python trong test YOLO. |
| `tests/helpers/step3_spec.py` | Oracle độc lập đọc manifest/classes/hash/mtime; kiểm schema, nhãn, số lượng và bbox; fake engine không torch; checkpoint giả chỉ dùng kiểm guards. |
| `tests/test_yolo_config.py` | Trainer từ chối riêng từng null cả normal/smoke; typo cấp cao/lồng nhau; `_check_nulls` có positive control; `_merge_configs` từ chối khóa lạ; B01/B02 chỉ khác ba trường identity; hash ổn định, độc lập thứ tự khóa/seed, đổi infer thì hash đổi. |
| `tests/test_yolo_view.py` | Chặn fusion/test cả trong tuple hỗn hợp; YAML không test, đúng class order; good label rỗng; 6 chữ số thập phân; round-trip ≤0.01 px với `Sample.boxes` và manifest; copy/hardlink cùng nội dung/signature; limit chọn đầu sample_id; rebuild không giữ mẫu thừa; giả lập EXDEV; đổi một ảnh fixture đổi signature. |
| `tests/test_yolo_adapter.py` | Engine nhận RGB bất đối xứng kênh; describe đúng metadata; box 640×640, 800×600, 300×700 theo tọa độ tính tay; sort confidence/class/box; image_score max hoặc 0; thiếu file, checkpoint đổi một byte, smoke opt-in, class order engine sai. |
| `tests/test_yolo_extract.py` | Chặn test/train/partition lạ trước nạp artifact; dùng adapter thật + engine giả; một dòng mỗi ảnh, sample_id sorted, schema/provenance/timing null; lỗi ảnh còn dòng và error_reason, score null; inference rỗng hợp lệ có score 0; hai lần xuất giống byte. |
| `tests/test_yolo_artifacts.py` | Marker artifacts: schema run/artifact/calibration, checkpoint và manifest hash khớp file thật; smoke=false, git_dirty=false; đủ 460/306 dòng, không trùng/lỗi, conf_floor, báo saturation max_det. |
| `tests/test_yolo_smoke.py` | Marker smoke+dataset: CPU, một epoch, pretrained không tồn tại, view nhỏ, reload checkpoint bằng Ultralytics thuần; nạp adapter với allow_smoke qua bundle verifier dưới runs/smoke; so RGB adapter với predict(path) ≤0.5 px / 1e-3 conf; không sinh artifacts/ chính thức. |

Mỗi hàm test có comment liên kết plan/contract. Module Step 3 chỉ import trong test/fixture;
thiếu implementation phải fail khi chạy, không bị `importorskip` che thành PASS.

Test real-view giữ nguyên oracle plan: train **1.792 = 895 good + 897 defect**, calibration
**460 = 230 + 230**; bbox train theo lớp **1.216, 973, 1.228, 981, 884, 865**. Hash ảnh thực
trong view phải rời hash fusion/test lấy từ metadata. Các số này là kỳ vọng, chưa phải kết quả đo A.

Snapshot dataset so cả đường dẫn, mtime_ns, size, SHA-256 trước/sau, bao gồm file cache.
Riêng ảnh test và ảnh không xác định quyền đọc chỉ stat, không mở/hash/decode theo WORKFLOW.
Việc duyệt cây bằng os.walk chỉ để phát hiện ghi file; mẫu view/extraction được chọn qua manifest.

Giá trị `lr0=0.001`, `batch=2`, `workers=0`, augmentation tắt trong helper là **cấu hình test**,
không phê duyệt recipe train thật. Không hạ ngưỡng/kỳ vọng để ép số liệu khớp plan.

### Lệnh và kết quả thực chạy

Môi trường A: Python **3.14.7** từ `.venv` có sẵn của repo gốc. Venv đó có pytest/Pillow/PyYAML
nhưng thiếu numpy; chỉ bổ sung đường dẫn site-packages của venv Step 2 có sẵn để lấy numpy,
không đọc source implementation và không cài/sửa môi trường của implementer. Torch/Ultralytics chưa có.
**Đây không thay thế venv riêng + requirements CPU của Giai đoạn B.**

```powershell
$py = 'D:/FPTU/KLTN/PCB_YOLO_EfficientAD_Lab/.venv/Scripts/python.exe'
$env:PYTHONPATH = 'D:/FPTU/KLTN/PCB_Lab_impl/.venv/Lib/site-packages;D:/FPTU/KLTN/PCB_Lab_verify3/src'
$env:PYTHONDONTWRITEBYTECODE = '1'
# Vòng kiểm bản nháp tại $env:TEMP/pcb-step3-verifier-a trước khi ghi worktree:
& $py -m pytest tests --collect-only -q -p no:cacheprovider --rootdir=. --confcutdir=tests
& $py -m pytest tests -m 'not dataset and not smoke' -q --tb=no -p no:cacheprovider --rootdir=. --confcutdir=tests --basetemp=C:/Users/ASUS/AppData/Local/Temp/pcb-step3-verifier-a/pytest-fast-a1
& $py -m pytest tests -m smoke -q -rs -p no:cacheprovider --rootdir=. --confcutdir=tests --basetemp=C:/Users/ASUS/AppData/Local/Temp/pcb-step3-verifier-a/pytest-smoke-a1
```

| Kiểm tra bản nháp | Kết quả thật |
|---|---|
| Collection | **51 tests collected**, exit 0 |
| Fast | **46 failed, 2 skipped, 3 deselected**, 51.24 s; chưa có module YOLO Step 3 |
| Marker smoke | **1 skipped, 50 deselected**; thiếu torch, ultralytics; chưa train |
| Kiểm riêng oracle bằng script tạm stdlib/Pillow + loader Step 2 | Parse Python/comments PASS; nhận view hợp lệ + round-trip PASS; phát hiện cache mới PASS; từ chối class 6 PASS; fixture train/calibration mỗi tập 3 ảnh (1 good, 2 defect) |

Lần gọi pytest ban đầu bị sandbox chặn đọc thư mục cha `C:/Users/ASUS`; lần smoke dùng tmp mặc định
cũng gặp quyền truy cập `pytest-of-ASUS`. Đã chạy lại bằng root/confcutdir và basetemp riêng với quyền
được cấp, nhận kết quả ở bảng trên. Không tính lỗi môi trường đó là lỗi implementation.

Vòng cuối chạy **trên worktree `step3/tests`**, sau khi đưa các file đã kiểm vào đúng branch:

```powershell
Set-Location 'D:/FPTU/KLTN/PCB_Lab_verify3'
$env:PYTHONPATH = 'D:/FPTU/KLTN/PCB_Lab_impl/.venv/Lib/site-packages'
$env:STEP3_ARTIFACT_DIR = '' # A chỉ kiểm skip; chưa nghiệm thu artifact thật
$specTests = @(Get-ChildItem tests/test_yolo_*.py | ForEach-Object FullName)
& $py -m pytest @specTests --collect-only -q -p no:cacheprovider --rootdir=. --confcutdir=tests
& $py -m pytest @specTests -m 'not dataset and not smoke' -q --tb=no -p no:cacheprovider --rootdir=. --confcutdir=tests --basetemp=C:/Users/ASUS/AppData/Local/Temp/pcb-step3-verifier-a/pytest-fast-final --junitxml=C:/Users/ASUS/AppData/Local/Temp/pcb-step3-verifier-a/fast-final.xml
& $py -m pytest @specTests -m 'smoke or artifacts' -q -rs -p no:cacheprovider --rootdir=. --confcutdir=tests --basetemp=C:/Users/ASUS/AppData/Local/Temp/pcb-step3-verifier-a/pytest-optional-final
& 'C:/Program Files/Git/cmd/git.exe' diff main -- docs/handoff/step3-contract.md docs/handoff/WORKFLOW.md
& 'C:/Program Files/Git/cmd/git.exe' diff --check
```

- Collection Step 3: **51 tests**, exit 0.
- Collection toàn repo với `pytest tests --collect-only -qq -p no:cacheprovider --rootdir=. --confcutdir=tests`:
  **125 tests** (74 test có sẵn + 51 Step 3), exit 0; không chạy hành vi Step 1–2 trong lượt này.
- Fast: **46 failed, 2 skipped, 3 deselected**, **25.82 s**, exit 1.
  Đọc XML bằng `xml.etree.ElementTree` xác nhận **46/46 lỗi cùng nguyên nhân**
  `ModuleNotFoundError: No module named 'pcb_lab.models'`; **0 setup errors**.
  Không đổi test để che lỗi import; chờ merge ở B.
- Optional: **4 skipped, 47 deselected**, exit 0; ba test artifacts thiếu biến môi trường,
  một smoke thiếu torch/ultralytics. Đây là kiểm cơ chế skip, không phải PASS hành vi model.
- Diff hai file bất biến **rỗng**; `git diff --check` exit 0 (chỉ cảnh báo LF/CRLF của Windows).

Bản nháp và log/XML kiểm tra chỉ ở thư mục tạm, không commit. Trước commit kiểm `git status`;
chỉ stage 8 file thuộc `tests/` và 2 file thuộc `docs/handoff/`.

### Giới hạn và điểm bàn giao

- Chưa có kết luận PASS/PASS-WITH-CONDITIONS/FAIL cho implementation. Test đỏ A là dự kiến;
  collection thành công không chứng minh mô hình đúng.
- Test dataset thật, CPU train/reload và artifact thật **chưa được thực thi thành công**.
- Một số spelling/type schema chưa chốt được ghi rõ ở `questions-verify.md`; không sửa contract.
- Bắt lỗi bằng fake engine không chứng minh Ultralytics xử lý RGB đúng; smoke và C so với
  Ultralytics predict(path) mới kiểm ranh giới thư viện thật. Checkpoint ngẫu nhiên có thể trả rỗng.
- Chặn socket Python không phải chứng minh tuyệt đối không có mạng qua subprocess/native library;
  cần code review Giai đoạn B cho auto-download/cache/side effect.
- Không thay `step3-contract.md`, `WORKFLOW.md`, `PLAN.md`, code impl, requirements; không commit
  dataset, ảnh, checkpoint, dự đoán hay log chạy test.

## Giai đoạn B — chưa bắt đầu

Chờ người dùng báo IMPLEMENTER hoàn tất. Khi đó mới thực hiện merge được yêu cầu, venv riêng,
chạy nhanh rồi dataset/smoke, diff hai file bất biến, review source và ghi kết quả thật.

Nhật ký phá hoại dự kiến (chưa phải kết quả): class 6; bbox ngoài ảnh; bbox suy biến; checkpoint
đổi một byte; model.names đảo; test ở mọi API nhận partition; run đã tồn tại; null từng trường;
cùng seed/signature; đổi một ảnh/signature; fallback hardlink khác ổ đĩa (giả lập EXDEV đã có test).
Tự tính ít nhất hai số bằng stdlib (ví dụ số bbox theo lớp từ manifest và hash checkpoint smoke).

## Giai đoạn C — chưa bắt đầu

Chờ B PASS và người dùng báo B01/B02 seed42 train xong. Chạy artifact lần lượt; bổ sung nghiệm thu
git provenance/args/data/log, reload thuần, đối chiếu 5 ảnh, AP50 độc lập 101 điểm với ngưỡng 0.02,
recipe/nguồn pretrained/tài nguyên/model card. Các test artifacts A là nền kiểm tra, chưa đủ thay C.
PASS artifact chỉ nghĩa là đủ điều kiện dùng Bước 4–6, không khẳng định chất lượng mô hình.
