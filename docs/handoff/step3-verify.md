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

## Giai đoạn B — kiểm implementation

Ngày 2026-09-25, người dùng cho phép đọc và merge. Đã merge **`step3/impl@b646589`** vào
`step3/tests` bằng `git merge --no-edit step3/impl`; merge commit **`3b78289`**, không xung đột.
Không merge main, không sửa source implementation.

### B.1. Môi trường và phạm vi kiểm

- Venv riêng: `D:/FPTU/KLTN/PCB_YOLO_EfficientAD_Lab/.cache/step3-verifier-b/.venv`.
  Cài **nguyên `requirements/step3.txt` + pytest** bằng uv, không dùng site-packages venv impl.
- Python 3.14.7; torch **2.14.0+cpu**, torchvision **0.29.0+cpu**, Ultralytics **8.4.161**,
  numpy 2.5.3, Pillow 12.3.0, PyYAML 6.0.3, pytest 9.1.1. `torch.version.cuda=None`, CUDA unavailable.
- Cài đặt ban đầu bị sandbox chặn socket; chạy lại có quyền mạng để **cài dependency** thành công.
  Mạng trong test YOLO vẫn bị chặn bằng fixture; `YOLO_OFFLINE=true` tránh DNS probe lúc import.
- Đọc diff `main...step3/impl`, toàn bộ bốn module YOLO + CLI/config, notebook, packager, RUN_GPU,
  các API Step 1–2 liên quan, và **source Ultralytics của đúng bản cài** để kiểm hành vi thư viện.
- `git diff main -- docs/handoff/step3-contract.md docs/handoff/WORKFLOW.md` **rỗng**.
  Diff implementation không có checkpoint/prediction/ảnh/zip đã commit; số dataset chỉ nằm trong
  docstring/handoff, không dùng làm oracle đếm trong code chạy view.

Lệnh chính (PowerShell, cwd `D:/FPTU/KLTN/PCB_Lab_verify3`):

```powershell
$base = 'D:/FPTU/KLTN/PCB_YOLO_EfficientAD_Lab/.cache/step3-verifier-b'
$py = "$base/.venv/Scripts/python.exe"
uv --cache-dir "$base/uv-cache" venv --python 3.14.7 "$base/.venv"
uv --cache-dir "$base/uv-cache" pip install --python $py --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements/step3.txt pytest
$env:PYTHONPATH = ''
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:DATASET_ROOT = 'D:/FPTU/KLTN/DatasetVer4_Public'
$env:YOLO_OFFLINE = 'true'
$env:YOLO_CONFIG_DIR = "$base/settings"
$env:MPLCONFIGDIR = "$base/mpl"
$env:PATH = 'C:/Program Files/Git/cmd;' + $env:PATH
$env:STEP3_ARTIFACT_DIR = ''
# Các lượt dùng tên basetemp/xml/log riêng: fast-original, fast-corrected, full.
& $py -m pytest -m 'not dataset and not smoke' --tb=short -q -p no:cacheprovider --basetemp="$base/fast-original-tmp" --junitxml="$base/fast-original.xml"
& $py -m pytest -m 'not dataset and not smoke' --tb=short -q -p no:cacheprovider --basetemp="$base/fast-corrected-tmp" --junitxml="$base/fast-corrected.xml"
& $py -m pytest --tb=short -q -p no:cacheprovider --basetemp="$base/full-tmp" --junitxml="$base/full.xml"
```

Các lệnh thực tế redirect stdout/stderr vào các file `.log` cùng prefix, không commit log/checkpoint.
Trước lượt full, tạo settings dir và **copy font Arial có sẵn từ Windows** vào
`settings/Ultralytics/Arial.ttf`. Đây là chuẩn bị của verifier để đi tiếp sau khi đã bắt được
network request tìm font ở lượt đầu, **không phải implementation đã sửa lỗi mạng**.

### B.2. Sửa test chỉ khi có bằng chứng test sai

| Sửa trong verifier | Bằng chứng và phần kỳ vọng giữ nguyên |
|---|---|
| `names` chấp nhận khóa YAML số hoặc chuỗi số; zero-count class có thể vắng trong `box_counts` | Contract chốt index/thứ tự/số lượng, không chốt representation. Chuẩn hóa rồi vẫn so đủ sáu lớp và đúng số box. |
| Gọi `detector.image_score` | `adapter.py:169` cung cấp method; contract không bắt buộc module-level function. Kỳ vọng max/empty=0 giữ nguyên. |
| Mock thêm `ultralytics.YOLO` khi checkpoint chứa bytes giả | `adapter.py:119–138` deserialize **trước** constructor engine. Mock cũ làm UnpicklingError dù guard có thể đúng. Giữ nguyên `from_artifact`, hash, smoke guard, so `model.model.names`; chỉ thay deserializer/engine. |
| Failed-row score không bắt buộc null nếu đã có `error_reason` | Contract yêu cầu không bỏ dòng/không GOOD im lặng, không định kiểu score khi lỗi. Test độc lập xác nhận lỗi rõ ràng và đủ dòng; không gọi score=0 kèm error là lỗi “im lặng”. |
| Smoke kiểm `.yaml` + pretrained SHA null, không ép `args.pretrained=False` | Source Ultralytics `engine/trainer.py:858–865`: YAML + boolean True vẫn khởi tạo không weights. Không đổi yêu cầu random init/no-download. Chuyển kiểm schema xuống sau reload để không mất bằng chứng reload khi metadata sai. |
| Sửa `sys.environ` thành `os.environ` trong hai test CLI Step 2; `os_pathsep()` chưa định nghĩa thành `os.pathsep` | AttributeError/NameError đã tái hiện; đây là lỗi test, không phải lỗi implementation. Hai CLI được chạy lại thành công. |
| Ghép bbox một-một theo class và dung sai thay vì sort rồi zip | Hai GT có cùng x1=261 bị đảo thứ tự khi label sáu chữ số làm x1 thành 260.99968/261.00032 (`deeppcb_12100121_defect`), khiến zip báo lệch 407 px. Đo lại tọa độ chỉ lệch tối đa 0.00032000000010157237 px. Dùng ghép một-một, giữ nguyên số box và ngưỡng **0.01 px**; regression vẫn bắt box lệch 1 px. |
| Chặn model creation trong test config bị từ chối | Lượt đầu typo lồng nhau lọt tới trainer/font download. Sentinel cho lỗi chính xác tại ranh giới, tránh vô tình train ở một test “nhanh”; vẫn yêu cầu đúng ConfigError. |
| Tạo settings directory trước import smoke | Ultralytics 8.4.161 yêu cầu parent tồn tại, nếu không có thể fallback ngoài thư mục dự kiến. Đây là sửa fixture cô lập import. |

Các test phá hoại bổ sung nằm trong `tests/test_yolo_adversarial.py`. Không giảm oracle plan,
ngưỡng tọa độ/confidence hoặc sửa implementation để làm xanh test.

### B.3. Kết quả chạy

| Lượt | Kết quả thật |
|---|---|
| Fast nguyên test A, sau merge/cài đúng CPU deps | **14 failed, 90 passed, 11 skipped, 10 deselected**, 43.02 s |
| Fast sau sửa binding/test sai và bổ sung 13 ca phá hoại | **15 failed, 102 passed, 11 skipped, 10 deselected**, 118.90 s |
| Full (thêm hai probe pretrained/resume; gồm dataset và smoke) | **20 failed, 108 passed, 12 skipped**, 297.14 s |
| Targeted sau sửa oracle bbox và test CLI, thêm kiểm tính tất định độc lập schema | **3 failed, 11 passed**, 50.76 s; hai lỗi view thật, một NameError `os_pathsep` của test cũ |
| Chạy lại riêng CLI sau sửa `os_pathsep` | **1 passed**, 9.51 s |
| Collection cuối sau mọi chỉnh sửa test | **142 tests collected**, 0.40 s, exit 0 |

Không ghi một lượt full mới chưa chạy: trong 20 failure của full, hai failure thuộc test
(CLI cũ và cách ghép bbox) đã được sửa có bằng chứng rồi kiểm lại. **18 ca còn lại là lỗi
implementation**, bao gồm smoke fail metadata sau khi train/reload/predict đã hoàn tất.
Lượt targeted chạy toàn bộ `test_yolo_view.py` và xác nhận real-view PASS: số ảnh/box đúng,
hash ảnh view rời fusion/test, snapshot nguồn không đổi, round-trip trong ngưỡng.
Hai test thêm cuối xác nhận oracle ghép box và extraction hai lần giống byte cũng PASS.

Các node chạy lại (cùng các biến môi trường B.1, log/XML `targeted` và `loader-recheck`):

```powershell
& $py -m pytest tests/test_yolo_view.py tests/test_integration_real.py::test_check_loaders_deterministic tests/test_preprocessing.py::test_prep_preview_cli_matches_api tests/test_yolo_adversarial.py::test_roundtrip_oracle_handles_equal_x1_without_relaxing_tolerance tests/test_yolo_adversarial.py::test_extraction_byte_determinism_independent_of_schema --tb=short -q -p no:cacheprovider --basetemp="$base/targeted-tmp" --junitxml="$base/targeted.xml"
& $py -m pytest tests/test_integration_real.py::test_check_loaders_deterministic --tb=short -q -p no:cacheprovider --basetemp="$base/loader-recheck-tmp" --junitxml="$base/loader-recheck.xml"
```

Smoke thực tế: **1 epoch CPU, 16 ảnh train + 16 ảnh calibration**, không tạo `output/artifacts/`.
Checkpoint được nạp bằng Ultralytics thuần, kiểm names và so adapter RGB với predict(path)
đạt **≤0.5 px / ≤1e-3 confidence**; cuối test mới fail vì `git_commit=null`.
Snapshot dataset thật trước/sau smoke không đổi; cache nhãn nằm dưới output/data_refs,
không nằm trong DATASET_ROOT. AP của smoke đều 0; đây không phải đánh giá chất lượng baseline.

Skip artifact là chủ ý: B chưa có artifact train thật để nghiệm thu C. Chín test augmentation
Step 2 có sẵn tự skip vì binding A của Step 2 không nhận API `apply_augmentation(partition=...)`;
không tính những skip này là PASS. Test riêng Step 3 về augmentation chạy thật và phát hiện lỗi.

### B.4. Bảng phát hiện

Các line dưới đây tham chiếu source sau merge, cùng nội dung `step3/impl@b646589`.

| ID / mức | File:dòng | Bằng chứng / hệ quả | Đề xuất cho IMPLEMENTER |
|---|---|---|---|
| B-C01 **Critical** | `src/pcb_lab/models/yolo/train.py:103–113`, `:139–146` | `_geometric_augment(..., flip=True)` đổi `cx` của bbox nhưng không lật pixel. Probe ảnh bất đối xứng sai **36/300 giá trị kênh**, còn box đã chuyển sang nửa kia. Nhánh ảnh defect bị lệch ảnh–GT. | Dùng lại augmentation Step 2 hoặc bảo đảm xoay/lật đồng nhất ảnh và box; kiểm cả good/defect. |
| B-C02 **Critical** | `train.py:329–360`, `:398–401` | Pretrained tương đối được kiểm tồn tại trước `chdir(runs_base)` rồi truyền nguyên chuỗi sau chdir. Đường dẫn local hợp lệ trở thành đường dẫn thiếu; nhánh YOLO loader có thể tự download dù `allow_download=False`. | Resolve checkpoint thành absolute trước chdir; kiểm lại đường truyền vào trainer và khóa tải phụ. |
| B-C03 **Critical** | `train.py:347–353`; `extract.py:41–42` | Với dataset **giả**, train tạo `ILLEGAL/runs/smoke/.../seed42` rồi mới bị view guard chặn; extract ghi thẳng `ILLEGAL/preds_calibration.jsonl` trong dataset. | Guard tất cả output ngay đầu, trước bất kỳ mkdir/write; áp cả resolve/junction và model_id/path escape. |
| B-C04 **Critical** | `train.py:33–39`, `:301`, `:377`; Ultralytics `data/utils.py:667`, `utils/checks.py:452`, `:1020` | Lượt đầu đã bắt request `requests.head` tìm font khi allow_download=False. Env `ULTRALYTICS_NO_ANALYTICS` không được thư viện bản cài đọc. Review source: import có DNS probe; nhánh AMP GPU có thể nạp/tải checkpoint phụ. | Cô lập settings/cache dưới out_root, khai báo offline trước import, cấp sẵn tài nguyên và quản lý mọi download. Nhánh AMP chưa chạy được trên CPU, chỉ kết luận từ source. |
| B-M01 **Major** | `view.py:97–103`, `:157–165` | `link='copy'` vẫn gọi `os.link`; `samefile=True`. EXDEV được bắt ở helper rồi copy nhưng YoloView vẫn báo `hardlink`. | Truyền mode vào helper, không link khi yêu cầu copy; trả mode thực đã dùng. |
| B-M02 **Major** | `view.py:69–70`, `:97`, `:115` | Rebuild full→limit 1 good/1 defect báo 2 nhưng còn 3 ảnh, stale sample vẫn ở view. Ảnh đích đã tồn tại không được so hash/refresh. | Dựng staging rồi thay view có kiểm soát hoặc dọn/đối chiếu đúng manifest trước reuse. |
| B-M03 **Major** | `train.py:54–67`, `:304–308` | Unknown key ở train/selection/infer không bị ConfigError, lọt tới model creation; top-level và bốn null được chặn đúng. | Validate schema lồng nhau/type/recipe trước đọc dataset/tạo output. |
| B-M04 **Major** | `train.py:345–346`, `:361–383`, `:451` | Empty run directory được chấp nhận. `resume=True` chỉ bỏ guard, không truyền resume/checkpoint cho trainer; `resumed_from` luôn null. | Từ chối mọi run directory đã tồn tại trừ resume thật; nối đúng last.pt/trạng thái optimizer/epoch và provenance. |
| B-M05 **Major** | `adapter.py:160–162`, `:127–128` | Sort bỏ x2/y2 nên đảo thứ tự engine làm output khác. Model có canonical six-name prefix + lớp thứ 7 vẫn load. | Sort đủ bốn tọa độ; so class list đầy đủ và metadata/preprocessing provenance. |
| B-M06 **Major** | `extract.py:45`, `:76` | JSONL dùng `xyxy` thay `xyxy_original`. Run ID được tự dựng thay vì dùng run_manifest; fixture có run ID hợp lệ khác thì provenance không khớp. | Xuất đúng Detection schema và lấy run_id từ manifest đã xác minh. |
| B-M07 **Major** | `train.py:267–273`, `:443–451` | `_git_state` dùng `subprocess` chưa import, nuốt NameError thành `(None,None)`. `best_epoch`/`train_time_s` luôn null; epochs_run là cấu hình, early_stopped=(not smoke), args_used chỉ overrides chứ chưa phải toàn bộ args thật. | Ghi trạng thái/args/epoch/timing thật từ trainer; không nuốt lỗi provenance. |
| B-M08 **Major** | `train.py:514`, `:522–531` | pr_conf lấy từ `metrics.box.f1` (vector F1, không phải confidence), `_safe` biến thành null. Đếm GT tìm field `gt_nb` sai API bản cài. Smoke ghi `pr_conf=null` và cả sáu `n_gt_boxes=null` dù validator đã đếm 56 GT. | Lấy đúng confidence F1-optimal; đếm GT từ manifest hoặc trường đúng (`DetMetrics.nt_per_class` ở bản cài), giữ phân biệt operating threshold. |
| B-M09 **Major** | `train.py:496–506`, `:454–489`, `:577–580` | Validation cuối không truyền project/name dưới out_root; checkpoint reload bỏ project trong overrides. Không có code tạo train.log/report validation yêu cầu; model card versions/config_hash luôn n/a do đọc từ cfg không có các trường này. | Định vị mọi output và hoàn thiện files/model card theo schema, ghi phiên bản/hash thật. |
| B-M10 **Major** | `docs/RUN_GPU.md:93–101`; `notebooks/train_yolo_colab.ipynb:205` | Hướng dẫn chỉ mang artifacts về và notebook zip artifacts/yolo; artifact.json tham chiếu run_manifest dưới runs/ bị bỏ. Không có lock CUDA riêng, notebook giữ torch/torchvision hiện có. | Bundle cả manifest/env/args/log có hash/path portable; khóa và ghi lại môi trường CUDA đã kiểm. |
| B-m01 **Minor** | `train.py:292–296`, `:336` | allow_download=True vẫn gọi helper chỉ raise nếu file thiếu; source chỉ ghi local hoặc chuỗi chung, không URL/releaseversion. | Thực hiện đúng tính năng đã công bố hoặc báo rõ chưa hỗ trợ; provenance ghi nguồn/phiên bản cụ thể. |
| B-m02 **Minor** | `train.py:219–226`, `:240–263`, `:411` | Hook đổi epoch augmentation không được đăng ký; `_last_dataset` bị val ghi đè. “peak RAM” hiện lấy RSS một thời điểm, fallback dùng sys chưa import. | Nối hook thực và đo peak đúng method, hoặc ghi null + method trung thực. |

### B.5. Nhật ký phá hoại

Mọi mutation và thử đường output trái phép dùng fixture riêng; không thay ảnh/nhãn dataset thật.
“PASS” ở bảng này chỉ nói phép thử bảo vệ tương ứng, không phải toàn Step 3 PASS.

| # | Cách phá / phép thử | Kết quả |
|---|---|---|
| 1 | Đặt class_id=6 trong manifest fixture | **PASS**: ValueError, snapshot nguồn không đổi. |
| 2 | Box x1=-1 ngoài ảnh | **PASS**: ValueError. |
| 3 | Box x1=x2 suy biến | **PASS**: ValueError. |
| 4 | Đổi một byte best.pt, giữ SHA metadata cũ | **PASS**: ArtifactMismatchError. |
| 5 | Đảo hai tên trong checkpoint model.names | **PASS**: ClassOrderError; giữ metadata artifact đúng. |
| 6 | Thêm tên lớp thứ bảy sau sáu lớp chuẩn | **FAIL**: loader chấp nhận prefix. |
| 7 | Yêu cầu test/fusion ở view, test/train/unknown ở extraction, cả tuple hỗn hợp | **PASS**: PermissionError; không nạp model cho extraction bị cấm. |
| 8 | Lần lượt lr0/batch/workers/augment=null, cả smoke/normal | **PASS**: ConfigError. |
| 9 | Unknown key cấp cao và nested train/selection/infer | Cấp cao **PASS**; ba nested **FAIL**, đến model creation. |
| 10 | Run directory đã có sentinel / directory rỗng | Có sentinel **PASS**, nội dung giữ nguyên; rỗng **FAIL**. |
| 11 | resume=True với đường last.pt có sẵn | **FAIL**: spy tại trainer không nhận resume/checkpoint; dừng trước train, không sửa checkpoint. |
| 12 | Dựng cùng view hai lần / đổi một ảnh fixture + metadata hợp lệ | **PASS**: signature giữ nguyên / thay đổi. Không tuyên bố đã train hai seed-run đầy đủ. |
| 13 | Ép os.link ném EXDEV (khác ổ đĩa giả lập) | Nội dung copy được nhưng **FAIL** link_mode vẫn hardlink; chưa thử bằng ổ đĩa vật lý khác. |
| 14 | Yêu cầu link='copy', so inode bằng samefile | **FAIL**: vẫn hardlink với nguồn. |
| 15 | Rebuild view đầy đủ thành limit 1 good/1 defect | **FAIL**: file ảnh/nhãn cũ không bị loại. |
| 16 | out_root/out_dir là thư mục con dataset giả | **FAIL**: train tạo directory trước reject; extraction ghi JSONL. |
| 17 | Engine đảo các box có cùng conf/class/x1/y1, khác x2/y2 | **FAIL**: thứ tự output thay đổi. |
| 18 | Ảnh bất đối xứng + flip=True trong augmentation | **FAIL**: pixel không lật, box đã lật. |
| 19 | Engine ném RuntimeError trên ảnh calibration | **PASS** ca độc lập: đủ dòng, mỗi error_reason chứa sentinel; không lỗi im lặng. |
| 20 | Pretrained tương đối có sẵn, allow_download=False; ghi nhận path khi nạp | **FAIL**: đến YOLO factory, path không còn tồn tại sau chdir; spy chặn trước nạp/download. |

### B.6. Đo độc lập bằng stdlib

`tests/helpers/step3_independent.py` chỉ dùng stdlib; không import pcb_lab/torch/Ultralytics.
Đã chạy với dataset thật, chỉ mở metadata, label train và release; không mở ảnh test.

- Manifest: train **1.792 (895/897)**, calibration **460 (230/230)**, fusion **306 (153/153)**,
  test metadata **440 (220/220)**.
- Tổng bbox train từ manifest **6.147**; đếm lại dòng nhãn train trực tiếp cũng **6.147**.
  Theo ID 0..5, cả hai cách cho **1216 / 973 / 1228 / 981 / 884 / 865**.
- SHA-256 **file** manifest: `0ea6bdb4182b9adbd19b15ca00d8a1711d668f5dc06c1cbff9d9555ace250073`.
- SHA-256 **file** release.json: `9719b5cc2e4475bb1a911a9e6fab345306b3239c084522d556b7237c83266de6`.
  Lưu ý runner `_release_files_sha` hash JSON của map files, không hash bytes release.json;
  chưa đánh đồng hai định nghĩa này thành lỗi checkpoint.

Lệnh stdlib có thể chạy lại để tái lập cả đếm nhãn và hash checkpoint:

```powershell
$smokeRun = "$base/full-tmp/test_one_epoch_cpu_train_reloa0/output/runs/smoke/B01_yolo11n/seed42"
& $py tests/helpers/step3_independent.py --dataset-root $env:DATASET_ROOT --smoke-run $smokeRun
```

Đã tính độc lập trên file thật và so với `run_manifest.json` của smoke:

| File | Bytes | SHA-256 | So manifest |
|---|---:|---|---|
| weights/best.pt | 5,468,250 | `fb3a41c6494ffb40df02f2ef98e4bc5b1ef76cde4a78b6e90009ccfc09941899` | Khớp |
| weights/last.pt | 5,468,250 | `6830e4b5aec2db292b05f78e5a36de12a45b92e5462bf3c9832a2e9e96833699` | Khớp |

Smoke view_signature: `c2a1ffca7b49f12749e4d820baabd66955e1b95610a48224831837c39c2101f9`.
Hash khớp không bù được metadata thiếu: run có `git_commit=null`, `git_dirty=null`,
`best_epoch=null`, `train_time_s=null`; thư mục run không có `train.log`.
Các phép đo này chỉ nghiệm thu hành vi smoke ở B, không phải artifact B01/B02 của C.

### B.7. Kết luận

**FAIL — chưa đủ điều kiện chạy huấn luyện thật B01/B02.** Lỗi ảnh–nhãn augmentation,
đường pretrained, bảo vệ dataset, view tái sử dụng và provenance cần IMPLEMENTER sửa.
VERIFIER không sửa code impl; các test đỏ giữ làm bằng chứng và regression cho lượt kiểm tiếp theo.

Kiểm cuối: `git diff --check` sạch; diff hai file hợp đồng/WORKFLOW so với main vẫn rỗng;
diff `src`, `configs`, `requirements`, `scripts`, `notebooks`, `docs/RUN_GPU.md` so với
`step3/impl` rỗng. Chỉ commit tests và báo cáo này. Log/XML, JSON phép đo, checkpoint smoke
và file settings do lần import đầu sinh ra được giữ dưới `.cache/step3-verifier-b`, không commit.

## Giai đoạn C — chưa bắt đầu

Chờ B PASS và người dùng báo B01/B02 seed42 train xong. Chạy artifact lần lượt; bổ sung nghiệm thu
git provenance/args/data/log, reload thuần, đối chiếu 5 ảnh, AP50 độc lập 101 điểm với ngưỡng 0.02,
recipe/nguồn pretrained/tài nguyên/model card. Các test artifacts A là nền kiểm tra, chưa đủ thay C.
PASS artifact chỉ nghĩa là đủ điều kiện dùng Bước 4–6, không khẳng định chất lượng mô hình.
