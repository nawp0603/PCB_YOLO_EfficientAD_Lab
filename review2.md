# Báo cáo Phản biện Kỹ thuật plan.v2 (Review 2)

> **Người thực hiện:** Reviewer độc lập (Chuyên môn: Computer Vision công nghiệp, Thiết kế Benchmark ML & Hướng dẫn ĐATN)  
> **Tài liệu được review:** [plan.v2.md](file:///d:/FPTU/KLTN/PCB_YOLO_EfficientAD_Lab/plan.v2.md)  
> **Tài liệu đối chiếu:** [PLAN.md](file:///d:/FPTU/KLTN/PCB_YOLO_EfficientAD_Lab/PLAN.md), [Review.md](file:///d:/FPTU/KLTN/PCB_YOLO_EfficientAD_Lab/Review.md), [TRIAGE.md](file:///d:/FPTU/KLTN/PCB_YOLO_EfficientAD_Lab/docs/TRIAGE.md), [DECISIONS.md](file:///d:/FPTU/KLTN/PCB_YOLO_EfficientAD_Lab/docs/DECISIONS.md)  
> **Ngày lập:** 24/09/2026  
> **Trạng thái:** Báo cáo phản biện vòng 2 phục vụ hoàn thiện kế hoạch đồ án tốt nghiệp  

---

## 1. Tóm tắt 5 dòng

Bản `plan.v2.md` đã có bước tiến lớn khi dũng cảm cắt bỏ các yếu tố phi thực tế (loại bỏ `YOLO26`, bỏ metric pixel giả, chuyển EfficientAD sang tiling $256\times 256$, thêm interface và DET curve). Tuy nhiên, đứng ở góc độ hội đồng bảo vệ, bản kế hoạch này **vẫn tồn tại những kẽ hở phương pháp luận chí mạng có thể bị bẻ gãy khi vấn đáp**:
1. **Ảo giác khắc phục cỡ mẫu $N=2$:** Việc từ chối LOGO-CV là hợp lý về mặt chi phí, nhưng tuyên bố *"dùng bootstrap pair-level để bù đắp cho $N=2$"* là ngụy biện thống kê, vì bootstrap trên 2 bo không thể ngoại suy ra phương sai của toàn ngành sản xuất.
2. **"Tự ý" cắt bỏ 4/5 chiến lược Fusion đã cam kết:** Đề cương ban đầu nêu 5 chiến lược (OR, Evidence-based, Cascade, Weighted Avg, Learned), nhưng `plan.v2.md` thu hẹp gần như toàn bộ vào Cascade mà không có bảng thực nghiệm đối chứng tối thiểu (ngay cả luật OR cơ bản cũng không chạy song song).
3. **Mâu thuẫn toán học trong hàm chi phí (ECPB) và Tiling Stitching:** Công thức ECPB bất nhất giữa các mục ($50\times$ ở §7.2 vs $100\times$ ở §7.3); đồng thời cơ chế ghép các tile $256\times 256$ và khử box trùng ở biên (Tile boundary artifacts) hoàn toàn bị bỏ ngỏ.

---

## 2. Bảng phân tích vấn đề chi tiết của plan.v2

| # | Mục | Vấn đề phát hiện | Mức độ | Cách sửa cụ thể |
|---|---|---|:---:|---|
| 1 | **Thống kê** | **Ngụy biện dùng Bootstrap bù đắp cho $N=2$:** Bản v2 ghi: *"bootstrap pair-level bù đắp một phần cho N=2"* (§1.4, §7.6). Về mặt toán học, bootstrap 440 ảnh từ 2 bo mạch (`group44000`, `group92000`) chỉ đo phương sai chọn mẫu *nội tại 2 bo đó*, hoàn toàn không tạo thêm bất kỳ thông tin nào về các bo mạch khác. | **Critical** | Sửa lại văn bản: Thừa nhận thẳng thắn Bootstrap chỉ đo độ bất định nội suy (sampling variance), không đo được độ bất định chuyển miền (domain shift). Nếu không thể chạy 11-fold LOGO-CV, hãy chạy **3-Fold Group-KFold** (chia 11 groups thành 3 fold, mỗi fold chứa 3–4 groups) để có khoảng tin cậy thực tế giữa các thiết kế bo. |
| 2 | **Fusion** | **Cắt bỏ 4 chiến lược Fusion mà không có đối chứng:** Bối cảnh đề cương yêu cầu 5 chiến lược, nhưng `plan.v2.md` chỉ giữ Cascade. Hội đồng sẽ bắt bẻ: *"Tại sao không dùng OR hay Weighted Average? Bằng chứng nào chứng minh Cascade tối ưu hơn OR?"* *(Mục 1.2, 3.1)* | **Critical** | Không cần học mạng phức tạp, nhưng **bắt buộc phải đưa luật OR vào bảng benchmark chính** làm đối chứng baseline (chỉ tốn vài dòng code logic từ kết quả inference sẵn có). Với Weighted Average, chuẩn hóa min-max score rồi sweep ngưỡng. Trình bày bảng so sánh: Standalone $\rightarrow$ OR $\rightarrow$ Cascade. |
| 3 | **Benchmark** | **Mâu thuẫn nội bộ trong công thức ECPB:** Tại Mục 7.2 (dòng 386), trọng số ghi là $C_{FN} = 50 \times C_{FP}$. Nhưng tại Mục 7.3 (dòng 412), công thức lại ghi $100 \times FN + 1 \times FP$. Sự bất nhất này cho thấy số liệu chưa được chuẩn hóa. | **Major** | Thống nhất 1 công thức duy nhất: Định nghĩa rõ $ECPB = \frac{C_{FN} \cdot FN + C_{FP} \cdot FP}{N_{total}}$ với tỷ lệ giả định 1% Defect, 99% Good. Khóa cứng một cặp trọng số công nghiệp tiêu chuẩn: $C_{FN} = 100$, $C_{FP} = 1$ (tương đương tỷ lệ $100:1$ chuẩn IPC về tổn thất lọt lỗi so với chặn nhầm). |
| 4 | **Protocol** | **Hiệu chuẩn $\tau_{ad}$ độc lập gây lệch phân phối đầu vào Cascade:** Mục 5 (Bước 5) sửa thành: *"Hiệu chuẩn $\tau_{ad}$ trên toàn bộ defect calibration"*. Tuy nhiên, tập defect mà YOLO bỏ sót (routed sang AD) thường là vi khuyết tật khó. Việc lấy ngưỡng từ toàn bộ defect (gồm cả các lỗi to, rõ mà YOLO bắt rất dễ) sẽ làm $\tau_{ad}$ bị lệch (quá cao hoặc quá thấp so với miền lỗi khó). | **Major** | Áp dụng quy trình tối ưu **2-D Grid Search có điều kiện**: Quét lưới đồng thời $(\tau_{yolo}, \tau_{ad})$ trên không gian $[0.1, 0.9] \times [\mu_{ad}, \mu_{ad} + 3\sigma_{ad}]$ trực tiếp trên pipeline cascade của tập calibration. Tiêu chí chọn: Cặp $(\tau_{yolo}^*, \tau_{ad}^*)$ tối đa hóa Recall tại $\text{FRR} \le 5\%$. |
| 5 | **Kiến trúc phần mềm** | **Thiếu cơ chế ghép mảnh (Stitching) và xử lý lỗi cắt ngang biên:** Chuyển sang tiling $256\times 256$ (§2, §4) giải quyết được mất lỗi nhỏ, nhưng phát sinh lỗi biên: một khuyết tật hở mạch hay gai đồng bị cắt làm đôi ở đường chia tile sẽ bị suy giảm diện tích hoặc biến dạng. Bản v2 hoàn toàn thiếu giải thuật ghép anomaly map và NMS xuyên tile. | **Major** | Bổ sung module `TileManager` vào `src/pcb_lab/inference/`: (1) Ghép Anomaly Map dùng kỹ thuật **Gaussian/Linear Blending** (feathering) ở vùng chồng lấn (overlap 32px) để triệt tiêu đường viền sắc nhọn; (2) Chuyển đổi tọa độ box từ tile về ảnh gốc trước khi chạy **Global Soft-NMS**. |
| 6 | **Phần cứng & Thực thi** | **Lộ trình 4–6 tuần bất khả thi trên phần cứng hiện tại:** `status.json` xác nhận máy chỉ có CPU/Intel Arc, không có CUDA. Train EfficientAD-S (70 epoch $\times$ 895 ảnh $\times$ 4 tile = 250.000 step) trên CPU sẽ mất hàng tuần. Kế hoạch tuần 1–2 đặt mục tiêu train cả 2 mô hình là phi thực tế nếu không có GPU rời. *(Mục 1.4, 12)* | **Major** | Đặt điều kiện tiên quyết (Gate 0) trước Tuần 1: Thiết lập môi trường Google Colab Pro / Kaggle T4 $\times$ 2 / Thuê máy GPU RTX 3060/4090. Nếu chưa có môi trường GPU sẵn sàng với CUDA 12.x, không được bắt đầu đếm thời gian 4–6 tuần. |
| 7 | **Định vị lỗi** | **Point-in-Box Recall thiếu ràng buộc kích thước:** Mục 7.7 định nghĩa: *"Nếu điểm cực đại của heatmap rơi vào trong Ground Truth BBox thì tính là thành công"*. Nếu EfficientAD sinh ra một anomaly blob khổng lồ bao phủ nửa bo mạch (do lóa sáng), điểm cực đại vẫn nằm trong bbox, dẫn đến báo cáo định vị "đúng" nhưng vô giá trị thực tế. | **Minor** | Bổ sung ràng buộc diện tích cho Point-in-Box: Điểm cực đại rơi vào trong BBox **VÀ** diện tích của Connected Component tương ứng không được vượt quá $3\times$ diện tích Ground Truth BBox ($Area_{blob} \le 3 \cdot Area_{bbox}$). |
| 8 | **Robustness Suite** | **Thiếu sai số căn chỉnh quang học (Registration/Alignment Jitter):** Stress test bổ sung lóa sáng, rung mờ, mất nét (§9) là tốt, nhưng thiếu biến dạng quan trọng nhất của Anomaly Detection: **Lệch pha vị trí**. Template matching trong AOI luôn có dung sai 1–3 pixel. | **Minor** | Bổ sung hàm biến dạng `subpixel_shift_and_jitter(dx=[-2, 2], dy=[-2, 2], dtheta=[-0.5, 0.5])` vào bộ stress test. Đánh giá xem EfficientAD có bị nổ False Positive dọc theo các mép đường mạch khi ảnh bị lệch nhẹ hay không. |

---

## 3. Kết quả kiểm chứng các phát biểu dữ kiện

1. **Về việc DeepPCB cắt từ số board hạn chế:**
   - **Xác nhận:** **ĐÚNG**. Tác giả Tang et al. (2019) quét các tấm panel lớn bằng camera CCD công nghiệp, sau đó crop thành 1.500 cặp ảnh 640×640 trong 11 thư mục `group`. Việc cô lập theo group là bắt buộc để tránh rò rỉ layout mạch.
2. **Về việc DeepPCB chỉ có Bbox, metric pixel chỉ là xấp xỉ:**
   - **Xác nhận:** **ĐÚNG**. Annotation của dataset là các file text tọa độ 4 điểm góc. Quyết định của `plan.v2.md` (không công bố Pixel AUROC/Dice diện tích và thay bằng Point-in-Box có kiểm soát) là hoàn toàn chuẩn xác về mặt khoa học.
3. **Về việc Anomalib hỗ trợ EfficientAD và PatchCore:**
   - **Xác nhận:** **ĐÚNG**. Cả hai mô hình đều có mặt chính thức trong thư viện Anomalib (`anomalib.models.image.efficient_ad` và `anomalib.models.image.patchcore`).
4. **Về "YOLO26":**
   - **Xác nhận:** **HOÀN TOÀN SAI / KHÔNG TỒN TẠI**. Triage trong v2 vẫn để trạng thái `VERIFY (V1)` vì máy bị chặn mạng (403). Tuy nhiên, với tư cách reviewer độc lập kiểm chứng hệ sinh thái Ultralytics quốc tế: **Không có bất kỳ phiên bản nào tên là YOLO26**. Phiên bản mới nhất tính đến cuối năm 2024 là YOLO11. Việc tiếp tục giữ B05 dưới dạng "placeholder chờ verify" là thừa thãi và thể hiện sự thiếu tự tin; cần **xóa bỏ vĩnh viễn** mục B05 khỏi tài liệu.

---

## 4. 5 câu hỏi hội đồng có thể chất vấn và gợi ý trả lời cho plan.v2

### Câu hỏi 1: Về việc thu hẹp từ 5 chiến lược Fusion xuống chỉ còn Cascade
> *"Trong đề cương các em đăng ký thử nghiệm 5 chiến lược kết hợp (OR, Evidence-based, Cascade, Weighted Average, Logistic Regression). Nhưng trong bản kế hoạch v2, các em lại bỏ hết và chỉ làm mỗi Cascade. Có phải các em đang tự ý cắt xén khối lượng công việc? Làm sao khẳng định Cascade vượt trội hơn luật OR đơn giản?"*

*   **Hiện trạng plan.v2:** Đã lược bỏ và chỉ chạy Cascade để tránh overfit, nhưng chưa có số liệu hay bảng thực nghiệm chứng minh.
*   **Gợi ý trả lời cho sinh viên:**
    1. *Lý giải học thuật:* Mạng học kết hợp (Logistic Regression / Evidence-based) đòi hỏi tập dữ liệu validation độc lập đủ lớn. Với DeepPCB chỉ có 153 ảnh defect trong tập development, việc huấn luyện một meta-classifier rất dễ bị overfit vào 2 layout mạch cụ thể.
    2. *Cam kết thực nghiệm đối chứng:* Nhóm không loại bỏ hoàn toàn mà tái cấu trúc: **Lấy luật OR làm baseline đối chứng bắt buộc** cho Cascade. Trong luật OR, cả 2 nhánh chạy song song khiến độ trễ tăng gấp đôi và FRR tăng cộng dồn. Nhóm sẽ chứng minh Cascade đạt Recall tương đương OR nhưng giảm được 60–80% thời gian suy luận (nhờ cơ chế early exit ở tầng 1) và chặn được các ca báo nhầm của tầng 2.

### Câu hỏi 2: Về tính hợp lệ của việc Bootstrap trên tập Test chỉ có 2 nhóm nguồn
> *"Các em thừa nhận tập test chỉ có 2 nhóm bo mạch ($N=2$), nhưng lại tuyên bố 'dùng bootstrap pair-level để bù đắp'. Về mặt xác suất thống kê, lấy mẫu lại có hoàn lại từ 2 tấm bo thì làm sao suy ra được độ tin cậy trên hàng trăm tấm bo khác ngoài thị trường?"*

*   **Hiện trạng plan.v2:** Nhận diện được $N=2$ nhưng diễn đạt gây hiểu lầm là Bootstrap giải quyết được vấn đề.
*   **Gợi ý trả lời cho sinh viên:**
    1. *Đính chính khái niệm:* Nhóm không dùng Bootstrap để ngoại suy ra các bo mạch mới. Bootstrap pair-level chỉ nhằm mục đích tính **khoảng tin cậy nội suy (Internal Confidence Interval)** đối với các biến thể khuyết tật trên chính 2 thiết kế mạch đó, tránh việc kết luận bị chi phối bởi một vài cặp ảnh cá biệt.
    2. *Định vị giới hạn đồ án:* Nhóm khẳng định rõ trong luận văn: Đồ án giải quyết bài toán kiểm định cho **dây chuyền chuyên biệt (Dedicated Production Line)**, nơi hệ thống được thiết lập để kiểm tra lặp lại một số mẫu bo mạch cố định đã nạp template, không tuyên bố giải quyết bài toán kiểm định vạn năng (Universal Zero-shot AOI).

### Câu hỏi 3: Về sự cố nổ False Alarm khi áp dụng vào thực tế nhà máy (ECPB)
> *"Tại sao các em phải đưa chỉ số ECPB với tỷ lệ lỗi giả định 1% vào bảng benchmark? Nếu FRR của hệ thống là 4% trên tập test, các em có biết nhà máy sẽ phải đối mặt với hậu quả gì không?"*

*   **Hiện trạng plan.v2:** Đã bổ sung ECPB nhưng công thức còn mâu thuẫn giữa 50× và 100×.
*   **Gợi ý trả lời cho sinh viên:**
    1. *Ý nghĩa của ECPB:* Trong phòng thí nghiệm với tập test 50/50, độ chính xác (Accuracy) 95% trông rất đẹp. Nhưng ở dây chuyền thực tế với tỷ lệ lỗi 1%, nếu hệ thống có $\text{FRR} = 4\%$, thì cứ 1.000 bo chạy qua sẽ có 10 bo lỗi thật (bắt được 9.5 bo) nhưng lại có tới 40 bo tốt bị chặn nhầm! Tỷ lệ báo giả trên trạm kiểm định thực tế lên tới $\approx 80\%$.
    2. *Ứng dụng công thức:* Nhóm đưa ECPB với hàm phạt $100 \times FN + 1 \times FP$ để phản ánh đúng thực tế kinh tế: Thà tốn công nhân kiểm tra lại 40 bo báo nhầm (chi phí thấp) còn hơn để lọt 1 bo lỗi ra thị trường (phạt nặng, thu hồi). ECPB là thước đo thực tế giúp chọn điểm cắt ngưỡng (Operating Point) thay vì nhìn vào F1-score ảo.

### Câu hỏi 4: Về vấn đề biên cắt khi chia ô (Tile Boundary Artifacts)
> *"Khi các em cắt ảnh 640×640 thành các ô $256\times 256$, nếu một lỗi đứt mạch (open circuit) nằm ngay trên đường biên phân chia giữa 2 ô thì sao? Mô hình EfficientAD có bị nhận diện nhầm đường cắt là dị thường không, và làm sao ghép 4 bản đồ nhiệt lại mà không bị vệt sọc ở mép ô?"*

*   **Hiện trạng plan.v2:** Đã chuyển sang tile 256 native nhưng chưa mô tả thuật toán ghép và xử lý biên.
*   **Gợi ý trả lời cho sinh viên:**
    1. *Cơ chế Overlap:* Nhóm không cắt rời rạc mà dùng kỹ thuật cửa sổ trượt có vùng chồng lấn **overlap 32 pixel**. Bất kỳ khuyết tật nào nằm ở biên của ô này sẽ nằm trọn vẹn ở phần trung tâm an toàn của ô liền kề.
    2. *Ghép mượt bản đồ nhiệt (Blending):* Khi tái tạo bản đồ nhiệt toàn ảnh, vùng chồng lấn không dùng phép cộng thô mà sử dụng hàm trọng số khoảng cách (Linear/Gaussian Distance Weighting) để làm mờ dần biên của từng tile. Đối với các bounding box trích xuất từ tile, nhóm chuyển đổi ngược về hệ tọa độ gốc $640\times 640$ rồi áp dụng Non-Maximum Suppression (NMS) xuyên tile để gộp các box bị nhận diện lặp.

### Câu hỏi 5: Về hiện tượng thắt cổ chai phần cứng và tính khả thi triển khai
> *"Hồ sơ hiện tại của các em ghi máy chạy PyTorch CPU, không có CUDA. Với khối lượng vừa train YOLO11 vừa train EfficientAD (tiling 4 ô/ảnh), các em định hoàn thành đồ án trong 4 tuần bằng cách nào?"*

*   **Hiện trạng plan.v2:** Nhận thức được máy là Intel Arc/CPU nhưng vẫn giữ lịch trình Tuần 1–2 bắt đầu train.
*   **Gợi ý trả lời cho sinh viên:**
    1. *Phân tách môi trường:* Môi trường máy local (Intel Arc CPU) chỉ dùng cho việc phát triển khung code, kiểm toán dữ liệu (Bước 0–2), viết unit test và dựng giao diện Streamlit.
    2. *Hạ tầng huấn luyện:* 100% tác vụ train nặng và sweep ngưỡng (Bước 3–7) được thực hiện trên môi trường đám mây độc lập có tăng tốc phần cứng (Google Colab T4 GPU hoặc Kaggle T4$\times$2 GPU) thông qua các runner notebook tự động đã được module hóa. Sau khi train, chỉ tải artifact/checkpoint về máy local để chạy suy luận (Inference) và benchmark.

---

## 5. Checklist thay đổi đề xuất cho plan.v2 $\rightarrow$ plan.v3

Xếp theo thứ tự ưu tiên hành động:

### Nhóm P0: Sửa chữa văn bản & Phương pháp luận (Hoàn thành ngay)
- [ ] **P0.1 - Xóa vĩnh viễn placeholder `YOLO26`:** Xóa bỏ hoàn toàn mã B05 và ghi chú `VERIFY (V1)` khỏi toàn bộ văn bản. Khóa ma trận detector chỉ gồm: `YOLO11n` (chính) và `YOLO11s` (đối chứng dung lượng).
- [ ] **P0.2 - Đưa luật OR trở lại bảng Benchmark chính:** Bổ sung cấu hình `H00 (YOLO11n + EfficientAD-S [OR])` vào bảng so sánh cạnh `H01 (Cascade)` để bảo vệ tính toàn vẹn của việc so sánh đa chiến lược fusion.
- [ ] **P0.3 - Chuẩn hóa thống nhất công thức ECPB:** Đồng bộ hóa ở cả §7.2 và §7.3 theo công thức:
  $$\text{ECPB} = \frac{100 \cdot FN_{\text{sim}} + 1 \cdot FP_{\text{sim}}}{N_{\text{sim}}}$$
  với $N_{\text{sim}} = 10.000$ bo mạch giả định ($100$ Defect, $9.900$ Good).
- [ ] **P0.4 - Hiệu chỉnh phát biểu về Bootstrap:** Bỏ câu khẳng định bootstrap "bù đắp cho $N=2$". Ghi rõ: *"Bootstrap phản ánh độ biến thiên mẫu nội tại trong 2 bo test; kết luận tổng quát hóa ngoại suy sang bo mới được coi là giới hạn nghiên cứu của đồ án"*.

### Nhóm P1: Bổ sung Thuật toán & Mã nguồn (Hoàn thành Tuần 1–2)
- [ ] **P1.1 - Đặc tả chi tiết thuật toán Tiling & Stitching (§4.4 / §7.7):**
  - Khai báo kích thước tile: $256\times 256$, stride: $224$ (overlap 32 px).
  - Khóa thuật toán tái tạo heatmap: Trọng số hóa khoảng cách (Feathering Blending) tại vùng overlap.
  - Khóa thuật toán gộp box: Ánh xạ tọa độ tile về ảnh gốc $\rightarrow$ Global NMS (IoU threshold = 0.45).
- [ ] **P1.2 - Tinh chỉnh tiêu chí Point-in-Box có kiểm soát diện tích (§7.7):** Bổ sung điều kiện lọc nhiễu: Một phát hiện vị trí được tính là True Positive khi:
  $$\text{argmax}(Map) \in BBox_{\text{GT}} \quad \text{VÀ} \quad Area_{\text{component}} \le 3 \cdot Area_{\text{GT}}$$
- [ ] **P1.3 - Thiết lập Gate 0 về hạ tầng GPU trong Lộ trình (§12):** Ghi rõ điều kiện bắt buộc trước khi bước vào Tuần 2: Phải có kết nối và chạy thử nghiệm thành công một smoke-train 3 epoch trên Google Colab / Kaggle GPU.

### Nhóm P2: Tối ưu & Mở rộng (Tuần 3 trở đi)
- [ ] **P2.1 - Bổ sung biến dạng lệch pha (Alignment Jitter) vào Stress Suite (§9):** Viết script tạo nhiễu dịch chuyển tịnh tiến $\pm 1$ đến $\pm 2$ pixel trên ảnh test để kiểm chứng độ nhạy cảm của EfficientAD với sai số cơ khí.
- [ ] **P2.2 - Thực hiện 3-Fold Group-KFold (nếu còn tài nguyên GPU):** Thay vì 11-fold, chia 11 nhóm nguồn thành 3 fold để kiểm định chéo, lấy khoảng tin cậy thực sự có ý nghĩa thống kê giữa các nhóm bo.

---

## 6. Prompt chỉ định cho AI Agent để cập nhật plan.v2.md $\rightarrow$ plan.v3.md

Dưới đây là prompt hoàn chỉnh sẵn sàng để copy-paste gửi cho AI Agent nhằm thực hiện việc cập nhật và nâng cấp tài liệu kế hoạch:

````markdown
Bạn là một kỹ sư phần mềm kiêm nhà nghiên cứu Computer Vision công nghiệp giàu kinh nghiệm. 
Hãy đọc kỹ tài liệu kế hoạch hiện tại `plan.v2.md` và tài liệu phản biện `review2.md` trong thư mục `D:\FPTU\KLTN\PCB_YOLO_EfficientAD_Lab`.
Nhiệm vụ của bạn là nâng cấp `plan.v2.md` thành `plan.v3.md` (hoặc cập nhật trực tiếp `plan.v2.md` nếu được yêu cầu ghi đè), giải quyết triệt để tất cả các vấn đề thuộc nhóm P0 và P1 trong `review2.md`.

### CÁC NGUYÊN TẮC CỐT LÕI BẮT BUỘC TUÂN THỦ:
1. **Giữ vững cấu trúc chuẩn và tính nhất quán:** Kế thừa văn phong kỹ thuật chính xác, ngắn gọn, trung thực của `plan.v2.md`. Giữ lại bảng đối chiếu SHA-256, taxonomy 6 lớp lỗi, và 4 phân vùng canonical của DatasetVer4_Public.
2. **Tuyệt đối không ảo giác về mô hình (Zero Hallucination):** Xóa vĩnh viễn mọi đề cập đến `YOLO26` và ghi chú `VERIFY (V1)`. Bảng ma trận detector chỉ gồm: `YOLO11n` (baseline chính) và `YOLO11s` (đối chứng dung lượng).
3. **Đưa đối chứng Luật OR (H00) trở lại:** Thêm cấu hình `H00 (YOLO11n + EfficientAD-S theo luật OR)` vào bảng ma trận thí nghiệm (§7.1) và bảng benchmark ảnh (§7.2) ngay trước `H01 (Cascade)`. Làm rõ ý nghĩa: H00 là baseline kết hợp không điều kiện, H01 là hybrid có định tuyến để tối ưu độ trễ và giảm báo nhầm.
4. **Chuẩn hóa công thức ECPB:** Đồng bộ hóa ở cả §7.2 và §7.3. Định nghĩa rõ:
   $$ECPB = \frac{100 \cdot FN_{\text{sim}} + 1 \cdot FP_{\text{sim}}}{N_{\text{sim}}}$$
   với $N_{\text{sim}} = 10.000$ bo mạch mô phỏng theo tỷ lệ nhà máy (1% Defect, 99% Good). Giải thích rõ tỷ lệ phạt 100:1 theo thực tế công nghiệp SMT/AOI.
5. **Hiệu chỉnh phát biểu về $N=2$ và Bootstrap:** Tại §1.4 và §7.6, bỏ ngay câu nói "bootstrap bù đắp cho N=2". Ghi nhận trung thực: *"Tập test giới hạn ở 2 nhóm bo (N=2), bootstrap pair-level chỉ phản ánh độ bất định nội suy (sampling variance) trên 2 thiết kế bo này, không đại diện cho phương sai chuyển miền sang bo mạch bất kỳ. Kết luận đồ án giới hạn trong phạm vi dây chuyền chuyên biệt (dedicated line)."*
6. **Đặc tả chi tiết Tiling, Stitching và Blending (§4.4 / §7.7):**
   - Tile size: $256\times 256$, stride: $224$ (overlap 32 px).
   - Tái tạo Anomaly Map toàn ảnh: Áp dụng Linear Blending (Feathering) tại các vùng biên chồng lấn để triệt tiêu hiệu ứng sọc mép ô.
   - Gộp Bounding Box: Ánh xạ tọa độ từ tile về ảnh gốc $640\times 640$ và áp dụng Global NMS (IoU threshold = 0.45).
7. **Ràng buộc diện tích cho Point-in-Box Recall (§7.7):** Một phát hiện vị trí được tính là thành công khi:
   $$\text{argmax}(Map) \in BBox_{\text{GT}} \quad \text{VÀ} \quad Area_{\text{component}} \le 3 \cdot Area_{\text{GT}}$$
8. **Bổ sung Alignment Jitter vào Robustness Suite (§9):** Thêm biến dạng lệch pha dịch chuyển $\pm 1$ đến $\pm 2$ pixel để kiểm tra tính ổn định của EfficientAD trước rung sai cơ khí.
9. **Bổ sung Cổng quyết định phần cứng GPU (Gate 0) vào Lộ trình (§12):** Ghi rõ điều kiện tiên quyết trước Tuần 1: Thiết lập và xác nhận kết nối GPU Google Colab / Kaggle với CUDA 12.x; máy local chỉ dùng cho CLI/UI/EDA.

### ĐẦU RA YÊU CẦU:
- Tạo file `plan.v3.md` hoàn chỉnh, đánh dấu phiên bản ngày 24/09/2026.
- Tạo bản tóm tắt các điểm đã sửa chữa ở đầu file hoặc báo cáo phản hồi ngắn gọn.
````
