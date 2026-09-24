# Đánh giá & Phản biện Kỹ thuật Kế hoạch PCB Hybrid YOLO + EfficientAD

> **Người thực hiện review:** Reviewer độc lập (Chuyên môn: Computer Vision công nghiệp, Thiết kế Benchmark ML & Hướng dẫn ĐATN)  
> **Tài liệu được review:** [PLAN.md](file:///d:/FPTU/KLTN/PCB_YOLO_EfficientAD_Lab/PLAN.md)  
> **Ngày lập:** 24/09/2026  
> **Trạng thái:** Báo cáo phản biện phục vụ hoàn thiện kế hoạch đồ án tốt nghiệp  

---

## 1. Tóm tắt 5 dòng

Kế hoạch thể hiện tư duy kiểm soát thực nghiệm và tính cẩn trọng cao (chặn rò rỉ dữ liệu, từ chối metric pixel giả, phân định ranh giới bài toán), vượt trội so với các đồ án thông thường. Tuy nhiên, bản kế hoạch vẫn mang nặng tính **học thuật trên tập ảnh tĩnh sạch** hơn là bài toán kiểm định công nghiệp (AOI). Ba rủi ro lớn nhất đe dọa sự thành bại khi ra hội đồng gồm:
1. **Thiếu mẫu đại diện cấp bo mạch (Board-level variance):** 440 ảnh test thực chất chỉ đến từ **2 tấm bo mạch lớn (2 source groups)**, khiến độ tự do thống kê về khả năng tổng quát hóa mạch mới là $N=2$.
2. **Hiệu ứng thắt cổ chai của Cascade (Cascaded Masking & Asymmetric Bias):** Nếu YOLO ở tầng 1 báo nhầm (FP), EfficientAD bị khóa hoàn toàn; đồng thời AD bị ép phân giải 256×256 sẽ làm tiêu biến các vi khuyết tật (pin-hole, mouse bite).
3. **Ảo tưởng về phạm vi và khối lượng thực hiện:** Mục tiêu thử nghiệm quá dàn trải (thử nhiều họ AD, nhắc đến model không có thật như YOLO26, đòi hỏi đo đa backend trên phần cứng chưa ổn định) trong khi thời gian và tài nguyên GPU của sinh viên có hạn.

---

## 2. Bảng phân tích vấn đề chi tiết

| # | Mục | Vấn đề phát hiện | Mức độ | Cách sửa cụ thể |
|---|---|---|:---:|---|
| 1 | **Protocol** | **Test set thực chất chỉ có 2 bo mạch ($N=2$):** Dù có 440 ảnh test, chúng chỉ nằm trong 2 nhóm nguồn (`group44000`, `group92000`). Mọi kết luận về khả năng "tổng quát hóa trên bo chưa thấy" là thiếu cơ sở thống kê. *(Mục 1.2, 7.6)* | **Critical** | Báo cáo rõ ràng: "Đánh giá giới hạn trên 2 thiết kế bo mạch cụ thể". Thực hiện **Leave-One-Group-Out Cross-Validation (LOGO-CV)** trên 11 groups thay vì 1 split cố định để đo phương sai thực giữa các thiết kế mạch. |
| 2 | **Protocol** | **Hiệu chuẩn $\tau_{ad}$ trên tập lỗi quá nhỏ ở tầng 2:** Khi YOLO đạt recall cao trên tập calibration, số ảnh defect lọt qua tầng 1 vào EfficientAD chỉ còn khoảng 10–20 ảnh. Dùng lượng mẫu này để chọn $\tau_{ad}$ sẽ gây overfit ngưỡng trầm trọng. *(Mục 5 - Bước 5)* | **Critical** | Chạy toàn bộ ảnh defect của calibration qua EfficientAD độc lập để lập phân phối score, sau đó mới giải bài toán tối ưu đồng thời cặp $(\tau_{yolo}, \tau_{ad})$ trên đường cong Pareto (Recall vs FRR) của toàn bộ pipeline. |
| 3 | **Model & Data** | **Tồn tại mô hình không có thật ("YOLO26"):** Kế hoạch liệt kê B05 là `YOLO26n` kèm link docs.ultralytics.com/models/yolo26. Đây là thực thể giả lập/ảo giác (hallucination). *(Mục 7.1, 13)* | **Critical** | Xóa bỏ ngay lập tức `YOLO26` khỏi ma trận thí nghiệm. Thay thế bằng các baseline chuẩn mực có sẵn trong Ultralytics: **YOLOv8n** hoặc **RT-DETR-r18** (đối chứng kiến trúc Transformer). |
| 4 | **Benchmark** | **Trùng lặp metric và thiếu bối cảnh tỷ lệ lỗi công nghiệp:** Accuracy và Balanced Accuracy trên test set cân bằng (50/50) là giống hệt nhau về toán học. Đồng thời, tỷ lệ 50/50 che giấu hoàn toàn hiện tượng sụp đổ Precision khi áp dụng vào nhà máy (nơi tỷ lệ lỗi thực tế chỉ 0.1%–1%). *(Mục 7.2, 7.3)* | **Major** | (1) Bỏ bớt metric trùng: giữ Balanced Acc, Defect Recall, FRR, F1. (2) Bổ sung **Prevalence-Adjusted Precision** hoặc **Expected Cost per Board (ECPB)** mô phỏng ở tỷ lệ lỗi công nghiệp (1% Defect, 99% Good) với trọng số chi phí $C_{FN} = 50 \times C_{FP}$. |
| 5 | **Fusion** | **Quy tắc Cascade chặn đứng khả năng phát hiện đa lỗi:** Nếu YOLO phát hiện 1 lỗi giả hoặc lỗi thứ nhất với confidence $\ge \tau_{yolo}$, EfficientAD không bao giờ được gọi. Nếu trên bo có 1 lỗi lạ khác nằm ngoài 6 lớp đã học, hệ thống sẽ bỏ qua hoàn toàn. *(Mục 3.1, 3.2)* | **Major** | Thiết kế lại cơ chế routing: Chế độ kiểm định tiêu chuẩn công nghiệp phải chạy EfficientAD toàn ảnh (hoặc vùng ngoài các box YOLO) ở chế độ ngầm để gán cờ `Anomaly Flag` nếu anomaly score vùng ngoài box vượt ngưỡng an toàn. |
| 6 | **Fusion** | **Thiếu so sánh đẳng điều kiện (Iso-FRR Comparison):** Khẳng định hybrid tốt hơn chỉ vì Recall tăng là sai lầm kinh điển, vì luật OR luôn tăng Recall kèm tăng FP. So sánh chỉ có giá trị khi đặt hai mô hình tại cùng một mức báo sai. *(Mục 7.4)* | **Major** | Vẽ đường cong **DET (Detection Error Tradeoff: FRR vs Miss Rate)** cho cả 3 cấu hình: YOLO-only, EfficientAD-only, và Cascade. So sánh Recall của Cascade tại các mốc cố định: $\text{FRR} = 1\%, 3\%, 5\%$. |
| 7 | **Dữ liệu / Model** | **Co ảnh 256×256 làm triệt tiêu vi khuyết tật trên EfficientAD:** DeepPCB gốc là 640×640. Khi nén về 256×256, diện tích giảm 6.25 lần. Một lỗ kim (`pin_hole`) hay khuyết mạch (`mouse_bite`) rộng 3–5 px sẽ biến mất hoàn toàn sau phép nội suy bilinear. *(Mục 5 - Bước 4)* | **Major** | Không nén toàn ảnh về 256×256. Bắt buộc dùng chiến lược **Tiling cố định**: Cắt ảnh 640×640 thành 4 tile kích thước $320\times 320$ hoặc $256\times 256$ có chồng lấn (overlap 32 px), giữ nguyên độ phân giải quang học trước khi đưa vào EfficientAD. |
| 8 | **Định vị lỗi** | **Chưa định nghĩa thuật toán chuyển Heatmap thành Bbox:** Kế hoạch chỉ nói "class-agnostic localization nằm ở bảng phụ" nhưng không nêu thuật toán chuyển đổi, khiến việc đánh giá định vị của AD trở nên bất khả thi. *(Mục 7.2)* | **Major** | Khóa thuật toán cụ thể: $Map \xrightarrow{\text{threshold } \tau_{pixel}} \text{Binary Mask} \xrightarrow{\text{Morph Open(k=3)}} \text{Connected Components} \xrightarrow{\text{Area} > A_{min}} \text{Bounding Rect}$. Công bố rõ $A_{min}$ và $\tau_{pixel}$. |
| 9 | **Kiến trúc mã nguồn** | **Thiếu trừu tượng hóa nguồn ảnh (`ImageSource`):** Mã nguồn chỉ phụ thuộc vào file ảnh tĩnh trên ổ đĩa và `st.file_uploader`. Khi chuyển sang camera hoặc video stream công nghiệp sau này sẽ phải đập đi xây lại toàn bộ pipeline tiền xử lý. *(Mục 4.1, 8)* | **Minor** | Xây dựng Interface trừu tượng `ImageSource` với các implementation: `FileImageSource`, `DirectoryWatcherSource`, và `MockFrameGrabber`. Tách lõi suy luận thành module độc lập không dính dáng tới Streamlit (`InferenceService`). |
| 10 | **Thực nghiệm** | **Bộ stress test camera thiếu các biến dạng vật lý cốt lõi:** Thử nghiệm xoay, làm mờ, đổi sáng đơn giản không phản ánh được hiện thực quang học của máy AOI. *(Mục 9)* | **Minor** | Bổ sung 3 biến dạng đặc thù của kiểm định mạch: (1) **Lóa phản xạ bề mặt thiếc/đồng** (Specular highlight clipping), (2) **Rung mờ lệch trục** (Directional motion blur do băng chuyền), và (3) **Mất nét biên** (Defocus blur do bo mạch cong vênh). |

---

## 3. Kết quả kiểm chứng các phát biểu dữ kiện

Sau đây là kết quả đối chiếu độc lập với tài liệu gốc của tập dữ liệu và các framework liên quan:

1. **Phát biểu 1: "DeepPCB được cắt từ số board hạn chế nên dễ leak giữa train/test"**
   - **Kết luận:** **ĐÚNG (Đã được xác minh)**.
   - **Căn cứ & Chi tiết:** Theo bài báo gốc *“Online PCB Defect Detector on a New PCB Defect Dataset”* (Tang et al., 2019, arXiv:1902.06197) và repository chính thức `tangsanli5201/DeepPCB`, 1.500 cặp ảnh (kích thước 640×640) được cắt ra từ các ảnh quét tuyến tính siêu phân giải (~16.000×16.000 px) của một số lượng rất hạn chế các tấm panel bo mạch thực tế. Toàn bộ dataset được chia theo các thư mục `groupXXXXX` (chỉ có 11–12 group). Các ảnh con trong cùng một group chia sẻ chung đặc tính layout, bề rộng đường mạch, chất lượng phủ mạ và ánh sáng. Do đó, nếu chia ngẫu nhiên (random split) ở cấp ảnh, hiện tượng rò rỉ layout (layout leakage) giữa train và test là chắc chắn xảy ra. Việc phân chia cách ly theo `source_group` như kế hoạch đang làm là bắt buộc.
2. **Phát biểu 2: "DeepPCB chỉ có bbox nên pixel-level metric chỉ xấp xỉ"**
   - **Kết luận:** **ĐÚNG (Đã được xác minh)**.
   - **Căn cứ & Chi tiết:** Dữ liệu nhãn gốc của DeepPCB chỉ lưu tọa độ hộp chữ nhật: `x1, y1, x2, y2, type`. Hoàn toàn không có mặt nạ phân đoạn (segmentation polygon/mask) cho từng pixel khuyết tật. Nếu lấy bounding box này làm ground truth mask để tính Pixel-AUROC hay Dice Score cho EfficientAD thì 70%–90% diện tích bên trong hộp thực chất là đường mạch bình thường và nền FR-4, biến phép đo pixel thành một ước lượng sai lệch lớn. Quyết định không công bố Pixel-AUROC trong kế hoạch là hoàn toàn chính xác về mặt khoa học.
3. **Phát biểu 3: "Anomalib hỗ trợ EfficientAD và PatchCore"**
   - **Kết luận:** **ĐÚNG (Đã được xác minh)**.
   - **Căn cứ & Chi tiết:** Thư viện `anomalib` của Intel/OpenVINO (các bản phát hành chính thức từ 0.x, 1.x đến 2.x) hỗ trợ trực tiếp cả `EfficientAd` (triển khai theo Batzner et al., WACV 2024 với kiến trúc Teacher-Student-Autoencoder) và `Patchcore` (triển khai theo Roth et al., CVPR 2022 với cơ chế trích xuất đặc trưng Neighborhood Aware và nén Memory Bank bằng Coreset Subsampling).
4. **Phát biểu 4: "YOLO26 là một tùy chọn detector mở rộng từ Ultralytics"**
   - **Kết luận:** **SAI (Sai sót nghiêm trọng trong tài liệu)**.
   - **Căn cứ & Chi tiết:** Hệ sinh thái Ultralytics và cộng đồng Computer Vision quốc tế tính đến thời điểm hiện tại hoàn toàn không có phiên bản "YOLO26". Các họ mô hình chính thức gồm có YOLOv5, YOLOv8 và gần nhất là YOLO11 (phát hành quý 3/2024). Đường dẫn `https://docs.ultralytics.com/models/yolo26` trong kế hoạch không tồn tại.

---

## 4. 5 câu hỏi hội đồng có thể chất vấn và gợi ý trả lời

### Câu hỏi 1: Về giá trị thực tế của cấu trúc Hybrid
> *"Tại sao phải làm phức tạp hệ thống bằng cách ghép thêm EfficientAD khi YOLO11n đã có thể học và nhận diện cả 6 loại lỗi với độ chính xác cao? Về mặt kỹ thuật và vận hành, EfficientAD mang lại giá trị gia tăng gì tương xứng với việc hệ thống phải gánh thêm độ trễ và tài nguyên bộ nhớ?"*

*   **Hiện trạng kế hoạch:** Đã chuẩn bị sẵn các metric chênh lệch (`rescued_defects`, `delta_latency`), nhưng chưa có số liệu thực tế để chứng minh.
*   **Gợi ý trả lời cho sinh viên:**
    1. *Bản chất bài toán công nghiệp:* YOLO là closed-set detector, chỉ phát hiện những gì đã được gán nhãn trong 6 lớp. Trong sản xuất, khuyết tật có hình thái vô định hình (vết xước nhẹ, cặn oxy hóa, sợi tơ đồng mỏng) không thể gán nhãn hết. EfficientAD đóng vai trò chốt chặn an toàn (safety net) dựa trên phân phối bo chuẩn.
    2. *Chiến lược bù trừ:* YOLO phụ trách phần lớn các lỗi phổ biến với tốc độ cực nhanh (early exit). Tầng 2 chỉ kích hoạt cho nhóm ảnh nghi vấn (YOLO score thấp), do đó độ trễ trung bình của toàn pipeline chỉ tăng thêm $\Delta t = \text{route\_rate} \times t_{AD}$. Nếu dữ liệu thực nghiệm cho thấy EfficientAD không cứu được tối thiểu 5% số lỗi YOLO bỏ sót ở cùng mức FRR, đồ án sẽ trung thực kết luận mô hình đơn là giải pháp tối ưu hơn theo nguyên lý Occam's Razor.

### Câu hỏi 2: Về tính hợp lệ của tập kiểm thử
> *"Tập test của các em có 440 ảnh nhưng thực chất chỉ được cắt ra từ 2 tấm quét bo mạch lớn (`group44000`, `group92000`). Làm sao hội đồng tin được mô hình của các em không học vẹt layout của 2 tấm bo này? Nếu thay bằng một bo vi điều khiển STM32 hoàn toàn mới, hệ thống có chạy được không?"*

*   **Hiện trạng kế hoạch:** Thừa nhận giới hạn trong văn bản nhưng chưa có giải pháp thực nghiệm để giảm nhẹ rủi ro.
*   **Gợi ý trả lời cho sinh viên:**
    1. *Thừa nhận khách quan:* Nhóm bảo vệ sự nghiêm ngặt khoa học bằng việc từ chối chia ngẫu nhiên (random split) vì chia ngẫu nhiên sẽ làm lọt hình ảnh cùng tấm quét vào train và test, tạo ra độ chính xác giả tạo (99%). Việc cách ly hoàn toàn 2 group cho test là kịch bản khắt khe nhất có thể thực hiện trên DeepPCB.
    2. *Giải pháp kỹ thuật bù đắp:* Báo cáo kết quả phân rã chi tiết trên từng group riêng biệt (`worst-group recall`). Đối với bo mạch hoàn toàn mới (khác layout), EfficientAD theo định nghĩa sẽ xem toàn bộ layout lạ là bất thường nếu không được nạp ảnh chuẩn của bo đó. Nhóm định vị rõ phạm vi của hệ thống là: **Kiểm tra bo mạch theo mã sản phẩm (SKU-specific inspection)**, trong đó EfficientAD yêu cầu nạp ảnh mẫu chuẩn của SKU đó trước khi chạy dây chuyền, không phải hệ thống nhận diện mù mọi loại bo mạch (Universal Zero-shot).

### Câu hỏi 3: Về sự thiếu hụt Ground Truth Mask cho Anomaly Detection
> *"DeepPCB chỉ có bounding box chứ không có ground truth mask ở mức pixel. Vậy bản đồ nhiệt (anomaly map) của EfficientAD được các em đánh giá định lượng bằng cách nào? Các em có đo IoU hay Dice không? Nếu quy heatmap thành bbox thì thuật toán cụ thể là gì?"*

*   **Hiện trạng kế hoạch:** Kế hoạch đang né tránh bằng cách đẩy việc đánh giá bbox của AD sang "bảng phụ" và chưa có thuật toán chuyển đổi cụ thể.
*   **Gợi ý trả lời cho sinh viên:**
    1. *Phương pháp luận:* Khẳng định không dùng bounding box thô để tính Pixel AUROC hay Dice Score vì làm vậy là phi khoa học (do bounding box chứa 80% là diện tích không lỗi).
    2. *Đánh giá kép:* 
       - Ở mức ảnh (Image-level): Đánh giá thuần túy bằng khả năng phân loại nhị phân Good/Defect thông qua Image Anomaly Score ($S = \max(Map)$ hoặc Top-$K$ mean).
       - Ở mức định vị (Localization): Áp dụng ngưỡng nhị phân hóa cố định trên heatmap, trích xuất vùng liên thông (Connected Components) có diện tích lớn hơn $A_{min}$ thành bounding box class-agnostic. Đánh giá bằng **Point-in-Box Recall** (nếu điểm cực đại của heatmap rơi vào trong Ground Truth BBox thì tính là phát hiện vị trí thành công), thay vì ép tính IoU diện tích.

### Câu hỏi 4: Về hiện tượng co cụm độ trễ (Latency Bottleneck) và mất chi tiết
> *"EfficientAD yêu cầu đầu vào kích thước nhỏ (256×256), trong khi DeepPCB là 640×640 và bo mạch thực tế còn lớn hơn nhiều. Khi nén ảnh, các khuyết tật nhỏ như lỗ kim (pin-hole) hoặc gai đồng (spur) chỉ có kích thước vài pixel sẽ bị triệt tiêu hoàn toàn. Các em giải quyết bài toán trade-off giữa độ phân giải và tốc độ xử lý như thế nào?"*

*   **Hiện trạng kế hoạch:** Mục 5 (Bước 4) mới chỉ ghi nhận "đánh giá rủi ro mất lỗi nhỏ", chưa có giải pháp kỹ thuật bắt buộc.
*   **Gợi ý trả lời cho sinh viên:**
    1. *Phân tích định lượng:* Nhóm đã tính toán độ phân giải tối thiểu: một khuyết tật hở mạch rộng 3 pixel trên 640×640 khi nén về 256×256 sẽ chỉ còn $3 \times (256/640) = 1.2$ pixel, mất hoàn toàn đặc trưng biên độ tương phản.
    2. *Kiến trúc giải quyết:* Không dùng phương pháp resize toàn ảnh. Hệ thống áp dụng cơ chế **Patched Inference (Tiling)**: Cắt bo thành các ô chuẩn $256\times 256$ với độ chồng lấn 15% (overlap) để giữ nguyên tỷ lệ $1:1$ của pixel cảm biến. Nhánh YOLO vẫn nhận ảnh tổng thể 640×640 để bắt bối cảnh, còn EfficientAD chỉ quét chi tiết trên các tile. Điều này giúp bảo toàn 100% các lỗi vi mô.

### Câu hỏi 5: Về tính thiên lệch của Thang đo Benchmark tổng hợp
> *"Nếu các em dùng một công thức tổng hợp điểm (Weighted Sum) hoặc chuẩn hóa Min-Max để xếp hạng các mô hình, làm sao các em chứng minh trọng số đó không phải do các em tự gán để hướng tới mô hình các em mong muốn? Hiện tượng đảo ngược thứ hạng (Rank Reversal) được xử lý ra sao?"*

*   **Hiện trạng kế hoạch:** Bản kế hoạch hiện tại (Mục 7.6) đã có quyết định đúng đắn là **từ chối điểm số tổng hợp tùy tiện** và đề xuất dùng mặt phẳng Pareto, nhưng cần làm rõ cơ chế ra quyết định cuối cùng.
*   **Gợi ý trả lời cho sinh viên:**
    1. *Chỉ ra khiếm khuyết của Weighted Sum:* Chuẩn hóa Min-Max làm biến dạng tỷ lệ khi có ngoại lai (outlier), và việc thêm/bớt mô hình trong tập khảo sát sẽ làm thay đổi giá trị Min/Max, dẫn đến hiện tượng mô hình A đang thắng mô hình B bỗng nhiên bị đảo ngược thứ hạng mà không hề thay đổi bản chất dự đoán.
    2. *Khung đánh giá chuẩn mực đã chọn:* Đồ án không dùng điểm số gộp 1 chiều. Nhóm sử dụng **Mặt phẳng hiệu quả Pareto (Pareto Efficiency Frontier)** giữa 2 trục mâu thuẫn trực tiếp: `Defect Recall` và `FRR (False Reject Rate)` tại ràng buộc trễ $p95 \le T_{max}$. Mô hình được chọn là điểm uốn (elbow point) trên đường Pareto có chi phí tổn thất sản xuất dự tính thấp nhất theo hàm tổn thất công nghiệp.

---

## 5. Checklist thay đổi đề xuất cho PLAN.md

Checklist được sắp xếp theo mức độ ưu tiên giảm dần từ **P0 (Sửa ngay lập tức để tránh điểm liệt/bắt bẻ hội đồng)** đến **P2 (Hoàn thiện kỹ thuật nếu còn thời gian)**:

### Nhóm P0: Cốt lõi sống còn (Sửa trong 24h tới)
- [ ] **P0.1 - Xóa bỏ toàn bộ các tham chiếu đến `YOLO26`:** Thay bằng `YOLOv8n` hoặc giữ duy nhất `YOLO11n` và `YOLO11s` làm detector baseline. Sửa triệt để các mục 7.1, 7.3 và 13 trong file [PLAN.md](file:///d:/FPTU/KLTN/PCB_YOLO_EfficientAD_Lab/PLAN.md).
- [ ] **P0.2 - Làm rõ ranh giới bài toán về độ tự do của Test set:** Bổ sung ghi chú kỹ thuật: Khẳng định tập test đánh giá trên 2 layout cụ thể (`group44000`, `group92000`). Bổ sung phương án chạy bổ trợ **LOGO-CV (Leave-One-Group-Out)** trên 11 groups nếu phần cứng cho phép để lấy khoảng tin cậy layout.
- [ ] **P0.3 - Khóa công thức bài toán tối ưu ngưỡng $(\tau_{yolo}, \tau_{ad})$:** Thay vì chỉnh tuần tự từng tầng khiến tầng 2 thiếu mẫu defect, chuyển sang thuật toán Grid Search 2 chiều trên tập `calibration` nhằm tối đa hóa `Defect Recall` dưới ràng buộc cứng $\text{FRR} \le 5\%$.
- [ ] **P0.4 - Thay đổi chiến lược kích thước ảnh của EfficientAD:** Bỏ hoàn toàn tùy chọn resize toàn ảnh về 256×256. Đặt mặc định là chiến lược chia ô (tiling $256\times 256$ hoặc $320\times 320$) để chống biến mất lỗi nhỏ.

### Nhóm P1: Chuẩn hóa Phương pháp luận & Phần mềm (Hoàn thành trong Tuần 1–2)
- [ ] **P1.1 - Khóa thuật toán Heatmap-to-BBox:** Viết mã rõ ràng cho hàm `heatmap_to_boxes(anomaly_map, threshold, min_area)` trong module hậu xử lý; công bố quy tắc đánh giá class-agnostic localization bằng tiêu chí Point-in-Box thay vì IoU diện tích.
- [ ] **P1.2 - Thêm trục so sánh Iso-FRR (DET Curve):** Đưa biểu đồ DET / ROC vào báo cáo benchmark. Đảm bảo bảng so sánh cuối cùng phải có cột: `Recall tại FRR = 3%` cho cả 3 mô hình B01 (YOLO), B03 (EfficientAD), và H01 (Cascade).
- [ ] **P1.3 - Thiết lập Abstract Interfaces trong Codebase:** Tạo file `src/pcb_lab/models/base.py` định nghĩa rõ hai lớp cơ sở:
  - `BaseDetector`: `predict(img) -> List[DetectionBox]`
  - `BaseAnomalyDetector`: `predict(img) -> Tuple[float, np.ndarray]`
  - `BaseImageSource`: `get_frame() -> Tuple[str, np.ndarray]` (tách hoàn toàn UI khỏi I/O).
- [ ] **P1.4 - Thêm mô phỏng tổn thất chi phí sản xuất (Industrial Cost Matrix):** Thêm 1 cột metric trong bảng benchmark: $Cost = 100 \times FN + 1 \times FP$ để phản ánh đúng bài toán kinh tế trong nhà máy sản xuất linh kiện điện tử.

### Nhóm P2: Tối ưu hóa & Nâng cao (Tuần 3 trở đi)
- [ ] **P2.1 - Cắt giảm danh sách mô hình mở rộng:** Tạm hoãn PatchCore, STFPM, FastFlow, EfficientAD-M ra khỏi scope chính thức của đồ án. Dồn 100% GPU-hours để hoàn thành trọn vẹn và chuẩn xác cặp đôi `YOLO11n` + `EfficientAD-S`.
- [ ] **P2.2 - Nâng cấp Robustness Suite với nhiễu công nghiệp:** Bổ sung các hàm tạo nhiễu đặc thù AOI trong module kiểm thử: lóa sáng kim loại (`specular_glare`), rung mờ cơ khí (`motion_blur`), và bụi giả lập (`dust_artifacts`).
- [ ] **P2.3 - Thiết lập quy trình cố định môi trường tái lập:** Đóng gói môi trường bằng file `requirements.lock` kèm thông tin driver và cờ bắt buộc cho PyTorch:
  ```python
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```
