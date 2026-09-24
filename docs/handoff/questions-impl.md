# Câu hỏi IMPLEMENTER — Step 2

## Q1 — Sample.pair_id khi canonical có nhiều pair alias (đang chờ)

- Contract `data/samples.py` yêu cầu `Sample.pair_id` nhưng chưa chốt kiểu/ánh xạ từ `pair_ids` của manifest.
- Bằng chứng đọc metadata qua adapter Step 1, không decode ảnh: `deeppcb_20085159_good` có `pair_ids=["20085159","20085167","20085168"]`.
- Đề xuất: `pair_id: str = min(pair_ids)` và thêm `pair_ids: tuple[str,...]` chứa toàn bộ alias theo thứ tự ổn định; mọi isolation/grouping cần đầy đủ quan hệ sẽ dùng pair_ids. Phương án khác: chốt pair_id là tuple toàn bộ alias.
- Dừng phần Sample/loader phụ thuộc quyết định này; tiếp tục TileManager/preprocessing. Không sửa contract. Cần người điều phối chốt để VERIFIER dùng cùng giao ước.

## Q2 — Duyệt recipe aug_v1 (đang chờ theo yêu cầu Phase 0.3)

- Đề xuất chi tiết tại `step2-impl.md`: train-only, k×90° (k đều 0..3) rồi horizontal flip p=0.5, seed cục bộ; hoán vị pixel/box chính xác, không nội suy/crop/photometric.
- Rủi ro: thay phân phối hướng layout. Không làm mất hoặc thay hình thái chi tiết lỗi do resampling, nhưng chưa chứng minh cải thiện model.
- Chờ duyệt trước khi viết/áp augmentation; đây là yêu cầu trực tiếp của người dùng, không phải approval do skill.
