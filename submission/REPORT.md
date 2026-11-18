# Báo cáo Day 09 — Cao Văn Dương, 2A202604056

## 1. Kết luận trước
**Hold** (mã thoát 10). Quét **224** ảnh, loại **22** ở cửa vào, còn **202** ảnh vào QC; `export/clean_v1` có 202 ảnh nhưng chỉ là **bản nháp, chưa phát hành**.
Không còn lỗi bắt buộc (`n_errors = 0`), nhưng sau khi người duyệt hết ngân sách 80 ảnh vẫn còn **38/202 ảnh = 18,8%** có cảnh báo chưa ai xem, vượt ngưỡng 10% (`08_summary.json`). Phần lớn là `POSSIBLE_FALSE_POS` (21) và `POSSIBLE_LOOSE_BOX` (20) trong `07_error_list_v3`.

## 2. Lỗi và phân việc
Từ `02_error_list_v1` (211 issue; mẫu số 224 ảnh quét ở cửa vào, 202 ảnh ở QC nhãn):

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| DUP_IMAGE_EXACT / DUP_IMAGE_NEAR | 8 / 4 ảnh | máy: loại ở cửa vào | trùng làm lệch phân bố, rò rỉ giữa train/val |
| LEAK_TEST | 5 ảnh | máy: loại ở cửa vào | ảnh trùng tập test → mAP ảo |
| BAD_SIZE / CORRUPT_IMAGE | 3 / 2 ảnh | máy: loại ở cửa vào | ảnh < 250px hoặc không mở được |
| BOX_OUT_OF_IMAGE | 8 dòng | máy: `clip_box` (8) | cắt về [0,1], giữ lớp; người xem lại (`FIXED_CLIP_BOX`) |
| DUP_BOX / BAD_CLASS_ID / BOX_ZERO / BAD_FORMAT | 7 / 6 / 4 / 3 dòng | máy: `drop_line` (20) | dòng không dùng được; xoá có log trước → sau trong `03_fixes.csv` |
| BAD_CLASS_NAME | 2 issue, 1 file `data.yaml` | máy: `rename_classes` (1) | đổi `names` về đúng `cfg["classes"]` |
| MISSING_LABEL_FILE | 6 ảnh | **người** (`needs_label`) | máy không bịa nhãn, không tạo file rỗng (rỗng = ảnh nền) |
| POSSIBLE_MISSING | 53 cờ / 52 ảnh | người | model thấy vật, nhãn không có box |
| POSSIBLE_LOOSE_BOX | 28 cờ / 24 ảnh | người | đúng vật, box lệch/lỏng (IoU < 0,5) |
| POSSIBLE_FALSE_POS | 29 dòng / 27 ảnh | người | box nhãn không khớp vật nào model thấy |
| POSSIBLE_WRONG_CLASS | 18 ảnh | người | model nói lớp khác với IoU ≥ 0,5 |
| EMPTY_LABEL | 12 ảnh | người | ảnh nền hay quên gán? chỉ người phân biệt được |
| BOX_TINY / BOX_HUGE | 6 dòng (4 ảnh) / 3 ảnh | người | quy tắc dự án, chỉ cảnh báo |
| CLASS_RARE | 4 cờ cấp dataset | người (quyết định thu thêm dữ liệu) | không sửa được bằng nhãn |

**Hàng đợi (`05_review_queue.csv`, 125 ảnh > ngân sách 80):** điểm của mỗi ảnh là Σ trọng số × (1 + conf). Trọng số lấy từ `review.weights`: NEEDS_LABEL 10, WRONG_CLASS 3, MISSING/FALSE_POS/FIXED_DROP_LINE 2, LOOSE 1,5.
- Đứng đầu là `img_0020` (11,5): ảnh này vừa bị máy xoá dòng, vừa có nghi thiếu box, vừa nghi nhầm lớp.
- Tiếp theo là **6 ảnh `NEEDS_LABEL`** (10,0), vì không có người thì những ảnh này là lỗi bắt buộc và kết luận sẽ thành Rework.
- Ảnh thứ 80 có điểm 3,34.

**Với 80 ảnh duyệt**, người phủ được: MISSING ở 50/52 ảnh, WRONG_CLASS ở 18/18, NEEDS_LABEL ở 6/6, EMPTY_LABEL ở 11/15, FALSE_POS ở 12/23 và LOOSE_BOX ở 8/24. Tổng cộng ≈ 88/117 cặp (ảnh, cờ ngữ nghĩa) = 75%. Phần sót chủ yếu là FALSE_POS và LOOSE_BOX, vì hai loại này trọng số thấp và precision thấp.

## 3. Calibration (2 ngưỡng)
Sweep trong `calibration/calibration_report.md`; số dưới đây đo trên calib (39 ảnh), dạng precision / recall.

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **10** | NEAR 1,00/0,50 → **1,00/0,67** (EXACT giữ 1,00/1,00) | Cờ sai ở đây là **xoá ảnh tốt**, nên ưu tiên precision. Trên raw, mọi cặp ở dist ≤ 10 đều trùng thật (đã mở xem); từ dist 16 trở đi bắt đầu gộp nhầm. Chọn 10 thay vì 16 dù calib cho 16 recall 0,83. |
| `second_opinion.conf` | 0,25 | **0,6** | MISSING 0,50/1,00 → **0,57/1,00**; WRONG_CLASS 0,56/0,83 → **0,62/0,83** | Đây là giá trị cao nhất mà recall MISSING vẫn 1,00. Hàng đợi raw giảm 145 → 125 ảnh. Giá 0,8 cho Release (7,8%) nhưng recall MISSING chỉ 0,75 = Release giả, nên không chọn. Chấp nhận Hold. |

**Calibration đánh lừa ở `phash_dist`.** Trên calib, precision là 1,00 ở **mọi** giá trị 2–20 và dist 16 có F1 cao nhất (0,91). Nhưng trên raw:
- dist 16 loại **31** ảnh, so với 22 ảnh ở dist 10.
- Trong 3 cặp mới thêm ở dist 16 thì **2 cặp là ảnh hoàn toàn khác nhau**: `img_0011↔img_0008` (tennis và trượt tuyết), `img_0177↔img_0113` (lướt sóng và người ngồi bàn).
- dist 20 loại tới **139/224** ảnh.

Calib quá nhỏ nên gần như không có cặp ảnh nào khác nhau mà phash lại gần nhau, vì thế không lộ ra FP.

Chiều ngược lại cũng không đo được: calib **không có ca `LEAK_TEST`**, trong khi `phash_dist` cũng dùng cho rò rỉ test. Ở dist 10, máy bắt thêm `img_0101 ↔ test/test_039` (cùng cảnh, đã crop và đổi màu), một rò rỉ mà dist 2 bỏ sót. Calibration không hề cho thấy lợi ích này.

Ngoài ra, mỗi bậc của recall MISSING (1,00 → 0,88 → 0,75) chỉ tương ứng **1 lỗi chèn** trên tổng 8, nên con số này rất nhiễu.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Gate chặn đúng.** Chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người duyệt):
- Kết quả là **Rework**, **mã thoát 20**.
- `n_errors = 6`, tất cả là `MISSING_LABEL_FILE`: 6 ảnh chưa có nhãn mà không ai gán. `warn_ratio` là 112/202 = 55,4%.
- Log in `KHÔNG xuất: còn lỗi bắt buộc (Rework)`, và thư mục `work/exp_nb/clean_v1` **không được tạo**.
- So với lần chạy chính (budget 80): người gán 6 ảnh này nên `n_errors = 0` và kết luận chuyển sang Hold.

**(b) Điểm mù có bằng chứng:**
1. **Ảnh trùng biến đổi mạnh.** `img_0084` là bản crop của `img_0074`, nhưng có phash dist 16, đúng bằng khoảng cách của 2 cặp ảnh khác nhau ở trên. Không ngưỡng nào tách được, nên `img_0084` vẫn nằm trong dataset.
2. **Đám đông, vật bị che: model lớn cũng sót.** Ở `img_0127`, nhiều trẻ em trong đám đông không có box, nhưng máy chỉ cờ **1** `POSSIBLE_MISSING` (conf 0,68) và 2 `LOOSE_BOX`. Khi chính model cũng không thấy thì "ý kiến thứ hai" im lặng.
3. **Box lệch nhẹ và nhầm lớp mà model cũng nhầm.** Trên calib, `LOOSE_BOX` chỉ đạt 0,20/0,20 (bắt 1/5) và `WRONG_CLASS` sót 1/6. Box lệch nhưng IoU vẫn ≥ 0,5 thì không bị cờ (theo thiết kế).
4. **`FALSE_POS` nhiễu.** Precision chỉ 0,31 trên calib, và `conf` không lọc được loại này vì nó dùng `fp_conf` cố định. Dù vậy vẫn có cờ đúng: ở `img_0177`, một box "bus" nằm trên mặt biển. Gate chỉ đếm số ảnh, không phân biệt mức nặng nhẹ của cờ.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Train trên Colab, GPU T4. Model `yolo26n.pt`, **20 epoch**, batch 16, imgsz 640, **seed 0 và 1**. Cả hai biến thể dùng chung val `gold_val` (40 ảnh) và test (100 ảnh nhãn người). Bảng dưới lấy từ `submission/results/comparison.md`:

| biến thể | #ảnh train | số seed | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|---|
| raw | 224 | 2 | 0.726 ± 0.000 | 0.512 ± 0.004 | 0.764 ± 0.061 | 0.649 ± 0.028 |
| clean | 202 | 2 | 0.701 ± 0.025 | 0.487 ± 0.008 | 0.759 ± 0.003 | 0.625 ± 0.019 |

**Chênh lệch so với nhiễu.**
- **mAP50 (clean − raw) = −0,025.** Con số này **bằng đúng độ lệch chuẩn giữa 2 seed của clean** (0,025): clean seed 0 đạt 0,718, seed 1 chỉ 0,683, trong khi raw là 0,726 ở cả hai seed.
- **mAP50-95** chênh −0,024. Con số này lớn hơn độ lệch chuẩn (0,004 và 0,008), nhưng độ lệch chuẩn ước lượng từ **chỉ 2 điểm** thì không đáng tin.
- **Kết luận trung thực:** chưa kết luận được rằng `clean` tốt hơn, và cũng chưa đủ bằng chứng để nói `raw` tốt hơn. Chênh lệch nằm trong vùng nhiễu.

**Chênh lệch dồn vào một lớp.** Theo AP50 trung bình 2 seed (`*_seed*.json`):

| Lớp | raw | clean |
|---|---|---|
| car | 0,738 | 0,738 |
| person | 0,822 | 0,823 |
| bus | 0,695 | 0,703 |
| **truck** | **0,648** | **0,538** |

Ba lớp đầu gần như bằng nhau. Truck lại là lớp ít box nhất: 24 box trong test, 32 box ở raw và 41 box ở clean. Riêng truck của clean đã dao động **0,590 → 0,486** giữa 2 seed, tức một mức chênh cỡ bằng khoảng cách raw và clean. Vì vậy chênh lệch tổng chủ yếu là **nhiễu của một lớp hiếm**.

Điều đáng chú ý: sau khi người duyệt, clean có **nhiều** box truck hơn raw (41 so với 32), và car giảm từ 122 xuống 101. Nhãn đã được sửa theo hướng của nhãn người, nhưng 20 epoch với khoảng 40 box là quá ít để thấy được lợi ích.

**Nhìn bằng mắt** (`submission/results/predictions_seed0.png`: 6 ảnh test đầu tiên theo tên, seed 0, conf 0,25):
- Ở ảnh đường phố, raw nhận đúng xe tải vàng là `truck 0.47`, còn clean nhầm thành `bus 0.26`. Đây đúng là kiểu lỗi kéo AP truck của clean xuống.
- Ngược lại, clean gọn hơn ở 2 ảnh: ảnh bánh mì và ảnh bánh kem, raw ra thêm một box person lớn gần phủ cả ảnh, clean chỉ ra 1 box.
- Conf của clean thường cao hơn: người trượt tuyết 0,81 → 0,92, ảnh chiếc giày 0,66 → 0,82.

6 ảnh là quá ít để kết luận; chỉ dùng để minh hoạ cho các con số ở trên.

**Giả thuyết và cách kiểm:**
1. **Rò rỉ test thổi phồng điểm của raw.** raw chứa 5 ảnh trùng với tập test: `test_009`, `033`, `039`, `044`, `085`, gồm 8 car, 12 person và 1 truck (`LEAK_TEST` trong `01_intake`). Như vậy 5% tập test đã nằm trong tập train của raw. *Cách kiểm:* đánh giá lại cả hai model trên 95 ảnh test còn lại, sau khi bỏ 5 ảnh này.
2. **Khác cả số ảnh lẫn nhãn cùng lúc.** raw có nhiều hơn 22 ảnh, nhưng trong đó có 12 ảnh trùng, 2 ảnh hỏng, 6 ảnh không có file nhãn (Ultralytics coi là ảnh nền) và 6 dòng id lớp lạ. Hai biến số bị trộn với nhau nên không tách được tác động. *Cách kiểm:* train raw chỉ trên đúng 202 ảnh có trong `manifest.json` nhưng giữ nhãn gốc, để chỉ còn nhãn là khác nhau.
3. **clean chưa thật sự sạch.** Còn 38 ảnh có cảnh báo chưa ai duyệt (Hold), chủ yếu là FALSE_POS và LOOSE_BOX. *Cách kiểm:* thêm biến thể `oracle` (notebook có sẵn) để có giới hạn trên của "dữ liệu sạch".
4. **Nhiễu do test nhỏ và ít seed.** Mỗi ảnh test đáng khoảng 1%, và truck chỉ có 24 box. *Cách kiểm:* chạy thêm seed 2–4, và bootstrap tập test để có khoảng tin cậy.

**Việc tiếp theo:** chạy lại với ít nhất 3 seed, có biến thể `oracle`, và đánh giá trên tập test đã bỏ 5 ảnh rò rỉ. Sau đó mới đánh giá được lợi ích thật của pipeline.

## 6. Quyết định và ghi chú AI
**(a) Quyết định: Hold.** Dataset chưa phát hành; `clean_v1` chỉ dùng để thử nghiệm.

Để thành **Release**: số ảnh còn cảnh báo chưa duyệt phải nhỏ hơn 10% × 202, tức ≤ 20 ảnh. Nghĩa là cần **duyệt thêm ít nhất 18 trong 38 ảnh** còn lại, theo thứ tự hàng đợi từ vị trí 81 trở đi (ngân sách khoảng 98 ảnh thay vì 80), ưu tiên FALSE_POS và LOOSE_BOX.

**Không** nâng `conf` lên 0,8 để né Hold: làm vậy chỉ giấu bớt cờ (recall MISSING giảm còn 0,75), chứ không làm nhãn sạch hơn.

**(b) Ghi chú AI.** Dùng Claude Code để viết 4 module.
- **Prompt chính:** "viết `<module>.py`, đọc kỹ docstring trong file", kèm file test. AI đọc docstring, `common.py`, `conftest.py` và test trước khi viết.
- **Cách kiểm:**
  - đọc diff, xác nhận không xoá hoặc ghi vào `data/raw` và không viết cứng ngưỡng;
  - chạy `pytest` từng module rồi chạy cả bộ (79/79 xanh);
  - thử thêm các ca ngoài test (một dòng dính 2 lỗi, dòng trống ở giữa, chạy autofix lần 2 để kiểm idempotent);
  - mở ảnh bằng `viz_labels.py` để kiểm các cờ và các cặp trùng.
- **AI sai:** sau khi đổi `configs/pipeline.yaml`, AI để nguyên `calibration.json` sinh từ cấu hình cũ. Lần chạy lại đầu tiên còn bị cắt output nên trả mã 255. Mình phát hiện khi đối chiếu file với cấu hình mới, rồi chạy lại calibrate đầy đủ (mã 0).

*(Thưởng CVAT: không làm.)*
