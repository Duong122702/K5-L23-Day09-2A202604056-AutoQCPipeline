"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


NAME = "label_format"


def _overflow(cx, cy, w, h):
    """Mức tràn lớn nhất ra ngoài [0,1] trên 4 mép (<= 0 nghĩa là nằm trong ảnh)."""
    left, right = cx - w / 2, cx + w / 2
    top, bottom = cy - h / 2, cy + h / 2
    return max(0.0 - left, right - 1.0, 0.0 - top, bottom - 1.0)


def run(ds, cfg):
    p = cfg["label_format"]
    tol, min_px = p["overflow_tol"], p["min_box_px"]
    max_frac, dup_iou = p["max_box_frac"], p["dup_box_iou"]
    n_classes = len(ds.classes)

    issues = []
    for lab in ds.label_files():
        stem = lab.stem
        lines = ds.read_label_lines(stem) or []      # (số dòng gốc, text); dòng trống đã bị bỏ nhưng số dòng giữ nguyên
        size = ds.image_size(stem) if lines else None  # None: nhãn mồ côi / ảnh hỏng -> bỏ qua BOX_TINY
        seen = []                                     # (line_no, class_id, box) của các dòng hợp lệ TRƯỚC đó

        for ln, text in lines:
            # 1) Định dạng: dòng hỏng thì báo và bỏ qua, không crash cả file
            try:
                c, cx, cy, w, h = parse_yolo_line(text)
            except ValueError as e:
                issues.append(Issue(NAME, ERROR, "BAD_FORMAT", stem, f"sai định dạng: {e}", ln))
                continue
            # 2) ID lớp
            if not 0 <= c < n_classes:
                issues.append(Issue(NAME, ERROR, "BAD_CLASS_ID", stem,
                                    f"id lớp {c} ngoài 0..{n_classes - 1}", ln))
                continue
            # 3) Box rỗng
            if w <= 0 or h <= 0:
                issues.append(Issue(NAME, ERROR, "BOX_ZERO", stem, f"box rỗng w={w} h={h}", ln))
                continue

            box = (cx, cy, w, h)
            # 4) Tràn ảnh (một dòng có thể dính nhiều lỗi hình học -> không continue từ đây)
            over = _overflow(*box)
            if over > tol:
                issues.append(Issue(NAME, ERROR, "BOX_OUT_OF_IMAGE", stem,
                                    f"box tràn ảnh {over:.4f}", ln, {"overflow": round(over, 6)}))
            # 5) Box lặp: cùng lớp với một dòng trước, IoU > ngưỡng; báo ở dòng sau
            for prev_ln, prev_c, prev_box in seen:
                if prev_c == c:
                    iou = iou_xywh(prev_box, box)
                    if iou > dup_iou:
                        issues.append(Issue(NAME, ERROR, "DUP_BOX", stem,
                                            f"trùng box dòng {prev_ln} (IoU={iou:.3f})", ln,
                                            {"dup_of": prev_ln, "iou": round(iou, 4)}))
                        break
            seen.append((ln, c, box))
            # 6) Box quá nhỏ: cạnh ngắn tính theo pixel
            if size:
                short_px = min(w * size[0], h * size[1])
                if short_px < min_px:
                    issues.append(Issue(NAME, WARN, "BOX_TINY", stem,
                                        f"cạnh ngắn {short_px:.1f}px < {min_px}px", ln,
                                        {"short_px": round(short_px, 2)}))
            # 7) Box quá to
            if w * h > max_frac:
                issues.append(Issue(NAME, WARN, "BOX_HUGE", stem,
                                    f"box chiếm {w * h:.0%} ảnh", ln, {"area": round(w * h, 4)}))
    return issues
