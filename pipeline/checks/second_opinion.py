"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds

NAME = "second_opinion"


def _labels_with_lines(ds, stem):
    """[(số dòng gốc, id_lớp, (cx, cy, w, h))] — cùng bộ lọc với ds.valid_boxes nhưng GIỮ số dòng."""
    out = []
    for ln, text in ds.read_label_lines(stem) or []:
        try:
            c, *box = parse_yolo_line(text)
        except ValueError:
            continue                             # dòng hỏng: label_format đã báo, ở đây chỉ bỏ qua
        if 0 <= c < len(ds.classes):
            out.append((ln, c, tuple(box)))
    return out


def _best(cands):
    """(iou, ...) có IoU lớn nhất, hoặc None."""
    return max(cands, key=lambda t: t[0]) if cands else None


def run(ds, cfg, preds=None):
    p = cfg["second_opinion"]
    conf_min, min_area, fp_conf = p["conf"], p["min_area_px"], p["fp_conf"]
    iou_missing, iou_match, iou_loose = p["iou_missing"], p["iou_match"], p["iou_loose"]
    if preds is None:
        preds = get_preds(ds, cfg, "checker")
    names = ds.classes

    issues = []
    for img in ds.image_files():                 # thứ tự: theo ảnh ...
        stem = img.stem
        if stem not in preds or not ds.label_path(stem).exists():
            continue                             # không có ý kiến model / chưa có nhãn (MISSING_LABEL_FILE lo) -> bỏ qua
        rows = preds[stem] or []
        labels = _labels_with_lines(ds, stem)
        size = ds.image_size(stem)

        # ---- A. Từ phía MODEL: vật model thấy (tin cậy, đủ to) có được gán đúng không?  (... rồi theo thứ tự dự đoán)
        for r in rows:
            rc, rconf, rbox = int(r[0]), float(r[1]), tuple(r[2:6])
            if rconf < conf_min:
                continue
            if size and rbox[2] * size[0] * rbox[3] * size[1] < min_area:
                continue                         # vật quá nhỏ: guideline không gán -> không đòi nhãn
            cands = [(iou, lc, ln) for ln, lc, lbox in labels
                     if (iou := iou_xywh(rbox, lbox)) >= iou_missing]
            extra = {"conf": round(rconf, 4), "pred_class": names[rc] if 0 <= rc < len(names) else rc,
                     "pred_box": [round(v, 4) for v in rbox]}

            if not cands:                                            # (1) không nhãn nào chạm vật này
                issues.append(Issue(NAME, WARN, "POSSIBLE_MISSING", stem,
                                    f"model thấy {extra['pred_class']} (conf {rconf:.2f}) nhưng không có box nhãn", None, extra))
                continue
            same = _best([c for c in cands if c[1] == rc])
            if same:                                                 # (2) có nhãn cùng lớp
                if same[0] < iou_loose:
                    issues.append(Issue(NAME, WARN, "POSSIBLE_LOOSE_BOX", stem,
                                        f"box {extra['pred_class']} dòng {same[2]} lệch/lỏng: IoU {same[0]:.2f} < {iou_loose}",
                                        None, {**extra, "iou": round(same[0], 4), "label_line": same[2]}))
                continue                                             # IoU đủ cao: nhãn đúng lớp đã phủ vật -> im lặng
            other = max(cands, key=lambda t: t[0])                   # (3) chỉ có nhãn khác lớp (cands không rỗng)
            if other[0] >= iou_match:
                lc = other[1]
                issues.append(Issue(NAME, WARN, "POSSIBLE_WRONG_CLASS", stem,
                                    f"dòng {other[2]} gán {names[lc]}, model nói {extra['pred_class']} (IoU {other[0]:.2f})",
                                    None, {**extra, "iou": round(other[0], 4), "label_line": other[2],
                                           "label_class": names[lc]}))
            # IoU khác lớp trong [iou_missing, iou_match): mơ hồ -> không cờ

        # ---- B. Từ phía NHÃN: box nhãn không khớp vật nào model thấy (conf >= fp_conf, KHÔNG lọc diện tích)
        fp_boxes = [tuple(r[2:6]) for r in rows if float(r[1]) >= fp_conf]
        for ln, lc, lbox in labels:                                  # theo số dòng
            best = max((iou_xywh(lbox, b) for b in fp_boxes), default=0.0)
            if best < iou_missing:
                issues.append(Issue(NAME, WARN, "POSSIBLE_FALSE_POS", stem,
                                    f"box {names[lc]} không khớp vật nào model thấy (IoU tốt nhất {best:.2f})",
                                    ln, {"iou": round(best, 4)}))
    return issues
