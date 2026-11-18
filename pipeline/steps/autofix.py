"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict

import yaml

from ..common import ERROR, parse_yolo_line

DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
CLIP_CODES = {"BOX_OUT_OF_IMAGE"}
REMOVE_CODES = {"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}


def _fx(image, line, action, before, after):
    return {"image": image, "line": line, "action": action, "before": before, "after": after}


def _clip(text):
    """Cắt box về trong [0,1]. Trả về dòng mới, hoặc None nếu box không còn diện tích (nằm hẳn ngoài ảnh)."""
    c, cx, cy, w, h = parse_yolo_line(text)
    x1, x2 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
    y1, y2 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
    nw, nh = x2 - x1, y2 - y1
    if nw <= 0 or nh <= 0:
        return None
    return f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {nw:.6f} {nh:.6f}"


def _remove_image(ds, stem):
    """Xoá mọi file ảnh có stem này + file nhãn. Trả về list tên file đã xoá (rỗng nếu không còn gì: idempotent)."""
    gone = []
    for p in ds.image_files():
        if p.stem == stem:
            p.unlink()
            gone.append(p.name)
    lab = ds.label_path(stem)
    if lab.exists():
        lab.unlink()
        gone.append(f"labels/{lab.name}")
    ds._size_cache.pop(stem, None)
    return gone


def _rename_classes(ds, cfg):
    """Ghi lại `names` trong data.yaml đúng cfg["classes"], giữ các khoá khác. None nếu đã đúng."""
    data = {}
    if ds.yaml_path.exists():
        data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
    want = list(cfg["classes"])
    before = ds.yaml_names()
    if [before.get(i) for i in range(len(want))] == want and len(before) == len(want):
        return None
    data["names"] = want
    ds.yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return _fx("data.yaml", None, "rename_classes",
               str([before[k] for k in sorted(before)]), str(want))


def _fix_label_file(ds, stem, by_line):
    """Sửa một file nhãn: by_line = {số dòng gốc: {code, ...}}. Đọc 1 lần, sửa trong bộ nhớ, ghi 1 lần."""
    path = ds.label_path(stem)
    if not path.exists():                       # ảnh đã bị xoá / chưa có nhãn: bỏ qua êm
        return []
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()   # GIỮ dòng trống -> số dòng = số dòng gốc
    fixes, out = [], []
    for ln, text in enumerate(raw, 1):
        codes = by_line.get(ln)
        if not codes or not text.strip():       # dòng không lỗi (hoặc issue cũ trỏ vào dòng trống): giữ nguyên
            out.append(text)
            continue
        if codes & DROP_CODES:                  # xoá dòng thắng mọi cách sửa khác -> chỉ 1 fix
            fixes.append(_fx(stem, ln, "drop_line", text, ""))
            continue
        if codes & CLIP_CODES:
            try:
                new = _clip(text)
            except ValueError:                  # phòng hờ: dòng không parse được thì không thể cắt
                new = None
            if new is None:                     # nằm hẳn ngoài ảnh -> xoá, không ghi box rỗng/âm
                fixes.append(_fx(stem, ln, "drop_line", text, ""))
                continue
            fixes.append(_fx(stem, ln, "clip_box", text, new))
            out.append(new)
            continue
        out.append(text)                        # lỗi khác (không thuộc phạm vi autofix): giữ nguyên
    if fixes:
        path.write_text("".join(t + "\n" for t in out), encoding="utf-8")
    return fixes


def fix(ds, issues, cfg):
    errors = [i for i in issues if i.severity == ERROR]     # không bao giờ động tới WARN
    fixes = []

    # 1) Xoá ảnh trước: issue dòng của ảnh đã xoá sẽ tự bị bỏ qua ở bước 4
    removed = set()
    for stem in dict.fromkeys(i.image for i in errors if i.code in REMOVE_CODES):   # giữ thứ tự, bỏ trùng
        removed.add(stem)
        code = next(i.code for i in errors if i.image == stem and i.code in REMOVE_CODES)
        gone = _remove_image(ds, stem)
        if gone:
            fixes.append(_fx(stem, None, "remove_image", ", ".join(gone), f"(đã xoá: {code})"))

    # 2) Tên lớp trong data.yaml (cấp dataset, làm 1 lần)
    if any(i.code == "BAD_CLASS_NAME" for i in errors):
        f = _rename_classes(ds, cfg)
        if f:
            fixes.append(f)

    # 3) Thiếu file nhãn: KHÔNG bịa nhãn, KHÔNG tạo file rỗng -> chuyển người gán
    for stem in dict.fromkeys(i.image for i in errors if i.code == "MISSING_LABEL_FILE"):
        if stem not in removed and not ds.label_path(stem).exists():
            fixes.append(_fx(stem, None, "needs_label", "(không có file)", "chuyển người gán"))

    # 4) Lỗi theo dòng: gom theo file -> {dòng gốc: {code}}, rồi sửa mỗi file đúng 1 lần
    per_file = defaultdict(lambda: defaultdict(set))
    for i in errors:
        if i.line is not None and i.code in DROP_CODES | CLIP_CODES and i.image not in removed:
            per_file[i.image][i.line].add(i.code)
    for stem in sorted(per_file):
        fixes.extend(_fix_label_file(ds, stem, per_file[stem]))

    return fixes
