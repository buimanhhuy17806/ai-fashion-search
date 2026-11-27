"""
categories.py - Danh mục sản phẩm cho giao diện cửa hàng (đọc từ CSV, chỉ lấy sản phẩm có trong vector DB).
"""
import csv
import os
from collections import Counter
from urllib.parse import urlencode

from django.conf import settings
from django.urls import reverse

import logging

from . import ai_engine, catalog

logger = logging.getLogger(__name__)

CSV_CANDIDATES = [os.path.join(settings.BASE_DIR, "data", n) for n in ("dataset_mini.csv", "styles.csv")]
COL_ID, COL_GENDER, COL_MASTER, COL_TYPE, COL_COLOUR = "id", "gender", "masterCategory", "articleType", "baseColour"

# Nhóm hiển thị trên thanh menu
GROUPS = [
    {"key": "men", "label": "Nam", "genders": ["Men"]},
    {"key": "women", "label": "Nữ", "genders": ["Women"]},
    {"key": "kids", "label": "Trẻ em", "genders": ["Boys", "Girls"]},
    {"key": "footwear", "label": "Giày dép", "masters": ["Footwear"]},
    {"key": "accessories", "label": "Phụ kiện", "masters": ["Accessories"]},
]
GROUP_BY_KEY = {g["key"]: g for g in GROUPS}

# Tên hiển thị tiếng Việt (loại nào chưa có thì hiện tên gốc)
TYPE_VI = {
    "Tshirts": "Áo thun", "Shirts": "Áo sơ mi", "Tops": "Áo kiểu", "Sweatshirts": "Áo nỉ",
    "Sweaters": "Áo len", "Jackets": "Áo khoác", "Kurtas": "Áo kurta", "Jeans": "Quần jean",
    "Trousers": "Quần dài", "Shorts": "Quần short", "Track Pants": "Quần thể thao",
    "Leggings": "Quần legging", "Capris": "Quần lửng", "Dresses": "Váy đầm", "Skirts": "Chân váy",
    "Casual Shoes": "Giày thường", "Sports Shoes": "Giày thể thao", "Formal Shoes": "Giày tây",
    "Heels": "Giày cao gót", "Flats": "Giày bệt", "Sandals": "Sandal", "Flip Flops": "Dép xỏ ngón",
    "Watches": "Đồng hồ", "Handbags": "Túi xách", "Backpacks": "Ba lô", "Wallets": "Ví",
    "Belts": "Thắt lưng", "Sunglasses": "Kính mát", "Caps": "Mũ lưỡi trai", "Socks": "Tất",
    "Ties": "Cà vạt", "Scarves": "Khăn quàng", "Earrings": "Khuyên tai", "Bra": "Áo ngực",
    "Briefs": "Quần lót", "Innerwear Vests": "Áo lót", "Perfume and Body Mist": "Nước hoa",
}
COLOUR_VI = {   # tên màu -> (tên tiếng Việt, mã màu để vẽ chấm tròn)
    "Black": ("Đen", "#111827"), "White": ("Trắng", "#FFFFFF"), "Blue": ("Xanh dương", "#2563EB"),
    "Navy Blue": ("Xanh navy", "#1E3A5F"), "Red": ("Đỏ", "#DC2626"), "Green": ("Xanh lá", "#16A34A"),
    "Grey": ("Xám", "#9CA3AF"), "Brown": ("Nâu", "#78350F"), "Pink": ("Hồng", "#F472B6"),
    "Yellow": ("Vàng", "#FACC15"), "Purple": ("Tím", "#7C3AED"), "Orange": ("Cam", "#F97316"),
    "Beige": ("Be", "#E7D8BF"), "Cream": ("Kem", "#FFF1D6"), "Maroon": ("Đỏ đô", "#7F1D1D"),
    "Olive": ("Ô liu", "#6B7A2A"), "Silver": ("Bạc", "#C0C4CC"), "Gold": ("Vàng kim", "#D4AF37"),
    "Multi": ("Nhiều màu", "linear-gradient(135deg,#F472B6,#FACC15,#2563EB)"),
}

_ITEMS = None
_NAV = None


def type_label(t):
    return TYPE_VI.get(t, t)


def colour_label(c):
    return COLOUR_VI.get(c, (c, "#D1D5DB"))[0]


def colour_hex(c):
    return COLOUR_VI.get(c, (c, "#D1D5DB"))[1]


def url(g="", t="", colour=""):
    """Đường dẫn trang danh mục với các bộ lọc."""
    params = {k: v for k, v in (("g", g), ("t", t), ("colour", colour)) if v}
    return reverse("category") + ("?" + urlencode(params) if params else "")


def _items():
    global _ITEMS
    if _ITEMS is not None:
        return _ITEMS
    path = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
    items = []
    if path:
        allowed = set(ai_engine.db_ids) if ai_engine.is_ready() else None
        skipped = 0
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                pid = (r.get(COL_ID) or "").strip()
                if not pid or (allowed is not None and pid not in allowed):
                    continue
                if not catalog.has_image(pid):      # bỏ sản phẩm không có ảnh trong images_mini
                    skipped += 1
                    continue
                items.append({
                    "id": pid,
                    "gender": (r.get(COL_GENDER) or "").strip(),
                    "master": (r.get(COL_MASTER) or "").strip(),
                    "type": (r.get(COL_TYPE) or "").strip(),
                    "colour": (r.get(COL_COLOUR) or "").strip(),
                })
    if path and skipped:
        logger.warning("Bỏ qua %d sản phẩm có trong CSV/vector DB nhưng thiếu ảnh.", skipped)
    _ITEMS = items
    return items


def _in_group(it, group):
    if not group:
        return True
    return (not group.get("genders") or it["gender"] in group["genders"]) and \
           (not group.get("masters") or it["master"] in group["masters"])


def nav_groups():
    """Dữ liệu cho thanh menu: mỗi nhóm kèm các loại sản phẩm nhiều nhất."""
    global _NAV
    if _NAV is None:
        nav = []
        for g in GROUPS:
            members = [it for it in _items() if _in_group(it, g)]
            if not members:
                continue
            top = Counter(it["type"] for it in members if it["type"]).most_common(8)
            nav.append({
                "key": g["key"], "label": g["label"], "url": url(g["key"]),
                "children": [{"label": type_label(t), "count": n, "url": url(g["key"], t)} for t, n in top],
            })
        _NAV = nav
    return _NAV


def tiles():
    """Ô danh mục trên trang chủ: mỗi nhóm có số sản phẩm và một sản phẩm mẫu làm ảnh."""
    out = []
    for g in GROUPS:
        members = [it for it in _items() if _in_group(it, g)]
        if members:
            out.append({"key": g["key"], "label": g["label"], "url": url(g["key"]),
                        "count": len(members), "sample_id": members[len(members) // 3]["id"]})
    return out


def browse(g_key="", t="", colour=""):
    """Lọc sản phẩm theo nhóm, loại, màu; kèm số lượng cho từng bộ lọc."""
    group = GROUP_BY_KEY.get(g_key)
    base = [it for it in _items() if _in_group(it, group)]
    ids = [it["id"] for it in base if (not t or it["type"] == t) and (not colour or it["colour"] == colour)]
    types = Counter(it["type"] for it in base if it["type"] and (not colour or it["colour"] == colour))
    colours = Counter(it["colour"] for it in base if it["colour"] and (not t or it["type"] == t))
    return {"group": group, "ids": ids, "types": types.most_common(), "colours": colours.most_common()}

# ==========================================
# NỐI VỚI BỘ DỊCH TRUY VẤN (vi_fashion_translate)
# ==========================================
# Nhãn "category" mà translate_query trả về -> các articleType trong CSV
VI_CATEGORY_TYPES = {
    "Áo": {"Tshirts", "Shirts", "Tops", "Sweatshirts", "Sweaters", "Jackets", "Kurtas", "Waistcoat",
           "Blazers", "Rain Jacket", "Tunics", "Shrug"},
    "Quần": {"Jeans", "Trousers", "Shorts", "Track Pants", "Leggings", "Capris", "Jeggings",
             "Tracksuits", "Patiala", "Churidar", "Salwar"},
    "Giày": {"Casual Shoes", "Sports Shoes", "Formal Shoes", "Heels", "Flats", "Sports Sandals"},
    "Dép": {"Sandals", "Flip Flops", "Sports Sandals"},
}
_CATEGORY_IDS = {}


def total():
    """Tổng số sản phẩm đang có trong kho (hiện ở trang chủ)."""
    return len(_items())


def ids_for_category(label):
    """Tập id thuộc nhãn danh mục ('Áo', 'Quần', 'Giày', 'Dép'); None nếu nhãn không xác định."""
    types = VI_CATEGORY_TYPES.get(label)
    if not types:
        return None
    if label not in _CATEGORY_IDS:
        _CATEGORY_IDS[label] = {it["id"] for it in _items() if it["type"] in types}
    return _CATEGORY_IDS[label]