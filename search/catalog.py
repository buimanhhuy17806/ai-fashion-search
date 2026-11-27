"""
catalog.py - Dữ liệu sản phẩm (tên, giá, đường dẫn ảnh).

Tách riêng khỏi AI để sau này thay "dữ liệu ảo" bằng Database thật chỉ cần sửa file này.

Thư mục ảnh và file CSV được TỰ DÒ trong thư mục data/ (không phụ thuộc tên cụ thể).
Muốn chỉ định tay, thêm vào settings.py (đều không bắt buộc):
    FASHION_IMAGE_DIR = BASE_DIR / "data" / "ten_thu_muc_anh"
    FASHION_CSV       = BASE_DIR / "data" / "ten_file.csv"
    FASHION_IMAGE_URL = "/media/ten_thu_muc_anh/"
"""
import logging
import os
import random
import re
import unicodedata
from functools import lru_cache

import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)

# ==========================================
# CẤU HÌNH
# ==========================================
DATA_DIR = os.path.join(settings.BASE_DIR, "data")
ID_COLUMN = "id"                    # Cột mã sản phẩm trong CSV
NAME_COLUMN = "productDisplayName"  # Cột tên sản phẩm trong CSV
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
NAME_SEARCH_LIMIT = 24              # Số kết quả tối đa khi khớp theo tên
PREFERRED_CSV = ("dataset_mini.csv", "styles.csv")


def _find_image_dir():
    """Thư mục chứa nhiều file ảnh nhất trong data/ (hoặc MEDIA_ROOT), trừ khi đã chỉ định trong settings."""
    configured = getattr(settings, "FASHION_IMAGE_DIR", None)
    if configured:
        return str(configured)
    roots = [DATA_DIR]
    media_root = getattr(settings, "MEDIA_ROOT", "")
    if media_root and str(media_root) not in roots:
        roots.append(str(media_root))
    best, best_n = None, 0
    for root in roots:
        for folder, _dirs, files in os.walk(root):
            n = sum(f.lower().endswith(IMAGE_EXTS) for f in files)
            if n > best_n:
                best, best_n = folder, n
    if best is None:
        logger.error("Không tìm thấy thư mục ảnh nào trong %s. Hãy đặt FASHION_IMAGE_DIR trong settings.py.", DATA_DIR)
        return os.path.join(DATA_DIR, "images")
    return best


def _find_csv():
    """File CSV có cột id + productDisplayName (ưu tiên dataset_mini.csv, styles.csv)."""
    configured = getattr(settings, "FASHION_CSV", None)
    if configured:
        return str(configured)
    if os.path.isdir(DATA_DIR):
        names = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv"))
        names.sort(key=lambda f: (f not in PREFERRED_CSV, f))
        for f in names:
            path = os.path.join(DATA_DIR, f)
            try:
                cols = pd.read_csv(path, nrows=0).columns
            except Exception:
                continue
            if ID_COLUMN in cols and NAME_COLUMN in cols:
                return path
    logger.error("Không tìm thấy file CSV có cột '%s' và '%s' trong %s.", ID_COLUMN, NAME_COLUMN, DATA_DIR)
    return os.path.join(DATA_DIR, "dataset_mini.csv")


def _image_url_prefix(image_dir):
    """URL để trình duyệt tải ảnh: suy ra từ MEDIA_ROOT / MEDIA_URL (cần cấu hình phục vụ media trong urls.py)."""
    configured = getattr(settings, "FASHION_IMAGE_URL", None)
    if configured:
        return configured if configured.endswith("/") else configured + "/"
    media_url = getattr(settings, "MEDIA_URL", "") or "/media/"
    media_root = str(getattr(settings, "MEDIA_ROOT", "") or "")
    if media_root:
        try:
            rel = os.path.relpath(image_dir, media_root).replace(os.sep, "/")
            if not rel.startswith(".."):
                return media_url + ("" if rel == "." else rel + "/")
        except ValueError:
            pass
    return f"{media_url}{os.path.basename(image_dir)}/"


IMAGE_DIR = _find_image_dir()
CSV_PATH = _find_csv()
IMAGE_URL_PREFIX = _image_url_prefix(IMAGE_DIR)
logger.info("Catalog: ảnh=%s | csv=%s | url=%s", IMAGE_DIR, CSV_PATH, IMAGE_URL_PREFIX)


# ==========================================
# ĐỌC CSV
# ==========================================
def _normalize_id(value):
    """'0108775015', 108775015, '108775015' -> cùng thành '108775015' (tránh lệch số 0 đầu)."""
    try:
        return str(int(value))
    except (ValueError, TypeError):
        return str(value).strip()


def _load_name_lookup():
    """Đọc CSV 1 lần, tạo từ điển {mã sản phẩm đã chuẩn hóa: tên}."""
    try:
        df = pd.read_csv(CSV_PATH, on_bad_lines="skip")  # styles.csv gốc có vài dòng thừa dấu phẩy
        return {
            _normalize_id(i): n
            for i, n in zip(df[ID_COLUMN], df[NAME_COLUMN])
            if isinstance(n, str) and n.strip()
        }
    except Exception as e:
        logger.warning("Không đọc được CSV %s (%s) -> dùng tên mặc định.", CSV_PATH, e)
        return {}


_names = _load_name_lookup()


@lru_cache(maxsize=1)
def _image_id_map():
    """{mã chuẩn hóa: tên file ảnh đầy đủ (kể cả đuôi)}, quét thư mục ảnh 1 lần."""
    try:
        files = os.listdir(IMAGE_DIR)
    except OSError as e:
        logger.error("Không đọc được thư mục ảnh %s: %s", IMAGE_DIR, e)
        return {}
    out = {}
    for f in files:
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXTS:
            out[_normalize_id(stem)] = f
    return out


# ==========================================
# TÌM THEO TÊN
# ==========================================
def _fold(text):
    """Chữ thường, bỏ dấu, chỉ giữ chữ-số -> danh sách từ."""
    s = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", s)


@lru_cache(maxsize=1)
def _name_index():
    """[(mã ảnh, chuỗi tên đã fold, tập từ, độ dài tên)] - chỉ gồm sản phẩm có ảnh (nếu đọc được thư mục ảnh)."""
    img_map = _image_id_map()
    index = []
    for norm_id, name in _names.items():
        if img_map and norm_id not in img_map:
            continue
        pid = os.path.splitext(img_map[norm_id])[0] if img_map else norm_id
        words = _fold(name)
        index.append((pid, " ".join(words), set(words), len(name)))
    return index


def search_by_name(query, limit=NAME_SEARCH_LIMIT):
    """
    Ưu tiên khớp tên khi người dùng gõ tên sản phẩm tiếng Anh, vd: 'Puma Men Red Slipper'.
    Điều kiện: query là ASCII (không có dấu tiếng Việt), >= 2 từ, và MỌI từ đều có trong tên.
    Các câu khác trả về [] để view chuyển sang tìm bằng AI.
    """
    if not query or not query.isascii():
        return []
    tokens = _fold(query)
    if len(tokens) < 2:
        return []

    phrase = " ".join(tokens)
    need = set(tokens)
    hits = [(0 if phrase in folded else 1, name_len, pid)
            for pid, folded, words, name_len in _name_index() if need <= words]
    hits.sort()
    return [pid for _, _, pid in hits[:limit]]


# ==========================================
# HÀM DÙNG CHUNG
# ==========================================
def has_image(product_id):
    """True nếu sản phẩm có file ảnh. Không đọc được thư mục ảnh thì coi như có (không chặn nhầm)."""
    img_map = _image_id_map()
    return not img_map or _normalize_id(product_id) in img_map


def product_exists(product_id):
    """True nếu sản phẩm hiển thị được. Không có dữ liệu nguồn nào (cấu hình sai) thì không chặn bằng 404."""
    norm_id = _normalize_id(product_id)
    img_map = _image_id_map()
    if img_map:
        return norm_id in img_map
    if _names:
        return norm_id in _names
    return True


def get_product(product_id, score=None):
    """
    Tạo thông tin 1 sản phẩm để đưa ra template.
    Giá là giá ảo nhưng CỐ ĐỊNH theo mã sản phẩm (reload trang không bị đổi giá).
    score=None -> template sẽ không hiện thanh "Độ tương đồng".
    """
    product_id = str(product_id)
    price = random.Random(product_id).randint(150, 500)
    filename = _image_id_map().get(_normalize_id(product_id), f"{product_id}.jpg")  # đuôi file thật

    return {
        "id": product_id,
        "name": _names.get(_normalize_id(product_id), f"Trang phục #{product_id}"),
        "image_url": f"{IMAGE_URL_PREFIX}{filename}",
        "price": f"{price}.000đ",
        "score": score,
    }


def get_products_from_results(ai_results):
    """Chuyển kết quả của AI [{"id", "score"}] thành danh sách sản phẩm đầy đủ (bỏ sản phẩm thiếu ảnh)."""
    return [get_product(r["id"], score=r["score"]) for r in ai_results if has_image(r["id"])]


def get_random_product_ids(n=12):
    """Lấy ngẫu nhiên n mã sản phẩm từ các ảnh có thật (danh sách ảnh được cache)."""
    ids = [os.path.splitext(f)[0] for f in _image_id_map().values()]
    return random.sample(ids, min(n, len(ids)))