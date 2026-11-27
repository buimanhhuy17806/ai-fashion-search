"""
views.py - Xử lý yêu cầu từ trình duyệt và trả về trang web.

Luồng chính:  Trình duyệt -> view -> (ai_engine tìm) -> (catalog gắn tên, giá) -> template
"""
import base64
import io
import logging
from urllib.parse import urlencode

from django.conf import settings
from django.core.paginator import Paginator
from django.core.files.base import ContentFile
from django.http import Http404
from django.shortcuts import render
from PIL import Image, ImageOps, UnidentifiedImageError

from . import ai_engine, catalog, categories
from .vi_fashion_translate import translate_query

logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = 5  # Giới hạn dung lượng ảnh tải lên
PAGE_SIZE = 24     # Số sản phẩm mỗi trang


# ==========================================
# HÀM PHỤ
# ==========================================
def make_preview(pil_image, max_side=300):
    """Thu nhỏ ảnh người dùng tải lên và đổi thành chuỗi Base64 để hiện thẳng lên web (không cần lưu file)."""
    thumb = pil_image.copy()
    thumb.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _load_image(upload):
    """Đọc file tải lên thành ảnh PIL. Trả về (ảnh, thông báo lỗi)."""
    if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"Ảnh quá lớn (tối đa {MAX_UPLOAD_MB}MB)."
    try:
        return ImageOps.exif_transpose(Image.open(upload)).convert("RGB"), ""  # sửa ảnh điện thoại bị xoay
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return None, "File tải lên không phải ảnh hợp lệ."


def _translate(query):
    """Dịch từ khóa Việt -> Anh. Trả về (câu tiếng Anh, danh mục, các variants)."""
    if not getattr(settings, "USE_TRANSLATOR", True):
        logger.info("KHÔNG dịch, đưa thẳng vào CLIP: %r", query)
        return query, None, [query]
    tq = translate_query(query, mt=lambda segs: [ai_engine.to_english(s) for s in segs])
    logger.info("Dịch: %r -> %r | danh mục: %s | variants: %s", query, tq["en"], tq["category"], tq["variants"])
    return tq["en"], tq["category"], tq["variants"]


def _decode_cropped(data_url):
    """Giải mã ảnh cắt dạng Base64 (từ Cropper.js) thành file trong RAM. Lỗi thì trả về None."""
    try:
        header, imgstr = data_url.split(";base64,", 1)
        return ContentFile(base64.b64decode(imgstr), name=f"cropped.{header.split('/')[-1] or 'jpeg'}")
    except Exception as e:
        logger.warning("Không giải mã được ảnh cắt: %s", e)
        return None


def _search_by_image(upload):
    """Tìm bằng ảnh. Trả về (kết quả, thông báo, ảnh xem trước)."""
    pil_image, error = _load_image(upload)
    if error:
        return [], error, ""
    preview = make_preview(pil_image)
    raw = ai_engine.rank(ai_engine.encode_image(pil_image), ai_engine.IMAGE_MIN_SCORE)
    if not raw:
        return [], "Không tìm thấy sản phẩm tương đồng.", preview
    return catalog.get_products_from_results(raw), "", preview


def _search_by_text(query):
    """
    Tìm bằng chữ. Trả về (kết quả, thông báo, câu tiếng Anh đã dịch).
    Thứ tự: khớp tên -> dịch -> CLIP (lọc theo danh mục nếu nhận ra, rỗng thì tìm lại không lọc).
    """
    matched = catalog.search_by_name(query)
    if matched:
        return [catalog.get_product(pid) for pid in matched], "", ""

    english_query, category, variants = _translate(query)
    if not english_query.strip():
        return [], "Chưa hiểu từ khóa này, hãy thử mô tả cụ thể hơn (ví dụ: áo thun trắng).", ""

    features = ai_engine.encode_texts(variants)  # nhiều variants -> trung bình vector
    allowed = categories.ids_for_category(category)
    raw = ai_engine.rank(features, ai_engine.TEXT_MIN_SCORE, allowed_ids=allowed) if allowed else []
    if not raw:
        raw = ai_engine.rank(features, ai_engine.TEXT_MIN_SCORE)

    if not raw:
        return [], "Không tìm thấy sản phẩm phù hợp với từ khóa này.", english_query
    return catalog.get_products_from_results(raw), "", english_query


def _search_hybrid(upload, query):
    """Tìm lai ảnh + chữ. Trả về (kết quả, thông báo, ảnh xem trước, câu tiếng Anh đã dịch)."""
    pil_image, error = _load_image(upload)
    if error:
        return [], error, "", ""
    preview = make_preview(pil_image)

    english_query, category, variants = _translate(query)
    if not english_query.strip():
        return [], "Chưa hiểu từ khóa này, hãy thử mô tả cụ thể hơn.", preview, ""

    image_features = ai_engine.encode_image(pil_image)
    text_features = ai_engine.encode_texts(variants)  # encode_texts đã tự thêm "a photo of"

    allowed = categories.ids_for_category(category)
    raw = ai_engine.rank_hybrid(image_features, text_features, allowed_ids=allowed) if allowed else []
    if not raw:
        raw = ai_engine.rank_hybrid(image_features, text_features)

    return catalog.get_products_from_results(raw), "", preview, english_query


# ==========================================
# CÁC TRANG
# ==========================================
def home(request):
    """Trang chủ cửa hàng: ô danh mục + 12 sản phẩm ngẫu nhiên."""
    products = [catalog.get_product(pid) for pid in catalog.get_random_product_ids(12)]
    tiles = []
    for t in categories.tiles():
        t["product"] = catalog.get_product(t["sample_id"])
        tiles.append(t)
    return render(request, "home.html", {
        "products": products, "category_tiles": tiles, "total_products": categories.total(),
    })


def category(request):
    """Trang danh mục: lọc theo nhóm (g), loại (t), màu (colour) và phân trang."""
    g = request.GET.get("g", "")
    t = request.GET.get("t", "")
    colour = request.GET.get("colour", "")

    data = categories.browse(g, t, colour)
    paginator = Paginator(data["ids"], PAGE_SIZE)
    page = paginator.get_page(request.GET.get("page"))

    context = {
        "group": data["group"], "group_key": g,
        "type_key": t, "type_label": categories.type_label(t) if t else "",
        "colour_key": colour, "colour_label": categories.colour_label(colour) if colour else "",
        "types": [{"label": categories.type_label(k), "count": n, "active": k == t,
                   "url": categories.url(g, "" if k == t else k, colour)} for k, n in data["types"]],
        "colours": [{"label": categories.colour_label(k), "hex": categories.colour_hex(k), "count": n,
                     "active": k == colour, "url": categories.url(g, t, "" if k == colour else k)}
                    for k, n in data["colours"]],
        "clear_type_url": categories.url(g, "", colour),
        "clear_colour_url": categories.url(g, t, ""),
        "group_url": categories.url(g),
        "products": [catalog.get_product(pid) for pid in page.object_list],
        "page": page, "total": paginator.count,
        "qs": urlencode({k: v for k, v in (("g", g), ("t", t), ("colour", colour)) if v}),
    }
    return render(request, "category.html", context)


def search(request):
    """
    Trang tìm kiếm. Form gửi bằng POST hoặc GET (chuyển trang) với:
        - "q"            : từ khóa chữ (tiếng Việt)
        - "image"        : ảnh tải lên file vật lý gốc
        - "cropped_image": ảnh đã cắt dạng chuỗi Base64 từ giao diện
    Có ảnh + chữ -> tìm lai; chỉ ảnh -> tìm bằng ảnh; chỉ chữ -> tìm bằng chữ.
    """
    context = {
        "query": "", "english_query": "",
        "message": "", "query_image": "", "search_mode": "text",
    }

    if not ai_engine.is_ready():
        context["message"] = "Chưa tải được mô hình AI hoặc cơ sở dữ liệu vector. Hãy kiểm tra thư mục clip_model và chạy extract_features.py."
        return render(request, "search.html", context)

    # 1. Lấy dữ liệu từ request
    query = request.POST.get("q", request.GET.get("q", "")).strip()
    upload = request.FILES.get("image")
    page_number = request.GET.get("page", 1)

    # 2. Ảnh đã cắt (Base64) được ưu tiên hơn file gốc
    cropped_b64 = request.POST.get("cropped_image")
    if cropped_b64:
        upload = _decode_cropped(cropped_b64) or upload

    context["query"] = query
    results = []

    # 3. Không có dữ liệu đầu vào
    if not query and upload is None:
        context["message"] = "Vui lòng nhập từ khóa hoặc tải ảnh lên để tìm kiếm."
        return render(request, "search.html", context)

    # 4. Chọn luồng tìm kiếm
    if upload is not None and query:
        context["search_mode"] = "hybrid"
        results, context["message"], context["query_image"], context["english_query"] = _search_hybrid(upload, query)
    elif upload is not None:
        context["search_mode"] = "image"
        results, context["message"], context["query_image"] = _search_by_image(upload)
    else:
        context["search_mode"] = "text"
        results, context["message"], context["english_query"] = _search_by_text(query)

    # 5. Phân trang
    if results:
        paginator = Paginator(results, PAGE_SIZE)
        context["page_obj"] = paginator.get_page(page_number)
        context["total_found"] = paginator.count
    else:
        context["page_obj"] = None
        context["total_found"] = 0

    return render(request, "search.html", context)


def product_detail(request, product_id):
    """Trang chi tiết: hiện sản phẩm + 8 sản phẩm giống nhất (dùng vector thật)."""
    if not catalog.product_exists(product_id):
        raise Http404("Không tìm thấy sản phẩm")
    product = catalog.get_product(product_id)
    similar = []
    if ai_engine.is_ready():
        similar = catalog.get_products_from_results(ai_engine.find_similar_to_product(product_id))
    return render(request, "product.html", {"product": product, "similar_products": similar})