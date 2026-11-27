"""
ai_engine.py - "Bộ não AI" của hệ thống (Phiên bản chạy Offline/Local)

Chịu trách nhiệm:
    1. Tải mô hình CLIP và mô hình dịch Việt -> Anh từ ổ cứng cục bộ (không cần Internet).
    2. Tải cơ sở dữ liệu vector (file .pkl) vào RAM.
    3. Mã hóa chữ / ảnh thành vector và xếp hạng sản phẩm giống nhất.

Nếu thiếu model hoặc vector DB, module vẫn import được; is_ready() sẽ trả về False
để view hiện thông báo thay vì văng lỗi NameError.
"""
import logging
import os
import pickle
import re
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from django.conf import settings
from transformers import CLIPModel, CLIPProcessor, MarianMTModel, MarianTokenizer
from PIL import Image

logger = logging.getLogger(__name__)

# ==========================================
# 0. CẤU HÌNH ĐƯỜNG DẪN LOCAL & THAM SỐ
# ==========================================
CLIP_MODEL_PATH = os.path.join(settings.BASE_DIR, "clip_model")
TRANSLATOR_PATH = os.path.join(settings.BASE_DIR, "translator_model")
VECTOR_DB_PATH = os.path.join(settings.BASE_DIR, "data", "vector_database.pkl")

TOP_K = 20                         # Số sản phẩm tối đa trả về
TEXT_MIN_SCORE = 0.15              # Tìm bằng chữ: dưới ngưỡng này bị loại (thang 0-1)
IMAGE_MIN_SCORE = 0.15             # Tìm bằng ảnh: dưới ngưỡng này bị loại
HYBRID_ALPHA = 0.5                 # Tìm lai: 1.0 = chỉ theo ảnh, 0.0 = chỉ theo chữ
PROMPT_TEMPLATE = "a photo of {}"  # Tiền tố giúp CLIP hiểu ngữ cảnh tốt hơn

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. TẢI MÔ HÌNH (Chỉ chạy 1 lần khi server khởi động)
# ==========================================
clip_model = clip_processor = None
translator_tokenizer = translator_model = None

logger.info("Đang tải CLIP Model (Offline) trên %s...", device)
try:
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
except Exception as e:
    clip_model = clip_processor = None
    logger.error("Không tải được CLIP tại %s (%s). Hãy chạy download_models.py trước!", CLIP_MODEL_PATH, e)

logger.info("Đang tải AI dịch Việt -> Anh (Offline)...")
try:
    translator_tokenizer = MarianTokenizer.from_pretrained(TRANSLATOR_PATH)
    translator_model = MarianMTModel.from_pretrained(TRANSLATOR_PATH).to(device).eval()
except Exception as e:
    translator_tokenizer = translator_model = None
    logger.error("Không tải được AI dịch tại %s (%s). Sẽ dùng từ điển, đoạn lạ giữ nguyên.", TRANSLATOR_PATH, e)


# ==========================================
# 2. CƠ SỞ DỮ LIỆU VECTOR
# ==========================================
def _load_vector_db(path):
    """
    Đọc file .pkl chứa vector của toàn bộ ảnh sản phẩm.
    Hỗ trợ 2 định dạng:
        - Mới: {"ids": [...], "embeddings": ma_trận_numpy}
        - Cũ : {id: vector, id: vector, ...}
    Trả về: (danh sách id, ma trận tensor đã chuẩn hóa L2 kích thước (N, 512)).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "ids" in data and "embeddings" in data:
        ids = [str(i) for i in data["ids"]]
        matrix = np.asarray(data["embeddings"], dtype=np.float32)
    else:
        ids = [str(k) for k in data.keys()]
        matrix = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in data.values()])

    # Chuẩn hóa L2: để "tích vô hướng" = "cosine similarity"
    matrix = F.normalize(torch.from_numpy(matrix), dim=-1)
    return ids, matrix


try:
    db_ids, db_matrix = _load_vector_db(VECTOR_DB_PATH)
    logger.info("Đã tải %d vector, shape = %s", len(db_ids), tuple(db_matrix.shape))
except Exception as e:
    logger.error("Không tải được vector database: %s", e)
    db_ids, db_matrix = [], None

# Tra nhanh: id sản phẩm -> vị trí hàng trong ma trận (dùng cho "sản phẩm tương tự")
_id_to_index = {pid: i for i, pid in enumerate(db_ids)}


def is_ready():
    """True nếu CLIP và vector database đều đã tải thành công."""
    return db_matrix is not None and clip_model is not None and clip_processor is not None


# ==========================================
# 3. MÃ HÓA (CHỮ / ẢNH -> VECTOR 512 CHIỀU)
# ==========================================
_VI_CHARS = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)


@lru_cache(maxsize=512)  # Gõ lại từ khóa cũ thì không phải dịch lại
def translate_vi_to_en(text):
    """Dịch câu tiếng Việt sang tiếng Anh. Nếu lỗi / chưa có model dịch thì trả về nguyên văn."""
    if translator_model is None:
        return text
    try:
        tokens = translator_tokenizer(text, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = translator_model.generate(**tokens, num_beams=4, max_new_tokens=64)
        return translator_tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        logger.warning("Dịch lỗi, dùng nguyên văn: %s", e)
        return text


def to_english(text):
    """Chỉ dịch khi câu có chữ tiếng Việt có dấu; câu tiếng Anh giữ nguyên."""
    return translate_vi_to_en(text) if _VI_CHARS.search(text) else text


@torch.no_grad()
def encode_texts(texts):
    """
    Một hoặc nhiều câu (tiếng Anh) -> MỘT vector 512 chiều đã chuẩn hóa, shape (1, 512).
    Nhiều câu (vd: các 'variants' của cùng 1 loại sản phẩm) sẽ được lấy trung bình.
    Hàm này tự thêm tiền tố "a photo of", nên nơi gọi KHÔNG cần thêm nữa.
    """
    if isinstance(texts, str):
        texts = [texts]
    inputs = clip_processor(
        text=[PROMPT_TEMPLATE.format(t) for t in texts],
        return_tensors="pt", padding=True, truncation=True,
    ).to(device)
    pooled = clip_model.text_model(**inputs).pooler_output
    feats = F.normalize(clip_model.text_projection(pooled), dim=-1)
    return F.normalize(feats.mean(dim=0, keepdim=True), dim=-1).cpu()


def encode_text(text):
    """Chữ (tiếng Anh) -> vector 512 chiều đã chuẩn hóa, shape (1, 512)."""
    return encode_texts([text])


@torch.no_grad()
def encode_image(pil_image):
    """Ảnh PIL -> vector 512 chiều đã chuẩn hóa, shape (1, 512)."""
    inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
    pooled = clip_model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
    feats = clip_model.visual_projection(pooled)
    return F.normalize(feats, dim=-1).cpu()


# ==========================================
# 4. XẾP HẠNG
# ==========================================
def _apply_filter(scores, allowed_ids):
    """Gán -inf cho sản phẩm ngoài tập allowed_ids để chúng không lọt top."""
    if not allowed_ids:
        return scores
    mask = torch.tensor([pid in allowed_ids for pid in db_ids], dtype=torch.bool)
    return scores.masked_fill(~mask, float("-inf"))


def rank(query_features, min_score, top_k=TOP_K, exclude_id=None, allowed_ids=None):
    """
    Xếp hạng theo cosine similarity với vector truy vấn.
        min_score  : ngưỡng loại kết quả (thang 0-1).
        exclude_id : loại 1 sản phẩm (dùng cho "sản phẩm tương tự").
        allowed_ids: chỉ xét các id này (lọc theo danh mục); None = không lọc.
    """
    scores = _apply_filter((query_features @ db_matrix.T).squeeze(0), allowed_ids)

    # Lấy dư 1 phần tử phòng khi phải loại exclude_id
    top = torch.topk(scores, k=min(top_k + 1, len(db_ids)))

    results = []
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        pid = db_ids[idx]
        if pid == exclude_id or score < min_score:
            continue
        results.append({"id": pid, "score": round(score * 100, 2)})
        if len(results) == top_k:
            break
    return results


def find_similar_to_product(product_id, top_k=TOP_K):
    """Gợi ý sản phẩm giống sản phẩm có sẵn: dùng luôn vector của nó trong DB."""
    idx = _id_to_index.get(str(product_id))
    if idx is None:
        return []  # sản phẩm không có trong vector database
    return rank(db_matrix[idx].unsqueeze(0), min_score=0.0, top_k=top_k, exclude_id=str(product_id))


def _minmax(scores):
    """Đưa điểm về thang 0-1 trên toàn kho, để điểm ảnh và điểm chữ so sánh được với nhau."""
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + 1e-8)


def rank_hybrid(image_features, text_features, alpha=HYBRID_ALPHA, top_k=TOP_K, allowed_ids=None):
    """
    Tìm lai: trộn độ giống theo ảnh và theo chữ.
        alpha: 1.0 = chỉ ảnh, 0.0 = chỉ chữ (chỉnh qua HYBRID_ALPHA).
    "score" trả về là điểm min-max đã trộn, KHÔNG phải cosine thật,
    nên không dùng ngưỡng TEXT_MIN_SCORE / IMAGE_MIN_SCORE.

    Vì sao min-max thay vì cộng vector: cosine ảnh-ảnh thường 0.7-1.0 còn chữ-ảnh chỉ
    khoảng 0.2-0.3, cộng thẳng vector thì ảnh lấn át chữ dù alpha là bao nhiêu.
    """
    s_img = (image_features @ db_matrix.T).squeeze(0)
    s_txt = (text_features @ db_matrix.T).squeeze(0)
    fused = _apply_filter(alpha * _minmax(s_img) + (1 - alpha) * _minmax(s_txt), allowed_ids)

    top = torch.topk(fused, k=min(top_k, len(db_ids)))
    return [
        {"id": db_ids[i], "score": round(v * 100, 2)}
        for v, i in zip(top.values.tolist(), top.indices.tolist())
        if v != float("-inf")
    ]