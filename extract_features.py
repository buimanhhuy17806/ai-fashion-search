"""
extract_features.py - Trích xuất vector CLIP cho toàn bộ sản phẩm (chạy 1 lần, offline).

Đặt file này ở thư mục gốc project (cạnh manage.py) rồi chạy:
    python extract_features.py

Cải tiến so với bản cũ:
  1. ID và vector luôn đi cùng nhau: ảnh lỗi/thiếu bị bỏ cả id lẫn vector, nên không thể lệch hàng.
  2. Xử lý theo lô (batch), có thanh tiến trình, ảnh hỏng chỉ bị bỏ qua và được ghi lại.
  3. Dùng đúng cách mã hóa như ai_engine.py (vision_model -> visual_projection, chuẩn hóa L2).
  4. Mã hóa thêm "vector mô tả" của từng sản phẩm (từ cột loại, màu, giới tính) để tìm kiếm chữ chính xác hơn.
  5. Sao lưu file pkl cũ, ghi file mới an toàn, và tự đánh giá chất lượng ngay sau khi chạy.
"""
import csv
import os
import pickle
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ==========================================
# CẤU HÌNH (sửa cho khớp project của bạn)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIP_DIR = os.path.join(BASE_DIR, "clip_model")
CSV_PATH = os.path.join(BASE_DIR, "data", "styles.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")        # thư mục chứa ảnh sản phẩm
OUT_PATH = os.path.join(BASE_DIR, "data", "vector_database.pkl")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")             # ảnh tên là <id>.<đuôi>
COL_ID, COL_TYPE, COL_COLOUR, COL_GENDER = "id", "articleType", "baseColour", "gender"
COL_SUB, COL_USAGE = "subCategory", "usage"          # Topwear/Bottomwear/Footwear...; Casual/Formal/Sports...
BATCH_SIZE = 32
PROMPT = "a photo of {}"                                    # phải giống PROMPT_TEMPLATE trong ai_engine.py

device = "cuda" if torch.cuda.is_available() else "cpu"

# Tên loại sản phẩm trong CSV -> cách nói tự nhiên hơn cho CLIP
TYPE_FIX = {"tshirts": "t-shirt", "tops": "top", "shirts": "shirt", "jeans": "jeans",
            "casual shoes": "casual shoes", "flip flops": "flip flops"}


def describe(row):
    """Câu mô tả sản phẩm từ các cột nhãn, ví dụ: 'navy blue shirt for men'."""
    atype = (row.get(COL_TYPE) or "").strip().lower()
    atype = TYPE_FIX.get(atype, atype)
    colour = (row.get(COL_COLOUR) or "").strip().lower()
    gender = (row.get(COL_GENDER) or "").strip().lower()
    usage = (row.get(COL_USAGE) or "").strip().lower()
    text = " ".join(p for p in (colour, atype) if p)
    if gender and gender != "unisex":
        text += f" for {gender}"
    if usage and usage != "casual":          # "casual" quá phổ biến nên bỏ, chỉ giữ formal/sports/ethnic...
        text += f", {usage} wear"
    return text or "fashion product"


# ==========================================
# ĐỌC DANH SÁCH SẢN PHẨM
# ==========================================
def find_image(pid):
    for ext in IMAGE_EXTS:
        path = os.path.join(IMAGE_DIR, f"{pid}{ext}")
        if os.path.exists(path):
            return path
    return None


def load_products():
    if not os.path.isdir(IMAGE_DIR):
        raise SystemExit(f"[LỖI] Không thấy thư mục ảnh: {IMAGE_DIR}. Hãy sửa IMAGE_DIR ở đầu file.")
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or COL_ID not in rows[0]:
        raise SystemExit(f"[LỖI] CSV không có cột '{COL_ID}'. Các cột hiện có: {list(rows[0]) if rows else []}")

    products, missing = [], []
    for r in rows:
        pid = str(r[COL_ID]).strip()
        path = find_image(pid)
        if path is None:
            missing.append(pid)
        else:
            products.append({"id": pid, "path": path, "row": r})
    print(f"[INFO] CSV có {len(rows)} dòng, tìm thấy ảnh cho {len(products)}, thiếu ảnh {len(missing)}.")
    if missing:
        print("       Ví dụ id thiếu ảnh:", missing[:10])
    return products


# ==========================================
# MÃ HÓA (giống hệt ai_engine.py)
# ==========================================
@torch.no_grad()
def encode_images(model, processor, pil_images):
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    pooled = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
    return F.normalize(model.visual_projection(pooled), dim=-1).cpu()


@torch.no_grad()
def encode_texts(model, processor, texts):
    inputs = processor(text=[PROMPT.format(t) for t in texts], return_tensors="pt",
                       padding=True, truncation=True).to(device)
    pooled = model.text_model(**inputs).pooler_output
    return F.normalize(model.text_projection(pooled), dim=-1).cpu()


def progress(iterable, total):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, unit="lô")
    except ImportError:
        return iterable


# ==========================================
# TỰ ĐÁNH GIÁ
# ==========================================
def self_check(ids, emb, products, n=300, k=8):
    """Lấy n sản phẩm làm truy vấn bằng chính ảnh của nó. Phát hiện lệch id và đo độ chính xác."""
    n = min(n, len(ids))
    sample = random.sample(range(len(ids)), n)
    sims = emb[sample] @ emb.T                                   # (n, N)
    self_hit = (sims.argmax(dim=1) == torch.tensor(sample)).float().mean().item()

    sims[torch.arange(n), torch.tensor(sample)] = -1.0           # bỏ chính nó
    top = sims.topk(k, dim=1).indices.tolist()
    same_sub = same_type = same_colour = 0
    for q, hits in zip(sample, top):
        rq = products[q]["row"]
        for h in hits:
            rh = products[h]["row"]
            same_sub += rq.get(COL_SUB) == rh.get(COL_SUB)
            same_type += rq.get(COL_TYPE) == rh.get(COL_TYPE)
            same_colour += rq.get(COL_COLOUR) == rh.get(COL_COLOUR)
    total = n * k
    print("\n===== TỰ ĐÁNH GIÁ (tìm bằng ảnh, %d truy vấn) =====" % n)
    print(f"Sản phẩm tự tìm ra chính nó ở hạng 1 : {self_hit * 100:.1f}%  (nên gần 100%, thấp là dấu hiệu lệch id)")
    print(f"Precision@{k} cùng nhóm (subCategory) : {same_sub / total * 100:.1f}%  (thấp nghĩa là áo lẫn giày, quần)")
    print(f"Precision@{k} cùng loại (articleType) : {same_type / total * 100:.1f}%")
    print(f"Precision@{k} cùng màu (baseColour)   : {same_colour / total * 100:.1f}%")


# ==========================================
# CHẠY CHÍNH
# ==========================================
def main():
    products = load_products()
    if not products:
        raise SystemExit("[LỖI] Không có sản phẩm nào để xử lý.")

    print(f"[INFO] Đang tải CLIP từ {CLIP_DIR} trên {device}...")
    model = CLIPModel.from_pretrained(CLIP_DIR, local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_DIR, local_files_only=True)

    ids, chunks, kept, bad = [], [], [], []
    t0 = time.time()
    batches = [products[i:i + BATCH_SIZE] for i in range(0, len(products), BATCH_SIZE)]
    for batch in progress(batches, len(batches)):
        images, good = [], []
        for p in batch:
            try:
                with Image.open(p["path"]) as im:
                    images.append(im.convert("RGB"))
                good.append(p)
            except Exception as e:                               # ảnh hỏng: bỏ cả id lẫn vector
                bad.append((p["id"], str(e)))
        if not images:
            continue
        chunks.append(encode_images(model, processor, images))
        ids.extend(p["id"] for p in good)
        kept.extend(good)

    emb = torch.cat(chunks)
    assert emb.shape[0] == len(ids) == len(kept), "ID và vector không khớp số lượng!"
    print(f"[INFO] Xong {len(ids)} ảnh trong {time.time() - t0:.0f}s, bỏ qua {len(bad)} ảnh lỗi.")
    if bad:
        print("       Ví dụ ảnh lỗi:", bad[:5])

    print("[INFO] Đang mã hóa mô tả chữ của sản phẩm...")
    descs = [describe(p["row"]) for p in kept]
    text_emb = torch.cat([encode_texts(model, processor, descs[i:i + 256]) for i in range(0, len(descs), 256)])

    # Sao lưu bản cũ rồi ghi bản mới (ghi ra file tạm, xong mới thay thế)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if os.path.exists(OUT_PATH):
        shutil.copy2(OUT_PATH, OUT_PATH + ".bak")
        print(f"[INFO] Đã sao lưu bản cũ: {OUT_PATH}.bak")
    data = {
        "ids": ids,                                              # ai_engine.py đọc 2 khóa này
        "embeddings": emb.numpy().astype(np.float32),
        "text_embeddings": text_emb.numpy().astype(np.float32),  # vector mô tả, dùng cho bước tìm chữ nâng cao
        "descriptions": descs,
        "model": "openai/clip-vit-base-patch32",
    }
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, OUT_PATH)
    print(f"[INFO] Đã lưu {len(ids)} vector vào {OUT_PATH}")

    self_check(ids, emb, kept)


if __name__ == "__main__":
    main()