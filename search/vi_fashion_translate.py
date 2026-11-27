"""Dich truy van thoi trang Viet -> Anh theo kieu 'tu dien truoc, MT sau'.
- Cum tu trong GLOSSARY duoc doi thang sang tieng Anh (khop cum DAI NHAT truoc, khong phan biet dau).
- Chi nhung tu CHUA co trong tu dien moi dua cho mo hinh dich (opus-mt-vi-en...).
- Tra them 'category' (Ao/Quan/Giay/Dep) neu phat hien -> dung de loc metadata truoc khi tinh cosine.
Mo rong GLOSSARY bang cach them dong; khong can sua phan code ben duoi."""
import re
import unicodedata


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(s)).strip()


# "cum tieng Viet": ("English", category hoac None)
# "cụm tiếng Việt": ("English", "category" hoặc None)
GLOSSARY = {
    # ================= ÁO =================
    "áo thun": ("t-shirt", "Áo"), "áo phông": ("t-shirt", "Áo"), "áo sơ mi": ("button-up shirt", "Áo"),
    "áo polo": ("polo shirt", "Áo"), "áo thun polo": ("polo shirt", "Áo"), "áo ba lỗ": ("tank top", "Áo"), 
    "áo hai dây": ("camisole", "Áo"), "áo croptop": ("crop top", "Áo"), "áo len": ("sweater", "Áo"), 
    "áo len cổ lọ": ("turtleneck sweater", "Áo"), "áo hoodie": ("hoodie", "Áo"), "áo nỉ": ("sweatshirt", "Áo"), 
    "áo khoác gió": ("windbreaker jacket", "Áo"), "áo khoác jean": ("denim jacket", "Áo"), 
    "áo khoác": ("jacket", "Áo"), "áo cardigan": ("cardigan", "Áo"), "áo vest": ("blazer", "Áo"), 
    "áo blazer": ("blazer", "Áo"), "áo kiểu": ("blouse", "Áo"), "áo dài tay": ("long sleeve shirt", "Áo"), 
    "áo bomber": ("bomber jacket", "Áo"), "áo khoác bomber": ("bomber jacket", "Áo"),
    "áo ống": ("tube top", "Áo"), "áo trễ vai": ("off-the-shoulder top", "Áo"),
    "áo lệch vai": ("one-shoulder top", "Áo"), "áo cổ vuông": ("square neck top", "Áo"),
    "áo tay lỡ": ("half-sleeve top", "Áo"), "áo khoác da": ("leather jacket", "Áo"), 
    "áo phao": ("puffer jacket", "Áo"), "áo dạ": ("wool coat", "Áo"), 
    "áo măng tô": ("trench coat", "Áo"), "áo voan": ("chiffon blouse", "Áo"), 
    "áo ren": ("lace top", "Áo"), "áo chống nắng": ("sun protection jacket", "Áo"), 
    "áo ngực": ("bra", "Áo"), "áo lót": ("bra", "Áo"), "áo": ("shirt", "Áo"),

    # ================= QUẦN =================
    "quần jean": ("jeans", "Quần"), "quần jeans": ("jeans", "Quần"), "quần bò": ("jeans", "Quần"), 
    "quần short": ("shorts", "Quần"), "quần đùi": ("shorts", "Quần"), "quần đùi nam": ("men's shorts", "Quần"),
    "quần jogger": ("jogger pants", "Quần"), "quần kaki": ("khaki pants", "Quần"),
    "quần tây": ("dress trousers", "Quần"), "quần âu": ("dress trousers", "Quần"),
    "quần legging": ("leggings", "Quần"), "quần cargo": ("cargo pants", "Quần"), "quần túi hộp": ("cargo pants", "Quần"),
    "quần culottes": ("culottes", "Quần"), "quần ống rộng": ("wide-leg pants", "Quần"),
    "quần ống loe": ("flared pants", "Quần"), "quần ống đứng": ("straight-leg pants", "Quần"),
    "quần ống suông": ("straight-leg pants", "Quần"), "quần thể thao": ("sweatpants", "Quần"),
    "quần ống chuông": ("bell-bottom pants", "Quần"), "quần baggy": ("baggy pants", "Quần"),
    "quần short jean": ("denim shorts", "Quần"), "quần lót": ("panties", "Quần"),
    "quần bơi": ("swimming trunks", "Quần"), "quần rách": ("ripped pants", "Quần"),
    "quần dài": ("long pants", "Quần"), "quần": ("pants", "Quần"),

    # ================= VÁY / ĐẦM / ĐỒ BỘ =================
    "chân váy": ("skirt", None), "chân váy chữ a": ("a-line skirt", None), "chân váy xòe": ("flared skirt", None),
    "chân váy xếp ly": ("pleated skirt", None), "chân váy bút chì": ("pencil skirt", None),
    "chân váy đuôi cá": ("mermaid skirt", None), "chân váy midi": ("midi skirt", None),
    "chân váy mini": ("mini skirt", None),
    "váy": ("dress", None), "đầm": ("dress", None), "đầm maxi": ("maxi dress", None),
    "đầm dạ hội": ("evening gown", None), "dạ hội": ("evening", None),
    "đầm xòe": ("flared dress", None), "đầm suông": ("shift dress", None),
    "đầm body": ("bodycon dress", None), "đầm trễ vai": ("off-the-shoulder dress", None),
    "đầm hai dây": ("slip dress", None), "đầm babydoll": ("babydoll dress", None),
    "váy yếm": ("pinafore dress", None), "quần yếm": ("overalls", None),
    "đồ bộ": ("matching set", None), "đồ ngủ": ("pajamas", None), 
    "đồ bơi": ("swimwear", None), "đồ lót": ("underwear", None), "quần áo": ("clothing", None),

    # ================= GIÀY =================
    "giày thể thao": ("sneakers", "Giày"), "giày sneaker": ("sneakers", "Giày"),
    "giày cao gót": ("high heels", "Giày"), "giày búp bê": ("ballet flats", "Giày"),
    "giày lười": ("loafers", "Giày"), "giày da": ("leather shoes", "Giày"), "giày tây": ("dress shoes", "Giày"),
    "giày bốt": ("boots", "Giày"), "giày boot": ("boots", "Giày"), "giày oxford": ("oxford shoes", "Giày"), 
    "giày bánh mì": ("platform shoes", "Giày"), "giày gót nhọn": ("stiletto heels", "Giày"),
    "giày đế xuồng": ("wedge shoes", "Giày"), "sục": ("mules", "Giày"), "giày": ("shoes", "Giày"),

    # ================= DÉP / PHỤ KIỆN =================
    "dép quai ngang": ("slide sandals", "Dép"), "dép xỏ ngón": ("flip-flops", "Dép"),
    "dép tông": ("flip-flops", "Dép"), "dép lào": ("flip-flops", "Dép"), "dép lê": ("slippers", "Dép"), 
    "dép sandal": ("sandals", "Dép"), "sandal": ("sandals", "Dép"), "xăng đan": ("sandals", "Dép"), 
    "guốc": ("clogs", "Dép"), "dép sục": ("crocs", "Dép"), "dép chéo": ("criss-cross sandals", "Dép"), 
    "dép": ("sandals", "Dép"),
    "túi xách": ("handbag", None), "balo": ("backpack", None),
    "mũ": ("hat", None), "nón": ("hat", None), "mũ lưỡi trai": ("baseball cap", None),
    "kính mát": ("sunglasses", None), "thắt lưng": ("belt", None), "dây nịt": ("belt", None),
    "vớ": ("socks", None), "tất": ("socks", None),

    # ================= MÀU SẮC =================
    "trắng": ("white", None), "đen": ("black", None), "đỏ": ("red", None), "đỏ đô": ("burgundy", None),
    "hồng": ("pink", None), "vàng": ("yellow", None), "cam": ("orange", None), "tím": ("purple", None), 
    "nâu": ("brown", None), "xám": ("gray", None), "ghi": ("gray", None), "bạc": ("silver", None),
    "xanh dương": ("blue", None), "xanh biển": ("blue", None), "xanh đen": ("dark navy", None),
    "xanh lá": ("green", None), "xanh navy": ("navy blue", None), "xanh rêu": ("olive green", None), 
    "xanh ngọc": ("turquoise", None), "xanh mint": ("mint green", None), "xanh": ("blue", None),
    "kem": ("cream", None), "be": ("beige", None),

    # ================= CHẤT LIỆU =================
    "da": ("leather", None), "ren": ("lace", None), "voan": ("chiffon", None),
    "lụa": ("silk", None), "nỉ": ("fleece", None), "len": ("knit", None),
    "đũi": ("linen", None), "nhung": ("velvet", None), "gấm": ("brocade", None),
    "kaki": ("khaki", None), "jean": ("denim", None), "bò": ("denim", None),

    # ================= CHI TIẾT / HỌA TIẾT / KIỂU DÁNG =================
    "tay ngắn": ("short sleeve", None), "tay dài": ("long sleeve", None), "không tay": ("sleeveless", None),
    "tay phồng": ("puff sleeve", None), "cổ tròn": ("crew neck", None), "cổ chữ v": ("v-neck", None), 
    "cổ lọ": ("turtleneck", None), "cổ vuông": ("square neck", None), "cổ tim": ("sweetheart neck", None),
    "cổ bẻ": ("collared", None), "form rộng": ("oversized", None), "oversize": ("oversized", None),
    "ôm body": ("fitted", None), "dáng ôm": ("slim fit", None), "dáng rộng": ("loose fit", None),
    "họa tiết": ("patterned", None), "họa tiết lớn": ("large graphic", None), "kẻ sọc": ("striped", None), 
    "sọc": ("striped", None), "caro": ("plaid", None), "hoa": ("floral", None), "hoa nhí": ("ditsy floral", None),
    "trơn": ("plain", None), "chấm bi": ("polka dot", None), "da báo": ("leopard print", None),
    "áo có hình": ("graphic shirt", "Áo"), "áo in hình": ("printed shirt", "Áo"),
    "hình chính giữa": ("with front center graphic", None), "hình in": ("graphic print", None),
    "áo thun có hình": ("graphic t-shirt", "Áo"),
    "cao cổ": ("high-top", None), "cổ cao": ("high-top", None), "cổ thấp": ("low-top", None),
    "ống rộng": ("wide-leg", None), "ống loe": ("flared", None), "ống đứng": ("straight-leg", None),
    "ống suông": ("straight-leg", None), "lưng cao": ("high-waisted", None), "cạp cao": ("high-waisted", None), 
    "lưng thun": ("elastic waist", None), "mũi nhọn": ("pointed toe", None), "mũi tròn": ("round toe", None), 
    "đế bằng": ("flat sole", None), "đế cao": ("platform", None), "rách gối": ("ripped knee", None), 
    "rách": ("ripped", None), "xẻ tà": ("slit", None), "tua rua": ("fringe", None), 
    "bèo nhún": ("ruffled", None), "thắt nơ": ("bow", None), "đính đá": ("rhinestone", None), 
    "xuyên thấu": ("sheer", None), "trễ vai": ("off-the-shoulder", None), "lệch vai": ("one-shoulder", None),

    # ================= TỪ NỐI / ĐỐI TƯỢNG =================
    "đi làm": ("office", None), "công sở": ("office", None),
    "nam": ("men's", None), "nữ": ("women's", None), "trẻ em": ("kids", None), "bé trai": ("boys", None),
    "bé gái": ("girls", None), "màu": ("", None), "loại": ("", None), "kiểu": ("", None),
    "và": ("and", None), "với": ("with", None), "có": ("", None), "cho": ("", None),
}

# Tu muon tieng Anh nguoi dung hay go thang -> giu nguyen, KHONG dua cho mo hinh dich
LOANWORDS = {"polo": "polo", "jean": "denim", "jeans": "jeans", "denim": "denim", "oversize": "oversized",
             "bomber": "bomber", "hoodie": "hoodie", "jogger": "jogger", "cargo": "cargo", "sweater": "sweater",
             "cardigan": "cardigan", "blazer": "blazer", "sneaker": "sneakers", "sneakers": "sneakers",
             "boots": "boots", "loafer": "loafers", "croptop": "crop top", "vintage": "vintage",
             "basic": "basic", "unisex": "unisex", "slim": "slim fit", "hoodies": "hoodie"}

# Mot loai san pham thuong co nhieu cach goi trong caption/ten san pham -> tao vai cau truy van
# (co the encode roi lay trung binh vector) de tang kha nang khop. Sua theo cach goi trong kho cua ban.
VARIANTS = {
    "flip-flops": ["flip-flops", "thong sandals", "rubber slippers"],
    "slippers": ["slippers", "house slippers", "slide sandals"],
    "slide sandals": ["slide sandals", "slides", "open-toe slippers"],
    "sandals": ["sandals", "strappy sandals", "slippers"],
    "shirt": ["shirt", "clothing top", "casual shirt"],
    "t-shirt": ["t-shirt", "tee", "short sleeve t-shirt"],
    "jacket": ["jacket", "coat", "outerwear"],
    "pants": ["pants", "trousers", "long pants"],
    "sneakers": ["sneakers", "athletic shoes", "running shoes"],
    "shoes": ["shoes", "footwear"],
}

_KEYS = {norm(k): v for k, v in GLOSSARY.items()}
_MAXN = max(len(k.split()) for k in _KEYS)


def translate_query(query: str, mt=None):
    """mt: ham tuy chon (list[str]) -> list[str] de dich cac doan chua co trong tu dien.
    Tra ve dict(en=..., category=..., unknown=[...])."""
    orig = query.strip().split()          # giu nguyen dau de dua cho MT
    toks = [norm(t) for t in orig]
    parts, unknown, cats, i = [], [], [], 0   # parts: (is_known, text)
    while i < len(toks):
        for n in range(min(_MAXN, len(toks) - i), 0, -1):
            hit = _KEYS.get(" ".join(toks[i:i + n]))
            if hit:
                if hit[0]:
                    parts.append((True, hit[0]))
                if hit[1]:
                    cats.append(hit[1])
                i += n
                break
        else:
            if toks[i] in LOANWORDS:
                parts.append((True, LOANWORDS[toks[i]]))
            else:
                parts.append((False, orig[i])); unknown.append(orig[i])
            i += 1

    # gom cac tu chua biet lien tiep thanh 1 doan de dich
    segs, cur = [], []
    for known, text in parts:
        if known:
            if cur: segs.append((False, " ".join(cur))); cur = []
            segs.append((True, text))
        else:
            cur.append(text)
    if cur: segs.append((False, " ".join(cur)))

    if mt is not None:
        todo = [t for k, t in segs if not k]
        done = iter(mt(todo) if todo else [])
        segs = [(True, next(done)) if not k else (k, t) for k, t in segs]

    # tieng Anh: loai "top" thua khi da co loai ao cu the; giu thu tu goc la du cho CLIP
    en = " ".join(t for _, t in segs)
    category = cats[0] if len(set(cats)) == 1 else None
    if category == "Áo":   # "cổ cao/thấp" cho áo là high/low-neck, không phải high/low-top (giày)
        en = en.replace("high-top", "high-neck").replace("low-top", "low-neck")
    variants = [en]
    for term, alts in VARIANTS.items():
        if re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", en):
            variants = [re.sub(rf"(?<![\w-]){re.escape(term)}(?![\w-])", a, en, count=1) for a in alts]
            break
    return {"en": en, "variants": variants,
            "category": category, "unknown": unknown}


if __name__ == "__main__":
    for q in ["Áo thun trắng tay ngắn", "quần jean xanh ống rộng", "giay the thao mau den",
              "dép quai ngang màu nâu", "váy xanh không tay", "áo khoác bomber nữ"]:
        print(q, "->", translate_query(q))