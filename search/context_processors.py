"""
context_processors.py - Đưa dữ liệu dùng chung cho MỌI template.

Cần thêm vào settings.py:
    TEMPLATES[0]["OPTIONS"]["context_processors"].append("<tên_app>.context_processors.nav")
"""
from . import categories


def nav(request):
    """Cấp biến `nav_groups` cho base.html (thanh menu danh mục)."""
    return {"nav_groups": categories.nav_groups()}