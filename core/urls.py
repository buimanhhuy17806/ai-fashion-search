from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    # Điều hướng mọi yêu cầu về thư mục app 'search'
    path('', include('search.urls')),
]

# Phục vụ ảnh sản phẩm qua /media/... (chỉ hoạt động khi DEBUG = True)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)