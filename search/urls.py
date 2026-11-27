from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("search/", views.search, name="search"),
    path("category/", views.category, name="category"),        # <-- thêm dòng này
    path("product/<str:product_id>/", views.product_detail, name="product_detail"),
]