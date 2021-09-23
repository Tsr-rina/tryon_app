from django.urls import path
from django_app import views

app_name = 'tryon_app'
urlpatterns = [
    path('', views.index, name='index'),
    path('cloth_select', views.select_cloth, name='cloth_select'),
    path('result', views.try_on, name='tryon')
]