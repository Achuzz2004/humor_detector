from django.urls import path
from .views import humor_classifier, register_view, login_view, logout_view

urlpatterns = [
    path('', humor_classifier, name='home'),
    path('register/', register_view, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
]