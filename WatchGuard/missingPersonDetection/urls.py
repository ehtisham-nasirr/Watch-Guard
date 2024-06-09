

# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.home, name='home_page'),
#     path('missingperson/', views.mp, name='mp_page'),
#     path('facerecognition/', views.fr, name='fr_page'),
#     path('contacts', views.contacts, name='contact_page'),
#     path('abouts', views.abouts, name='abouts_page'),

# ]











from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home_page'),
    path('missingperson/', views.mp, name='mp_page'),
    path('facerecognition/', views.fr, name='fr_page'),
    path('contacts/', views.contacts, name='contact_page'),
    path('abouts/', views.abouts, name='abouts_page'),
]
