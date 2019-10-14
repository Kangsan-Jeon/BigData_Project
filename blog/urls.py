from django.urls import path

from django.conf.urls import url
from . import views
from django.conf import settings # add
from django.conf.urls.static import static # add

urlpatterns = [
    path('', views.post_list, name='post_list'),
    url(r'^uimage/$', views.uimage, name='uimage'), # add
    url(r'^uvideo/$', views.uvideo, name='uvideo'), # add
]
 
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) # 이미지를 업로드하기 위한 설정