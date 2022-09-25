from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

## changed : add import include, add url path('')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',include('classifier.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)