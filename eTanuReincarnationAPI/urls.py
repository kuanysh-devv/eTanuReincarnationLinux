from django.contrib import admin
from django.urls import path, include
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView, TokenVerifyView
from metadata.views import *
from rest_framework import routers
from metadata.views import CustomTokenObtainPairView
from search.views import process_image


router = routers.DefaultRouter()
router.register(r'account', AccountViewSet)
router.register(r'metadata', MetadataViewSet)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include(router.urls)),
    path('api/v1/login/', CustomTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/v1/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/v1/token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    path('api/v1/commit-photos/', MetadataViewSet.as_view({'get': 'commit'}), name='commitPhotos'),
    path('api/v1/getUserInfo/', AccountViewSet.as_view({'post': 'getUserInfo'}), name='getUserInfo'),
    path('api/v1/register/', register, name='register'),
    path('api/v1/search/', process_image, name='process_image'),
]
