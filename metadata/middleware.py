# custom_middleware.py
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
import time

PUBLIC_KEY_PATH = "/mnt/c/Users/User4/PycharmProjects/eTanuReincarnationLinux/metadata/public_key_afm.pem"

def load_public_key():
    with open(PUBLIC_KEY_PATH, "r") as f:
        return f.read()

public_key = load_public_key()


class RequestTimeMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.time()
        response = self.get_response(request)
        end_time = time.time()
        request.wasted_time = end_time - start_time
        return response


class JWTAuthMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Skip auth for login/refresh/verify endpoints
        if request.path.startswith("/api/v1/login/") or \
           request.path.startswith("/api/v1/token/"):
            return self.get_response(request)

        token = request.COOKIES.get('access_token')
        if token:
            try:
                payload = jwt.decode(
                    token,
                    settings.PUBLIC_KEY,
                    algorithms=["RS256"],
                    options={"require": ["exp", "iat"]}
                )
                request.jwt_payload = payload  # Optional: for use in views
            except (ExpiredSignatureError, InvalidTokenError):
                return JsonResponse({'detail': 'Invalid or expired token'}, status=401)

        else:
            return JsonResponse({'detail': 'Authentication credentials were not provided'}, status=401)

        return self.get_response(request)