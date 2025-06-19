import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
import time
from dotenv import load_dotenv
import os
from django.http import JsonResponse

load_dotenv()

PUBLIC_KEY_PATH = os.getenv("PUBLIC_KEY_PATH")

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


class CookieJWTMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        access_token = request.COOKIES.get("access_token")
        if access_token:
            try:
                payload = decode(
                    access_token,
                    public_key,
                    algorithms=["RS256"],
                    options={"require": ["exp"]}
                )
                request.jwt_payload = payload
            except ExpiredSignatureError:
                return JsonResponse({"detail": "Token expired"}, status=401)
            except InvalidTokenError:
                return JsonResponse({"detail": "Invalid token"}, status=401)
        return self.get_response(request)

