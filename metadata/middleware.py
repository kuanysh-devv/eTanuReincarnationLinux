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


def validate_jwt_token(token: str):
    try:
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"]
        )
        return payload  # valid token
    except ExpiredSignatureError:
        raise PermissionError("Token has expired")
    except InvalidTokenError:
        raise PermissionError("Invalid token")