# permissions.py
from rest_framework.permissions import BasePermission
from jwt import ExpiredSignatureError, InvalidTokenError
import jwt
from django.conf import settings
from dotenv import load_dotenv
import os

load_dotenv()

PUBLIC_KEY_PATH = os.getenv("PUBLIC_KEY_PATH")

def load_public_key():
    with open(PUBLIC_KEY_PATH, "r") as f:
        return f.read()

public_key = load_public_key()
# 750127450079

class JWTTokenFromRequestPermission(BasePermission):
    def has_permission(self, request, view):
        auth_header = request.headers.get('Authorization')

        if not auth_header or not auth_header.startswith('Bearer '):
            return False

        try:
            token = auth_header.split(' ')[1]
        except IndexError:
            return False
        print(token)
        try:
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                options={"require": ["exp"]}
            )
            request.jwt_payload = payload
            return True
        except (ExpiredSignatureError, InvalidTokenError):
            return False
