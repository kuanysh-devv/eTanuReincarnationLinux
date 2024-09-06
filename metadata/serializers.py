from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import Person, Account, Gallery
from django.contrib.auth.models import User


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    auth_user_id = serializers.IntegerField(source='user.id', read_only=True)

    def validate(self, attrs):
        data = super().validate(attrs)
        user = self.user

        # Fetch the account instance once
        data['auth_user_id'] = user.id
        account_instance = Account.objects.filter(user=user).first()
        role_id = account_instance.role_id if account_instance else None

        # Determine the status and response based on role and face vector
        status = 1 if role_id != 'admin' and account_instance and account_instance.face_vector_id else 0

        # Construct response data
        response_data = {
            'status': status,
            'auth_user_id': data['auth_user_id'],
            'account_id': account_instance.id,
            'username': user.username,
            'first_name': account_instance.firstname,
            'surname': account_instance.surname,
            'patronymic': account_instance.patronymic,
            'role': role_id,
            'access_token': data.get('access'),
            'refresh_token': data.get('refresh')
        }

        # Remove tokens from response if the user is not an admin
        if status == 1 or (status == 0 and role_id != 'admin'):
            response_data.pop('access_token', None)
            response_data.pop('refresh_token', None)

        return response_data

    class Meta:
        fields = 'auth_user_id'  # Include fields needed in the response


class PersonSerializer(serializers.ModelSerializer):
    class Meta:
        model = Person
        fields = "__all__"


class AccountSerializer(serializers.ModelSerializer):
    class Meta:
        model = Account
        fields = "__all__"


class GallerySerializer(serializers.ModelSerializer):
    class Meta:
        model = Gallery
        fields = "__all__"
