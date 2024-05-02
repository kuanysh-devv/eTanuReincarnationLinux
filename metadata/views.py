from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from django.http import JsonResponse, HttpResponse
import time
from .tasks import process_image, process_image_from_row
import cv2
import torch
from django.contrib.auth.models import User
import json
import os
import pandas as pd
import csv
import uuid
from django.views.decorators.csrf import csrf_exempt
from rest_framework_simplejwt.views import TokenObtainPairView
import io
import sys
import numpy as np
from dotenv import load_dotenv
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
from pymilvus import Milvus, CollectionSchema, FieldSchema, DataType, Collection, connections, utility
import base64
from .models import Metadata, Account, SearchHistory
from .serializers import MetadataSerializer, AccountSerializer, CustomTokenObtainPairSerializer

load_dotenv()

MILVUS_HOST = os.environ.get('MILVUS_HOST')
MILVUS_PORT = os.environ.get('MILVUS_PORT')


def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
        return base64_encoded


def get_image_metadata(image_path):
    with Image.open(image_path) as img:
        metadata = {
            'format': img.format,
            'mode': img.mode,
            'size': img.size,
            'info': img.info
        }
    return metadata


def import_embeddings_from_csv(csv_path, collection_name):
    milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )

    # Check if collection exists
    if collection_name in milvus.list_collections():
        collection = Collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    else:
        # Create collection with embedding and id fields
        vector_id = FieldSchema(
            name="vector_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=36,
            auto_id=False
        )
        vector = FieldSchema(
            name="face_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=512
        )
        schema = CollectionSchema(
            fields=[vector_id, vector],
            description="Collection of face embeddings",
            enable_dynamic_field=True
        )
        collection = Collection(
            name=collection_name,
            schema=schema,
            using='default'
        )

    start = 1
    # Import embeddings into Milvus
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    if collection:
        collection.create_index(
            field_name="face_vector",
            index_params=index_params
        )

        utility.index_building_progress(collection_name)

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            task_result = process_image_from_row.delay(row, collection_name)


def import_embeddings_from_images(image_directory, collection_name):
    # Initializing milvus connection and checks for collection
    milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )

    # Check if collection exists
    if collection_name in milvus.list_collections():
        collection = Collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    else:
        # Create collection with embedding and id fields
        vector_id = FieldSchema(
            name="vector_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=36,
            auto_id=False
        )
        vector = FieldSchema(
            name="face_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=512
        )
        schema = CollectionSchema(
            fields=[vector_id, vector],
            description="Collection of face embeddings",
            enable_dynamic_field=True
        )
        collection = Collection(
            name=collection_name,
            schema=schema,
            using='default'
        )

    start = 1
    # Import embeddings into Milvus
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    if collection:
        collection.create_index(
            field_name="face_vector",
            index_params=index_params
        )

        utility.index_building_progress(collection_name)
    # taking every image in directory in uploading as task

    for index, file_name in enumerate(os.listdir(image_directory), start=start):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            image_path = os.path.join(image_directory, file_name)
            process_image.delay(image_path, collection_name)


class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer


class MetadataViewSet(viewsets.ModelViewSet):
    queryset = Metadata.objects.all()
    serializer_class = MetadataSerializer
    permission_classes = (IsAuthenticated,)

    @action(detail=False, methods=['get'])
    def commit(self, request, *args, **kwargs):
        start_time = time.time()
        image_directory = "/root/eTanuReincarnation/metadata/data/test02"
        csv_path = "C:/Users/User4/PycharmProjects/eTanuReincarnationAPI/output.csv"
        collection_name = 'face_embeddings02'

        import_embeddings_from_csv(csv_path, collection_name)

        end_time = time.time()
        wasted_time = end_time - start_time

        print("Request wasted time:", wasted_time)

        return JsonResponse({'status': 'Tasks uploaded into celery'})


@csrf_exempt
def register(request):
    if request.method == 'POST':
        # Extract data from the POST request
        data = json.loads(request.body.decode('utf-8'))
        username = data.get('username')
        password = data.get('password')
        firstname = data.get('firstname')
        surname = data.get('surname')
        patronymic = data.get('patronymic')
        role_id = data.get('role_id')
        print("okay")
        print(username)
        print(password)
        print(role_id)
        if username is not None and password is not None and role_id is not None:
            # Create the user
            print("hello")
            user = User.objects.create_user(username=username, password=password)

            # Create the account and associate with the user
            account = Account.objects.create(
                user=user,
                firstname=firstname,
                surname=surname,
                patronymic=patronymic,
                role_id=role_id
            )

            return JsonResponse({'status': 'Регистрация прошла успешно!'})

    # Return a failure response if data is missing or request method is not POST
    return JsonResponse({'status': 'Упс что-то пошло не так...'})


class AccountViewSet(viewsets.ModelViewSet):
    queryset = Account.objects.all()
    serializer_class = AccountSerializer
    permission_classes = (IsAuthenticated,)

    @action(detail=False, methods=['post'])
    def getUserInfo(self, request, *args, **kwargs):
        if request.method == 'POST':
            data = json.loads(request.body.decode('utf-8'))
            user_id = data.get('auth_user_id')
            user = User.objects.get(id=user_id)
            account = Account.objects.get(user=user)
            last_three_history = SearchHistory.objects.filter(account=account).order_by('-created_at')[:3]
            history_list = list(last_three_history.values())
            user_data = {
                'auth_user_id': user.id,
                'account_id': account.id,
                'username': user.username,
                'first_name': account.firstname,
                'surname': account.surname,
                'patronymic': account.patronymic,
                'role_id': account.role_id,
                'history': history_list
            }

        return JsonResponse(user_data)