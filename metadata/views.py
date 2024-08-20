from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from django.http import JsonResponse
import time
import numpy as np
from rest_framework.parsers import MultiPartParser
from .tasks import process_image, process_image_from_row
from django.shortcuts import get_object_or_404
from django.utils.decorators import method_decorator
from rest_framework import viewsets, status, pagination
from django.contrib.auth.models import User
from mtcnn import MTCNN
import json
import os
import cv2
import csv
from django.views.decorators.csrf import csrf_exempt
from rest_framework_simplejwt.views import TokenObtainPairView
from dotenv import load_dotenv
from PIL import Image
from tensorflow.keras.models import load_model
from pymilvus import Milvus, CollectionSchema, FieldSchema, DataType, Collection, connections, utility
import base64
from tensorflow.keras.preprocessing.image import img_to_array
from .models import Person, Account, SearchHistory, Gallery
from .serializers import PersonSerializer, AccountSerializer, CustomTokenObtainPairSerializer

load_dotenv()

MILVUS_HOST = os.environ.get('MILVUS_HOST')
MILVUS_PORT = os.environ.get('MILVUS_PORT')
detector = MTCNN(steps_threshold=[0.7, 0.8, 0.9], min_face_size=40)
csv.field_size_limit(1000000000)
model = load_model('metadata/models/model.h5')


def preprocess_frame(frame):
    # Resize the frame to match the model input size
    resized_frame = cv2.resize(frame, (150, 150))  # Adjust size according to your model's input size

    # Convert to an array and normalize pixel values
    image_array = img_to_array(resized_frame)
    normalized_array = image_array / 255.0

    # Expand dimensions to match the model input shape (batch_size, height, width, channels)
    return np.expand_dims(normalized_array, axis=0)


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


def import_embeddings_from_csv(csv_path, partition_name):
    milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    collection_name = "face_embeddings"
    # Check if collection exists
    if collection_name in milvus.list_collections():
        collection = Collection("face_embeddings")
        print(f"Collection already exists.")
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
            name="face_embeddings",
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
        if collection.has_partition(partition_name):
            print("Partition already exists")
        else:
            collection.create_partition(partition_name)

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            task_result = process_image_from_row.delay(row, partition_name)


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


def is_real_face(model_instance, frame):
    preprocessed_frame = preprocess_frame(frame)

    # Make predictions
    prediction = model_instance.predict(preprocessed_frame)

    # Assuming binary classification where >0.5 indicates a real face
    print(prediction[0])
    return prediction[0] > 0.5


def analyze_liveness_with_keras(frames, model_instance):
    liveness_score = 0

    for frame in frames:
        if is_real_face(model_instance, frame):
            liveness_score += 1

    # Adjust the threshold based on your testing
    return liveness_score >= len(frames) - 2


class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer


class PersonViewSet(viewsets.ModelViewSet):
    queryset = Person.objects.all()
    serializer_class = PersonSerializer
    permission_classes = (IsAuthenticated,)

    @action(detail=False, methods=['get'])
    def commit(self, request, *args, **kwargs):
        start_time = time.time()
        # image_directory = "/root/eTanuReincarnation/metadata/data/test02"
        # csv_path = "/root/eTanuReincarnationLinux/metadata/data/from1970to1974.csv"
        csv_path = "C:/Users/User4/Documents/photos_00-05.csv"
        partition_name = 'from2000to2005'

        import_embeddings_from_csv(csv_path, partition_name)

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


@csrf_exempt
def authenticate_user(request):
    if request.method == 'POST':
        num_frames = int(request.POST.get('num_frames', 0))  # Use request.POST for form data
        frames = []

        # Iterate over the files
        for i in range(1, num_frames + 1):  # Adjust to include num_frames
            file_key = f'frame_{i}'  # Frame keys are 'frame_1', 'frame_2', etc.
            if file_key in request.FILES:
                uploaded_file = request.FILES[file_key]
                frames.append(uploaded_file)

        if not frames:
            return JsonResponse({'error': 'No frames received'}, status=400)

        # Process frames
        decoded_frames = []
        for file in frames:
            # Convert file to an OpenCV-compatible format
            image_data = file.read()
            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            decoded_frames.append(image)

        # Analyze liveness
        liveness_passed = analyze_liveness_with_keras(decoded_frames, model)

        if liveness_passed:
            return JsonResponse({'success': 'Liveness detected'}, status=200)
        else:
            return JsonResponse({'error': 'Liveness detection failed'}, status=400)

    return JsonResponse({'error': 'Invalid request'}, status=400)


class HistoryPagination(pagination.PageNumberPagination):
    page_size = 10  # Set the page size to 10


class AccountViewSet(viewsets.ModelViewSet):
    queryset = Account.objects.all()
    serializer_class = AccountSerializer
    parser_classes = [MultiPartParser]
    permission_classes = (IsAuthenticated,)

    @action(detail=False, methods=['post'])
    def getUserInfo(self, request, *args, **kwargs):
        user_data = {}
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

    @method_decorator(csrf_exempt, name='dispatch')
    @action(detail=False, methods=['get'], url_path='history/(?P<auth_user_id>[^/.]+)')
    def get_history(self, request, auth_user_id=None):
        try:
            if not auth_user_id:
                return JsonResponse({"error": "auth_user_id is required"}, status=status.HTTP_400_BAD_REQUEST)

            user = get_object_or_404(User, id=auth_user_id)
            account = get_object_or_404(Account, user=user)
            history_list = SearchHistory.objects.filter(account=account).order_by('-created_at')

            paginator = HistoryPagination()
            paginated_history = paginator.paginate_queryset(history_list, request)

            history_data = []
            for history in paginated_history:
                history_data.append({
                    "searchedPhoto": history.searchedPhoto,
                    "created_at": history.created_at,
                })

            return paginator.get_paginated_response(history_data)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
