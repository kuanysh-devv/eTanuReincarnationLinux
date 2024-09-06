from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from django.http import JsonResponse
import time
from minio import Minio
import numpy as np
from rest_framework.parsers import MultiPartParser
from .tasks import process_image, process_image_from_row, convert_image_to_embeddingv2, upload_image_to_minio
from django.shortcuts import get_object_or_404
from django.utils.decorators import method_decorator
from rest_framework import viewsets, status, pagination
from django.contrib.auth.models import User
from mtcnn import MTCNN
import json
import os
from io import BytesIO
import uuid
import cv2
from insightface.app.common import Face
from insightface.model_zoo import model_zoo
import csv
from django.contrib.auth import authenticate
from django.views.decorators.csrf import csrf_exempt
from rest_framework_simplejwt.views import TokenObtainPairView
from dotenv import load_dotenv
from PIL import Image
from rest_framework_simplejwt.tokens import RefreshToken
from tensorflow.keras.models import load_model
from pymilvus import Milvus, CollectionSchema, FieldSchema, DataType, Collection, connections, utility
import base64
from search.views import retrieve_face_vectors, calculate_dot_product
from tensorflow.keras.preprocessing.image import img_to_array
from .models import Person, Account, SearchHistory, Gallery
from .serializers import PersonSerializer, AccountSerializer, CustomTokenObtainPairSerializer

load_dotenv()

MILVUS_HOST = os.environ.get('MILVUS_HOST')
MILVUS_PORT = os.environ.get('MILVUS_PORT')
MINIO_ENDPOINT = os.environ.get('MINIO_ENDPOINT')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY')
REC_MODEL_PATH = os.environ.get('REC_MODEL_PATH')

rec_model = model_zoo.get_model(REC_MODEL_PATH)
rec_model.prepare(ctx_id=0)
detector = MTCNN(steps_threshold=[0.7, 0.8, 0.9], min_face_size=40)
csv.field_size_limit(1000000000)
model = load_model('metadata/models/model.h5')
minio_client = Minio(
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=True,
    cert_check=False
)


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
            process_image_from_row.delay(row, partition_name)


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
    return liveness_score >= len(frames) / 2


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
def get_tokens(request):
    auth_user_id = int(request.POST.get('auth_user_id', 0))

    # Fetch the user based on auth_user_id
    try:
        user = User.objects.get(pk=auth_user_id)
    except User.DoesNotExist:
        return None, None

    # Generate tokens
    refresh = RefreshToken.for_user(user)
    return {
        'access': str(refresh.access_token),
        'refresh': str(refresh)
    }


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
        print(username)
        print(password)
        print(role_id)
        if username is not None and password is not None and role_id is not None:
            # Create the user
            user = User.objects.create_user(username=username, password=password)

            # Create the account and associate with the user
            Account.objects.create(
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
        auth_user_id = int(request.POST.get('auth_user_id', 0))

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
            tokens = get_tokens(request)

            verification_success, error_message = account_face_verification(auth_user_id, decoded_frames[-1])

            if verification_success:
                return JsonResponse({
                    'success': 'Liveness detected and face verified successfully',
                    'access_token': tokens['access'],
                    'refresh_token': tokens['refresh'],
                    'user_id': auth_user_id
                }, status=200)
            else:
                return JsonResponse({'error': error_message}, status=400)

        else:
            return JsonResponse({'error': 'Liveness detection failed'}, status=400)

    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def create_account_faces_collection():
    # Connect to Milvus
    milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )

    # Define collection name
    collection_name = "account_faces"

    # Check if collection exists
    if collection_name in milvus.list_collections():
        collection = Collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
        return collection
    else:
        # Create the collection schema
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
            description="Collection of account face vectors",
            enable_dynamic_field=True
        )
        collection = Collection(
            name=collection_name,
            schema=schema,
            using='default'
        )
        print(f"Collection '{collection_name}' created.")

    # Index parameters
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }

    # Create index if collection exists
    if collection:
        collection.create_index(
            field_name="face_vector",
            index_params=index_params
        )
        print(f"Index created for collection '{collection_name}'.")

        # Check index building progress
        utility.index_building_progress(collection_name)
        return collection


@csrf_exempt
def register_account_face(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)

    auth_user_id = int(request.POST.get('auth_user_id', 0))
    uploaded_face_image = request.FILES.get('face_image')

    if not uploaded_face_image:
        return JsonResponse({'error': 'No image provided'}, status=400)

    try:
        # Fetch user and account
        user_instance = User.objects.get(pk=auth_user_id)
        account_instance = Account.objects.get(user=user_instance)

        if account_instance.face_vector_id:
            return JsonResponse({'error': 'Account already has a registered face'}, status=400)

        # Process the uploaded image
        image_data = uploaded_face_image.read()
        image_rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Detect face
        faces = detector.detect_faces(image_rgb)
        if not faces:
            return JsonResponse({'error': 'No face detected'}, status=400)

        # Extract face details and generate embedding
        face = Face(
            bbox=faces[0]['box'],
            kps=np.array(list(faces[0]['keypoints'].values())),
            det_score=faces[0]['confidence']
        )
        embedding = convert_image_to_embeddingv2(image_rgb, face)
        embedding_id = str(uuid.uuid4())

        # Connect to Milvus, ensure collection exists, and insert embedding
        collection = create_account_faces_collection()
        collection.insert([{"vector_id": embedding_id, "face_vector": embedding}])

        # Upload image to MinIO
        object_name = f"{user_instance.id}/{embedding_id}.jpg"
        minio_client.put_object(
            "account-faces",
            object_name,
            BytesIO(image_data),
            len(image_data),
            content_type='image/jpg'
        )

        # Save embedding ID to account and return success response
        account_instance.face_vector_id = embedding_id
        account_instance.save()

        tokens = get_tokens(request)

        return JsonResponse({
            'success': 'Face registered successfully',
            'access_token': tokens['access'],
            'refresh_token': tokens['refresh'],
            'user_id': auth_user_id
        }, status=200)

    except User.DoesNotExist:
        return JsonResponse({'error': 'User not found'}, status=404)
    except Account.DoesNotExist:
        return JsonResponse({'error': 'Account not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def account_face_verification(auth_user_id, last_frame):
    try:
        # Fetch the associated account
        user_instance = User.objects.get(pk=auth_user_id)
        account_instance = Account.objects.get(user=user_instance)

        if not account_instance.face_vector_id:
            return False, 'No registered face for this account'

        # Convert the last frame to RGB format
        last_frame_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the last frame
        faces = detector.detect_faces(last_frame_rgb)
        if not faces:
            return False, 'No face detected in the provided frame'

        # Get bounding box, confidence score, and keypoints
        face = Face(
            bbox=faces[0]['box'],
            kps=np.array(list(faces[0]['keypoints'].values())),
            det_score=faces[0]['confidence']
        )

        # Convert the last frame to an embedding
        current_embedding = convert_image_to_embeddingv2(last_frame_rgb, face)

        # Retrieve the registered face embedding from Milvus
        face_vectors = retrieve_face_vectors([account_instance.face_vector_id], "account_faces")
        registered_embedding = face_vectors.get(account_instance.face_vector_id)
        if registered_embedding is None:
            return False, 'Registered face embedding not found'

        # Compare embeddings using dot product
        similarity_score = calculate_dot_product(registered_embedding, current_embedding)
        print("Similarity score: ", similarity_score)
        # Determine if the similarity score meets the threshold
        similarity_threshold = 0.6  # Adjust threshold based on your requirements
        if similarity_score >= similarity_threshold:
            return True, None  # Verification successful
        else:
            return False, 'Face verification failed. Faces do not match.'

    except User.DoesNotExist:
        return False, 'User not found'
    except Account.DoesNotExist:
        return False, 'Account not found'
    except Exception as e:
        return False, str(e)


class HistoryPagination(pagination.PageNumberPagination):
    page_size = 12  # Set the page size to 10


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
