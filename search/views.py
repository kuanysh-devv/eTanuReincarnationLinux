import os
import cv2
import json
import requests
import numpy as np
import urllib3
from io import BytesIO
from uuid import uuid4
from datetime import datetime
from PIL import Image

from django.apps import apps
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.contrib.auth.models import User

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.decorators import permission_classes, action

from minio import Minio
from mtcnn import MTCNN
from pymilvus import Milvus, DataType, Collection, connections, MilvusClient

import insightface
from insightface.app.common import Face
from insightface.model_zoo import model_zoo

from dotenv import load_dotenv

from .forms import *
from metadata.models import *
from metadata.permissions import JWTTokenFromRequestPermission


MILVUS_HOST = os.environ.get('MILVUS_HOST')
MILVUS_PORT = os.environ.get('MILVUS_PORT')
MINIO_ENDPOINT = os.environ.get('MINIO_ENDPOINT')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY')
REC_MODEL_PATH = os.environ.get('REC_MODEL_PATH')

detector = MTCNN(steps_threshold=[0.7, 0.8, 0.9], min_face_size=40)
connections.connect(
    host=MILVUS_HOST,
    port=MILVUS_PORT,
    timeout=100000
)
client = MilvusClient(
    uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}"
)

pool_manager = urllib3.PoolManager(
    num_pools=10,  # Number of pools
    maxsize=10,  # Maximum size of a pool
    retries=3,  # Number of retries
)

minio_client = Minio(
    endpoint=MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

rec_model_path = REC_MODEL_PATH
rec_model = model_zoo.get_model(rec_model_path)
rec_model.prepare(ctx_id=0)

collection = Collection('face_embeddings')
collection.load()


def upload_image_to_minio(image_data, bucket_name, content_type, directory_name):
    try:
        # Create BytesIO object from image data
        image_stream = BytesIO(image_data)

        # Generate unique object name using uuid4()
        object_name = str(uuid4()) + content_type.replace('image/',
                                                          '.')  # Example: '7f1d18a4-2c0e-47d3-afe1-6d27c3b9392e.png'
        object_name = f"{directory_name}/{object_name}"
        # Upload image to MinIO
        minio_client.put_object(
            bucket_name,
            object_name,
            image_stream,
            len(image_data),
            content_type=content_type  # Change content type based on image format
        )
        return object_name
    except Exception as err:
        print(f"MinIO Error: {err}")


def search_faces_in_milvus(embedding, limit):
    search_params = {"metric_type": "L2", "params": {"nprobe": 32}}

    results = collection.search(
        anns_field="face_vector",
        data=[embedding],
        limit=limit,
        param=search_params
    )
    # Retrieve vector IDs of the closest embeddings
    vector_ids = [result.id for result in results[0]]
    distances = [result.distance for result in results[0]]

    return vector_ids, distances


def retrieve_face_vectors(vector_ids, collection_name):
    # Perform a query in Milvus to get face_vectors for the given vector_ids
    res = client.query(
        collection_name=collection_name,
        filter=f'vector_id in {vector_ids}',
        output_fields=["vector_id", "face_vector"]
    )

    # Parse results
    face_vectors = {}
    for item in res:
        vector_id = item['vector_id']
        face_vector = np.array(item['face_vector'])
        face_vectors[vector_id] = face_vector

    return face_vectors


def calculate_dot_product(embedding1, embedding2):
    return np.dot(embedding1, embedding2)


def convert_image_to_embeddingv2(img, face):
    # Detect faces in the image
    rec_model.get(img, face)
    embeddings = face.normed_embedding
    return embeddings.squeeze().tolist()


class SearchView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [JWTTokenFromRequestPermission]

    @action(detail=False, methods=['post'])
    def post(self, request):
        # Get the uploaded image file and the limit parameter from the request
        limit = int(request.POST.get('limit', 10))  # Default limit is 10 if not provided
        user_id = request.POST.get('auth_user_id')
        reload = request.POST.get('reload')
        search_reason = request.POST.get('reason')
        reason_data = request.POST.get('reason_data')
        minimum_similarity = request.POST.get('minimum_similarity')
        bucket_name = 'history'

        if reason_data:
            try:
                reason_data = json.loads(reason_data)
            except json.JSONDecodeError:
                return JsonResponse({'error': 'Invalid reason_data format'}, status=400)

        if reload == "1":
            image_name = request.POST.get('image_name')
            image_url = f'http://{MINIO_ENDPOINT}/{bucket_name}/{image_name}'
            response = requests.get(image_url, verify=False)
            image_data = response.content
        # Read the image file and convert it to an OpenCV format
        else:
            image_file = request.FILES['image']
            image_data = image_file.read()

        user = User.objects.get(id=user_id)
        account = Account.objects.get(user=user)

        if account.role_id != 'admin' and (search_reason is None or not reason_data):
            return JsonResponse({'error': 'Основание поиска не заполнено'}, status=400)

        if minimum_similarity is not None and minimum_similarity != 'undefined':
            minimum_similarity = float(minimum_similarity)

        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JsonResponse({'error': 'Failed to decode the image'}, status=400)

        # Convert the image to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use MTCNN to detect faces and keypoints in the image
        faces = detector.detect_faces(img_rgb)

        face_results = []
        for face in faces:
            # Convert the face to an embedding

            bbox = face['box']
            det_score = face['confidence']
            kps_dict = face['keypoints']
            kps = np.array([list(kps_dict.values())]).squeeze()

            face = Face(bbox=bbox, kps=kps, det_score=det_score)

            embedding = convert_image_to_embeddingv2(img_rgb, face)
            # Search for the face in Milvus
            vector_ids, distances = search_faces_in_milvus(embedding, limit)
            # Retrieve metadata for each vector ID
            gallery_objects = Gallery.objects.filter(vector_id__in=vector_ids)

            face_vectors = retrieve_face_vectors(vector_ids, "face_embeddings")

            milvus_results = []
            for vector_id in vector_ids:
                known_embedding = face_vectors.get(vector_id)
                if known_embedding is not None:
                    similarity = calculate_dot_product(embedding, known_embedding)
                    similarity_percentage = round(similarity * 100, 2)

                    # Filter by minimum similarity if provided
                    if (minimum_similarity is None or minimum_similarity == 'undefined' or
                            similarity_percentage >= minimum_similarity):
                        milvus_results.append({
                            'vector_id': vector_id,
                            'similarity': similarity_percentage
                        })

            metadata_list = [
                {
                    'vector_id': obj.vector_id,
                    'iin': obj.personId.iin,
                    'name': obj.personId.firstname,
                    'surname': obj.personId.surname,
                    'patronymic': obj.personId.patronymic,
                    'birth_date': obj.personId.birthdate,
                    'photo': obj.photo
                } for obj in gallery_objects
            ]

            # Associate metadata with Milvus results based on vector ID
            for milvus_result in milvus_results:
                vector_id = milvus_result['vector_id']
                metadata = next((item for item in metadata_list if item['vector_id'] == vector_id), None)
                milvus_result['metadata'] = metadata

            keypoints = face.kps.tolist()  # Convert keypoints to a list
            bbox = face.bbox

            face_result = {
                'bbox': bbox,
                'keypoints': keypoints,
                'milvus_results': milvus_results
            }
            face_results.append(face_result)

        uploaded_object_name = upload_image_to_minio(image_data, bucket_name, content_type='image/jpg',
                                                     directory_name=account.id)

        search_history = SearchHistory.objects.create(
            account=account,
            searchedPhoto=uploaded_object_name,
            created_at=datetime.now(),
            reason=search_reason,
        )

        if account.role_id != 'admin':
            try:
                if search_reason == 'CRIMINAL_CASE':
                    reason_form = CriminalCaseReasonForm(reason_data)
                elif search_reason == 'INVESTIGATIVE_ORDER':
                    reason_form = InvestigativeOrderReasonForm(reason_data)
                elif search_reason == 'PROSECUTOR_INSTRUCTION':
                    reason_form = ProsecutorInstructionReasonForm(reason_data)
                elif search_reason == 'INTERNATIONAL_ORDER':
                    reason_form = InternationalOrderReasonForm(reason_data)
                elif search_reason == 'AFM_ORDER':
                    reason_form = AfmOrderReasonForm(reason_data)
                elif search_reason == 'HEAD_ORDER':
                    reason_form = HeadOrderReasonForm(reason_data)
                elif search_reason == 'OPERATIONAL_INSPECTION':
                    reason_form = OperationalInspectionReasonForm(reason_data)
                elif search_reason == 'ANALYTICAL_WORK':
                    reason_form = AnalyticalWorkReasonForm(reason_data)
                else:
                    return JsonResponse({'status': 'error', 'message': 'Invalid reason type'}, status=400)
                if reason_form.is_valid():
                    reason_instance = reason_form.save(commit=False)
                    reason_instance.search_history = search_history
                    reason_instance.save()
                else:
                    return JsonResponse({'status': 'error', 'message': reason_form.errors}, status=400)
            except Exception as e:
                # Handle exceptions, e.g., log the error and return an error response
                print(f"An error occurred while saving the reason data: {e}")
                return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

        return JsonResponse({'faces': face_results,
                             'image_name': uploaded_object_name})

