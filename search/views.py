from django.shortcuts import render
import cv2
from django.apps import apps
from mtcnn import MTCNN
import psycopg2
from facenet_pytorch import InceptionResnetV1
from pymilvus import Milvus, DataType, Collection, connections
import numpy as np
from minio import Minio
from io import BytesIO
from uuid import uuid4
import torch
import insightface
from insightface.app.common import Face
from insightface.model_zoo import model_zoo
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import math
from rest_framework.decorators import authentication_classes, permission_classes
import pytz
from rest_framework.permissions import IsAuthenticated
import base64
from django.contrib.auth.models import User
from collections import Counter
from datetime import datetime, timedelta
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import AuthenticationFailed
from PIL import Image
from metadata.models import Metadata, SearchHistory, Account

detector = MTCNN(steps_threshold=[0.7, 0.8, 0.9], min_face_size=40)
milvus = Milvus(host='localhost', port='19530')
connections.connect(
    host='localhost',
    port='19530'
)
minio_client = Minio(
    endpoint='192.168.122.101:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    secure=False  # Set to True if using HTTPS
)

rec_model_path = '/root/eTanuReincarnation/metadata/insightface/models/w600k_mbf.onnx'
rec_model = model_zoo.get_model(rec_model_path)
rec_model.prepare(ctx_id=0)

collection = Collection('face_embeddings020304')
collection.load()


def upload_image_to_minio(image_data, bucket_name, content_type):
    try:
        # Create BytesIO object from image data
        image_stream = BytesIO(image_data)

        # Generate unique object name using uuid4()
        object_name = str(uuid4()) + content_type.replace('image/',
                                                          '.')  # Example: '7f1d18a4-2c0e-47d3-afe1-6d27c3b9392e.png'

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


def convert_image_to_embeddingv2(img, face):
    # Detect faces in the image
    rec_model.get(img, face)
    embeddings = face.normed_embedding
    return embeddings.squeeze().tolist()


@csrf_exempt
@permission_classes([IsAuthenticated])
@authentication_classes([JWTAuthentication])
def process_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Get the uploaded image file and the limit parameter from the request
        image_file = request.FILES['image']
        limit = int(request.POST.get('limit', 5))  # Default limit is 10 if not provided
        user_id = request.POST.get('auth_user_id')
        # Read the image file and convert it to an OpenCV format
        image_data = image_file.read()
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
            print("Starting here")
            # Search for the face in Milvus
            vector_ids, distances = search_faces_in_milvus(embedding, limit)
            print("Done")
            # Retrieve metadata for each vector ID
            metadata_objects = Metadata.objects.filter(vector_id__in=vector_ids)

            # Prepare data for response

            milvus_results = [{'vector_id': vector_id, 'distance': round(dist, 5)} for vector_id, dist in
                              zip(vector_ids, distances)]
            metadata_list = [{'vector_id': obj.vector_id, 'iin': obj.iin, 'name': obj.firstname, 'surname': obj.surname,
                              'patronymic': obj.patronymic, 'birth_date': obj.birthdate,  'photo': obj.photo} for obj in metadata_objects]

            # Associate metadata with Milvus results based on vector ID
            for milvus_result in milvus_results:
                vector_id = milvus_result['vector_id']
                metadata = next((item for item in metadata_list if item['vector_id'] == vector_id), None)
                milvus_result['metadata'] = metadata

            keypoints = face.kps
            keypoints = keypoints.tolist()
            bbox = face.bbox

            face_result = {
                'bbox': bbox,
                'keypoints': keypoints,
                'milvus_results': milvus_results
            }
            face_results.append(face_result)

        bucket_name = 'history'
        with image_file.open('rb') as f:
            image_data = f.read()

        uploaded_object_name = upload_image_to_minio(image_data, bucket_name, content_type='image/png')
        user = User.objects.get(id=user_id)
        account = Account.objects.get(user=user)

        SearchHistory.objects.create(
            account=account,
            searchedPhoto=uploaded_object_name,
            created_at=datetime.now()
        )

        return JsonResponse({'faces': face_results})

    return JsonResponse({'error': 'Invalid request method or missing image file'}, status=400)
