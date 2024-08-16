from celery import shared_task
import cv2
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from django.http import JsonResponse, HttpResponse
import time
import torch
import os
import uuid
import sys
import numpy as np
from mtcnn import MTCNN
from datetime import datetime
from io import BytesIO
from uuid import uuid4
import torchvision.transforms as transforms
from minio import Minio
import io
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
from pymilvus import Milvus, CollectionSchema, FieldSchema, DataType, Collection, connections, utility
import base64
import insightface
from insightface.app.common import Face
from insightface.model_zoo import model_zoo
from .models import Person, Gallery
from .serializers import PersonSerializer
from dotenv import load_dotenv
import keras.applications
from facenet_pytorch import InceptionResnetV1

load_dotenv()

MILVUS_HOST = os.environ.get('MILVUS_HOST')
MILVUS_PORT = os.environ.get('MILVUS_PORT')
client = Milvus(MILVUS_HOST, MILVUS_PORT)
rec_model_path = '/root/eTanuReincarnationLinux/metadata/insightface/models/w600k_mbf.onnx'
detector = MTCNN(steps_threshold=[0.7, 0.8, 0.9], min_face_size=40)
rec_model = model_zoo.get_model(rec_model_path)
rec_model.prepare(ctx_id=0)
minio_client = Minio(
    endpoint='192.168.122.110:9000',
    access_key='minioadmin',
    secret_key='minioadmin',
    secure=False  # Set to True if using HTTPS
)


def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
        return base64_encoded


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


def get_image_metadata(image_path):
    with Image.open(image_path) as img:
        metadata = {
            'format': img.format,
            'mode': img.mode,
            'size': img.size,
            'info': img.info
        }
    return metadata


def alignment_procedure(img, face):
    x, y, w, h = face['box']
    face_coords = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    left_eye = face["keypoints"]["left_eye"]
    right_eye = face["keypoints"]["right_eye"]

    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_image = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

    rotated_face_coords = cv2.transform(face_coords.reshape(-1, 1, 2), rotation_matrix).reshape(-1, 2)

    # Convert rotated_face_coords to integers
    rotated_face_coords = rotated_face_coords.astype(int)

    aligned_face = aligned_image[int(abs(min(rotated_face_coords[:, 1]))):int(abs(max(rotated_face_coords[:, 1]))),
                   int(abs(min(rotated_face_coords[:, 0]))):int(abs(max(rotated_face_coords[:, 0])))]

    return aligned_face


def convert_image_to_embedding(img, face):
    # Detect faces in the image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_image = alignment_procedure(img_rgb, face)
    resized_face = cv2.resize(face_image, (112, 112))
    resized_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
    resized_face = np.transpose(resized_face, (2, 0, 1))  # Change from HWC to CHW format
    face_embedding = rec_model.get(resized_face)

    return face_embedding


def convert_image_to_embeddingv2(img, face):
    # Detect faces in the image
    rec_model.get(img, face)
    embeddings = face.normed_embedding
    return embeddings


@shared_task
def process_image(image_path, partition_name):
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )

    collection = Collection("face_embeddings")

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    bbox = faces[0]['box']
    det_score = faces[0]['confidence']
    kps_dict = faces[0]['keypoints']
    kps = np.array([list(kps_dict.values())]).squeeze()

    face = Face(bbox=bbox, kps=kps, det_score=det_score)

    embedding = convert_image_to_embeddingv2(image_rgb, face)
    # need to upload to minio (base64 to switch with minio)
    metadata = get_image_metadata(image_path)

    # Generate UUID for the embedding
    embedding_id = str(uuid.uuid4())
    first_name = None
    surname = None
    patronymic = None
    if metadata['info']['FIO'] is not None:
        full_name = metadata['info']['FIO']
        name_components = full_name.split()
        if len(name_components) == 3:
            surname, first_name, patronymic = name_components
        elif len(name_components) == 2:
            surname, first_name = name_components
            patronymic = ""

    bucket_name = 'photos'
    directory_name = partition_name
    uploaded_object_name = upload_image_to_minio(image_data, bucket_name, content_type='image/png', directory_name=directory_name)
    Person.objects.create(
        iin=None,
        firstname=first_name,
        surname=surname,
        patronymic=patronymic,
    )

    data = [
        [embedding_id],
        [embedding]
    ]

    collection.insert(data, partition_name=partition_name)
    print(f"Done: {image_path}")
    collection_name = "face_embeddings"
    client.flush([collection_name])

@shared_task
def process_image_from_row(row_data, partition_name):
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    collection_name = "face_embeddings"
    collection = Collection(collection_name)
    hex_photo = row_data['photo']

    hex_values = hex_photo.replace('\\x', '').split('\\')
    # Convert hex values to byte array
    image_data = bytearray.fromhex(''.join(hex_values))
    # Convert bytes to numpy array
    image_np = np.asarray(image_data, dtype=np.uint8)
    # Decode the image using OpenCV
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    bbox = faces[0]['box']
    det_score = faces[0]['confidence']
    kps_dict = faces[0]['keypoints']
    kps = np.array([list(kps_dict.values())]).squeeze()

    face = Face(bbox=bbox, kps=kps, det_score=det_score)

    embedding = convert_image_to_embeddingv2(image_rgb, face)
    # need to upload to minio (base64 to switch with minio)

    # Generate UUID for the embedding
    embedding_id = str(uuid.uuid4())
    first_name = row_data['first_name']
    surname = row_data['last_name']
    patronymic = row_data['patronymic']
    iin = row_data['iin']
    birthdate = row_data['birth_date']
    birthdate = datetime.strptime(birthdate, '%Y-%m-%d')
    # Format the date in the desired format
    reformatted_birthdate = birthdate.strftime('%Y-%m-%d')
    bucket_name = 'photos'
    directory_name = partition_name
    uploaded_object_name = upload_image_to_minio(image_data, bucket_name, content_type='image/png',directory_name=directory_name)
    if Person.objects.filter(iin=iin).exists():
        existedPerson = Person.objects.get(iin=iin)

        Gallery.objects.create(
            vector_id=embedding_id,
            photo=uploaded_object_name,
            personId=existedPerson
        )
    else:
        person_instance = Person.objects.create(
            iin=iin,
            firstname=first_name,
            surname=surname,
            patronymic=patronymic,
            birthdate=reformatted_birthdate,
        )

        Gallery.objects.create(
            vector_id=embedding_id,
            photo=uploaded_object_name,
            personId=person_instance
        )

    data = [
        [embedding_id],
        [embedding]
    ]

    collection.insert(data, partition_name=partition_name)
    print(f"Done: {row_data['iin']}")
