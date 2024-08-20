from django.apps import AppConfig
from pymilvus import Milvus, Collection
from dotenv import load_dotenv
import os

load_dotenv()


class SearchConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'search'

    def ready(self):
        MILVUS_HOST = os.environ.get('MILVUS_HOST')
        MILVUS_PORT = os.environ.get('MILVUS_PORT')
        client = Milvus(MILVUS_HOST, MILVUS_PORT)
        collection = Collection('face_embeddings')
        collection.load()
        self.collection = collection
