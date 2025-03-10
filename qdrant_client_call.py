import os
from qdrant_client import QdrantClient

from openai import AzureOpenAI, OpenAI

from qdrant_client.http import models
class QdrantHandler:
    def __init__(self, qdrant_host='localhost', qdrant_port=6333, collection_name='chatbot_world_data', embedding_model='text-embedding-3-small'):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.client = OpenAI(api_key=self.api_key)
        self.qdrant_client = QdrantClient(qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        # self.api_key=os.environ.get("AZURE_OPENAI_API_KEY"),  
        # self.api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        # self.azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        # self.emb_engine = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT")
        # self.client_azure = AzureOpenAI(
        #     api_key=os.environ.get("AZURE_OPENAI_API_KEY"),  
        #     api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        #     azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        #     )
    def create_collection(self):
        print(f"Creating collection '{self.collection_name}'...")
        if self.qdrant_client.collection_exists(self.collection_name):
            print(f"Collection '{self.collection_name}' already exists!")
        else:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' created!")
        

    def add_url_image_to_qdrant(
    self,
    file_url_img="data/texts/country_flags.txt",
    file_description="data/texts/description_flag.txt"
):
        # Đọc file urls
        urls = []
        try:
            with open(file_url_img, "r", encoding="utf-8") as file:
                urls = [line.strip() for line in file]
        except UnicodeDecodeError as e:
            print(f"Error decoding file: {e}")
            return

        # Đọc file tag_name
       

        # Đọc file texts (mô tả)
        texts = []
        try:
            with open(file_description, "r", encoding="utf-8") as file:
                texts = [line.strip() for line in file]
        except UnicodeDecodeError as e:
            print(f"Error decoding file: {e}")
            return
        print("Length of texts: ",len(texts))
        print("Length of urls: ",len(urls))
        # Tạo vector embedding từ texts
        vectors = [self.get_embedding(text) for text in texts]
        ids = list(range(len(texts)))

        if vectors:
            print("Adding data to Qdrant...")
            # Gộp url, tag_name và text thành payload chung
            payloads = []
            for i in range(len(texts)):
                payloads.append({
                    "url": urls[i],
                    "description": texts[i]
                })

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads
                )
            )
            print("Data added to Qdrant!")
        else:
            print("No vectors generated, Qdrant upsert skipped.")


    def get_embedding_azure(self,text):  
        text = text.replace("\n", " ")  
        
        embedding_response = self.client_azure.embeddings.create(input = [text], model=self.emb_engine).data[0].embedding
        return embedding_response  

    def get_embedding(self, text):
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding


qdhandler = QdrantHandler(collection_name='chatbot_world_img_data')
# qdhandler.create_collection()
qdhandler.add_url_image_to_qdrant(file_url_img="data/texts/country_flag_links.txt",file_description="data/texts/description_flag_sorted.txt")