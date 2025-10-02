from fastapi import FastAPI
from pydantic import BaseModel
from rag import generate_response
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from langchain_chroma import Chroma

app = FastAPI()

client = chromadb.PersistentClient(path="./chroma_db")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vector_store_from_client = Chroma(
    client=client,
    collection_name="chatbot_docs",
    embedding_function=embedding_model,
)

class Item(BaseModel):
    prompt: str

@app.get("/")
def hello():
    return {"message": "Hello World!"}

@app.post("/chat")
def chat_llm(item: Item):
    answer = generate_response(item.prompt,vector_store_from_client)
    return {"answer": answer}

#.venv\Scripts\activate
#uvicorn api_manager:app --reload

#ใช้ไลน์ รัน ngrok http 8000 เปลี่ยนลิ้งค์เอาลิ้งไปเปลี่ยนที่console