from fastapi import FastAPI,Request, HTTPException, Header
from pydantic import BaseModel
from rag import generate_response
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import (
    ApiClient, 
    MessagingApi, 
    Configuration, 
    ReplyMessageRequest, 
    TextMessage, 
)

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

load_dotenv(override=True)
get_channel_secret = os.getenv('CHANNEL_SECRET')
get_access_token = os.getenv('ACCESS_TOKEN')

def get_secret_value(secret_name, default=None):
    """Try reading from a mounted file, fallback to env variable."""
    secret_path = f"/secrets/{secret_name}"
    if os.path.exists(secret_path):  # For Cloud Run with Secret Manager
        with open(secret_path, "r") as f:
            return f.read().strip()
    return os.getenv(secret_name, default)  # Fallback to env variable

get_access_token = get_secret_value('ACCESS_TOKEN')
get_channel_secret = get_secret_value('CHANNEL_SECRET')

configuration = Configuration(access_token=get_access_token)
handler = WebhookHandler(channel_secret=get_channel_secret)

@app.post("/line-chat")
async def callback(request: Request, x_line_signature: str = Header(None)):
    body = await request.body()
    body_str = body.decode('utf-8')
    try:
        handler.handle(body_str, x_line_signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        raise HTTPException(status_code=400, detail="Invalid signature.")

    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        question = event.message.text
        answer = generate_response(question,vector_store_from_client) # TextMessage, FlexMessage เรียก generate_response LLM
        reply_message = TextMessage(text=answer)
        if not reply_message:
            return None

        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[reply_message]
            )
        )

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