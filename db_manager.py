from langchain_chroma import Chroma
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
from langchain_core.documents import Document
import json
from datetime import datetime

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("chatbot_docs")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vector_store_from_client = Chroma(
    client=client,
    collection_name="chatbot_docs",
    embedding_function=embedding_model,
)

def add_docs(path,vector_store_from_client):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    for i,item in enumerate(data, start=1):
        now = datetime.now()
        formatted = now.strftime("%d/%m/%Y %H:%M")
        doc = Document(
            page_content=item["content"],
            metadata=item["metadata"]
        )
        documents.append(doc)

    for doc in documents:
        print(doc.page_content)
        print(doc.metadata)
        print("-"*30)

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store_from_client.add_documents(documents=documents, ids=uuids)

def show_all_docs():
    print("Collection name:", collection.name)
    print("Number of documents:", collection.count())
    # print("Metadata fields:", collection.get()["metadatas"]) 

    documents = collection.get()
    for doc_id, content, metadata in zip(documents["ids"], documents["documents"], documents["metadatas"]):
        print(f"ID: {doc_id}")
        print(f"Content: {content}")
        print(f"Metadata: {metadata}")
        print("-" * 30)

def delete_docs(vector_store_from_client):
    # uuids_to_delete = [
    #     "1fd8ecfe-3b21-4899-a775-27e71f865e75"
    # ]
    # vector_store_from_client.delete(ids=uuids_to_delete)

    collection.delete(where={})

def retrieve_docs(question,vector_store_from_client):
    results = vector_store_from_client.similarity_search_with_score(
        question, k=3
    )
    print("\nsearch results:")
    context = ""
    for res, score in results: #สถานที่
        print(f"\n{res.page_content}")
        print(f"[{res.metadata}] [SIM={score:3f}]")
        if(res.metadata.get("category")!="ห้องเรียน"):
            context += f"\n{res.page_content}\n"
        else:
            context += f"\n{res.page_content} {res.metadata.get("building")} {res.metadata.get("floor")} {res.metadata.get("faculty")}"
    
    return context


# add_docs("docs-FAQ/.....json",vector_store_from_client)

# delete_docs(vector_store_from_client)

# show_all_docs()

