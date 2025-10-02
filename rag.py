from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_ollama.llms import OllamaLLM
import chromadb
from db_manager import retrieve_docs

def generate_llm_response(context, question):
    system_msg = SystemMessagePromptTemplate.from_template(
        """คุณเป็นผู้ช่วยของมหาวิทยาลัย ทำหน้าที่ตอบคำถามให้นักศึกษา 
        โดยใช้ข้อมูลภายในบริบทที่ได้รับเท่านั้น โดยดูจากความเกี่ยวข้องกับคำถามมากที่สุด 
        ตอบเป็นภาษาที่สุภาพ เข้าใจง่าย เหมาะกับนักศึกษา หากบริบทที่ได้รับเกี่ยวกับห้องเรียน
        ต้องเลือกตอบเพียงห้องเดียวเท่านั้น สามารถตอบด้วยอิโมจิได้ 
        และสามารถลงท้ายด้วยการแนะนำเพิ่มเติมเกี่ยวกับหัวข้อคำถามนั้น"""
    )
    human_msg = HumanMessagePromptTemplate.from_template(
        "\n{context}\n\nQuestion:\n{question}\n"
    )
    prompt = ChatPromptTemplate.from_messages([system_msg,human_msg])

    model = OllamaLLM(model="scb10x/typhoon2.1-gemma3-4b")
    chain = prompt | model
    final_prompt = prompt.format(context=context, question=question)
    print("\nfinal prompt:\n", final_prompt)
    response = chain.invoke({"context": context, "question": question})
    return response

def generate_response(question,vector_store_from_client):
    context = retrieve_docs(question,vector_store_from_client)
    answer = generate_llm_response(context, question)
    print(f"Answer:\n{answer}")
    return answer