import streamlit as st
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings

# API KEY 정보로드
load_dotenv()

def print_messages():
    # 이전 대화기록 출력 코드
    if 'messages' in st.session_state and len(st.session_state['messages']) > 0:
        for chat_message in st.session_state['messages']:
            st.chat_message(chat_message.role).write(chat_message.content)


class streamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text = ''):
        self.container = container
        self.text = initial_text
        
    def on_llm_new_token(self, token: str, **kwarngs) -> None:
        self.text += token
        self.container.markdown(self.text)

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)


def embed_file(file):
    # Streamlit의 UploadedFile 객체에서 파일 내용을 읽어 로컬 디렉토리에 저장
    file_path = f"./.cache/files/{file.name}"  # 파일을 저장할 경로를 설정
    with open(file_path, "wb") as f:  # 바이너리 쓰기 모드로 파일 열기
        f.write(file.read())  # 파일 내용을 로컬에 저장
    
    # PyPDFLoader를 사용하여 PDF 파일을 로드
    loader = PyPDFLoader(file_path)
    
    # PDF 파일을 로드하여 문서 객체로 변환
    docs = loader.load()
    
    # 텍스트를 분할하기 위한 설정: 각 텍스트 조각의 최대 크기와 중복 크기 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 각 텍스트 조각의 최대 길이
        chunk_overlap=50  # 인접한 텍스트 조각 간의 중복 길이
    )

    # 문서를 작은 텍스트 조각들로 분할
    splits = text_splitter.split_documents(docs)
    
    # 분할된 문서를 FAISS 벡터스토어에 임베딩
    vectorstore = FAISS.from_documents(
        documents=splits,  # 분할된 문서 리스트
        embedding=OpenAIEmbeddings()  # OpenAI 임베딩 모델을 사용해 벡터화
    )
    
    # 벡터스토어에서 문서를 검색할 수 있는 리트리버 생성
    retriever = vectorstore.as_retriever()
    
    # 생성된 리트리버를 반환
    return retriever


# 세션 기록을 가져오거나 새로 생성하는 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state['store']:
        st.session_state['store'][session_ids] = ChatMessageHistory()
    return st.session_state['store'][session_ids]