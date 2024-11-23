import streamlit as st
from streamlit_option_menu import option_menu

from pygwalker.api.streamlit import StreamlitRenderer
import pandas as pd

from dotenv import load_dotenv
from langchain_teddynote import logging

from utils import print_messages, streamHandler, format_docs, embed_file, get_session_history

from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.runnables import RunnablePassthrough

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from operator import itemgetter

# API KEY 정보로드
load_dotenv()

# Streamlit 페이지 설정
st.set_page_config(page_title='DataDriven Management Strategic', page_icon='🧑🏻‍💻')
st.title('📈 DataDriven Management Strategic')

# 맨처음 chatbbot chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?")
    ]

# 채팅 대화기록을 저장하는 store
if 'store' not in st.session_state:
    st.session_state['store'] = dict()

    
with st.sidebar:
    # 기존 메뉴에 크롤링 봇 추가
    selected = option_menu(
        "Category",
        ["Chat", "Visualization"],
        icons=["robot", "file-earmark-text"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical"
    )
    
    # 크롤링 봇 선택 시 추가 요소 표시
    if selected == "Chat":        
        # session_id 생성
        session_id = st.text_input(label='**Chat ID**', value="code")
        
        # 파일 업로드
        file = st.file_uploader(
            "**Data File Upload**",
            type="pdf"
        )
        
        # file이 있다면, embeddied
        if file:
            retriever = embed_file(file)
        
        clear_button = st.button('Reset')
        if clear_button:
            st.session_state['messages'] = []
            st.session_state['store'] = dict()
            st.experimental_rerun()
    else:
        # 파일 업로드
        file = st.file_uploader(
            "**Data File Upload**",
            type="csv"
        )
    

# 이전 대화 기록 출력
print_messages()

if selected == "Chat":
    # 사용자 입력이 있을 경우
    if user_input := st.chat_input('메시지를 입력해 주세요.'):
        # 사용자 메시지 출력
        st.chat_message('user').write(f'{user_input}')
        st.session_state['messages'].append(ChatMessage(role='user', content=user_input))
        
        # AI 응답 출력
        with st.chat_message('assistant'):
            stream_handler = streamHandler(st.empty())
            
            # LLM 생성
            llm = ChatOllama(
                model='Bllossom-8B-Q4_K_M:latest',
                streaming=True,
                callbacks=[stream_handler],
            )

            if file is not None:
                # PDF가 있는 경우의 프롬프트 템플릿
                prompt = ChatPromptTemplate.from_messages([
                    ("system",
                    '당신은 질문에 한글로 답변하는 업무를 수행하는 도우미입니다. 제공된 문맥 정보를 활용하여 질문에 답변하세요. 답을 모를 경우, 모른다고 말하세요. 최대 두 문장으로 답변을 작성하고, 간결하고 명확하게 표현하세요.'
                    ),
                    MessagesPlaceholder(variable_name="history"),
                    ('human', '{question}')
                ])

                
                # RAG 체인 생성
                # 여기서 history 넣는게 핵심.
                rag_chain = (
                    {'context': itemgetter('question') | retriever, "question": RunnablePassthrough(), "history": itemgetter('history')}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                chain_with_memory = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key='question',
                    history_messages_key='history'
                )

                # 세션별 구성으로 체인 실행
                response = chain_with_memory.invoke(
                    {'question': user_input},
                    config={'configurable': {'session_id': session_id}}
                )
                
            # pdf 입력이 없을 때 (RAG 사용 X)
            else:
                # PDF가 없는 경우의 기본 프롬프트 템플릿
                basic_prompt = ChatPromptTemplate.from_messages([
                    ("system", "질문에 대해서 간단히 답변해주세요."),
                    MessagesPlaceholder(variable_name="history"),
                    ('human', '{question}')
                ])

                # 기본 체인 생성
                basic_chain = (
                    basic_prompt
                    | llm
                    | StrOutputParser()
                )
                
                chain_with_memory = RunnableWithMessageHistory(
                    basic_chain,
                    get_session_history,
                    input_messages_key='question',
                    history_messages_key='history'
                )

                # 세션별 구성으로 체인 실행
                response = chain_with_memory.invoke(
                    {'question': user_input},
                    config={'configurable': {'session_id': session_id}}
                )

            # AI 응답 메시지
            msg = response
            st.session_state['messages'].append(ChatMessage(role='assistant', content=msg))
else:
    if file is not None:
            # 파일이 업로드된 경우에만 csv 파일 읽기
            try:
                @st.cache_resource
                def get_pyg_renderer() -> StreamlitRenderer:
                    df = pd.read_csv(file)
                    return StreamlitRenderer(df, spec_io_mode="rw", renderer="svg")
                
                renderer = get_pyg_renderer()
                
                # 탭 구성
                tab1, tab2, tab3 = st.tabs(["Explorer", "Data Profiling", "Charts"])

                with tab1:
                    renderer.explorer(key="explorer_tab1")

                with tab2:
                    renderer.explorer(default_tab="data", key="explorer_tab2")

                with tab3:
                    try:
                        st.subheader("Registered per Weekday")
                        renderer.chart(0)  # 0번 인덱스의 데이터가 있는지 확인 필요

                        st.subheader("Registered per Day")
                        renderer.chart(1)  # 1번 인덱스의 데이터가 있는지 확인 필요
                    except IndexError:
                        st.error("차트를 그릴 데이터가 부족합니다. 파일을 확인해주세요.")
                    except Exception as e:
                        st.error(f"예기치 못한 오류: {e}")
            except Exception as e:
                st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    else:
        st.write("")
