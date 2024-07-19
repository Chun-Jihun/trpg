from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

st.set_page_config(
    page_title="TRPG Main Page",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("TRPG")

st.markdown(
    """
게임 시작 페이지 (메인)
"""
)

# 변수 저장 공간 초기화
if 'messages' not in st.session_state:
    st.session_state["messages"] = []
if 'health' not in st.session_state: # 각 칸에는 이미지 path를 넣어서 0~3까지 표현
    st.session_state["health"] = ['', '', '', '']
    st.session_state["mental"] = ['', '', '', '']
    st.session_state["money"] = ['', '', '', '']

health = st.session_state["health"]
mental = st.session_state["mental"]
money = st.session_state["money"]
st.image(health) # 체력 스텟 이미지
st.image(mental) # 멘탈 스텟 이미지
st.image(money) # 자금 스텟 이미지
st.image('') # 상황 묘사 이미지

RuleBook = 'Rule_Book.docx'
retriever = embed_file(RuleBook)

send_message("I'm ready! Ask away!", "ai", save=False)

message = st.chat_input("Ask anything about your file...")

if message:
    send_message(message, "human")
    chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
    )
    with st.chat_message("ai"):
        chain.invoke(message)


paint_history()

