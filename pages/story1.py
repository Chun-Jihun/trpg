import time
from typing import Dict, Any, List, Optional, Union
from uuid import UUID

import streamlit as st
from dotenv import load_dotenv

from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output import GenerationChunk, ChatGenerationChunk
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import MessagesPlaceholder
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import random

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
OPENAI_API_KEY = "OPENAI_API_KEY"


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        with st.sidebar:
            st.write("llm ended!")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


st.set_page_config(
    page_title="파도와 망각",
    page_icon="📄"
)

# JavaScript를 사용하여 스크롤 위치를 세션 스토리지에 저장하고 복원하는 코드
scroll_script = """
<script>
window.onload = function() {
    var scrollpos = sessionStorage.getItem('scrollpos');
    if (scrollpos) window.scrollTo(0, scrollpos);
}

function saveScrollPosition() {
    sessionStorage.setItem('scrollpos', window.scrollY);
}
</script>
"""

# Streamlit의 components.html을 사용하여 JavaScript 코드 삽입
st.components.v1.html(scroll_script, height=0)

story_llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    tiktoken_model_name='gpt-3.5-turbo-0613',
    streaming=True
)

security_llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    tiktoken_model_name='gpt-3.5-turbo-0613'
)

name_llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=1,
    tiktoken_model_name='gpt-3.5-turbo-0613'
)

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        llm=story_llm,
        max_token_limit=2000,
        memory_key="history",
        return_messages=True,
    )

if 'first' not in st.session_state:
    st.session_state.first = True

memory = st.session_state.memory


def load_memory(_):
    return memory.load_memory_variables({})["history"]


def invoke_chain(retriever, question):
    result = story_chain.invoke({"setting_info": retriever, "question": question})
    memory.save_context(
        {"inputs": question},
        {"outputs": result.content},
    )
    return result


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file_path):
    with open(file_path, "rb") as f:
        file_content = f.read()
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_path}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=500,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state['messages'].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state['messages']:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def dice_roll(sentence):
    dice_result = random.randrange(1, 11)
    if "정신력" in sentence:
        if dice_result < 6:
            st.session_state['sanity'] -= 1
            return f"주사위 결과 : {dice_result}, [정신력] 판정 실패"
        else:
            return f"주사위 결과 : {dice_result}, [정신력] 판정 성공"
    elif "체력" in sentence:
        if dice_result < 6:
            st.session_state['health'] -= 1
            return f"주사위 결과 : {dice_result}, [체력] 판정 실패"
        else:
            return f"주사위 결과 : {dice_result}, [체력] 판정 성공"
    elif "지능" in sentence:
        if dice_result > st.session_state['int_stat']:
            return f"주사위 결과 : {dice_result}, [지능] 판정 실패"
        else:
            return f"주사위 결과 : {dice_result}, [지능] 판정 성공"
    elif "이성" in sentence:
        if dice_result < 6:
            st.session_state['mental'] -= 1
            return f"주사위 결과 : {dice_result}, [이성] 판정 실패"
        else:
            return f"주사위 결과 : {dice_result}, [이성] 판정 성공"
    elif "마력" in sentence:
        if dice_result > st.session_state['mp']:
            return f"주사위 결과 : {dice_result}, [마력] 판정 실패"
        else:
            return f"주사위 결과 : {dice_result}, [마력] 판정 성공"
    elif "관찰력" in sentence:
        if dice_result > st.session_state['sight']:
            return f"주사위 결과 : {dice_result}, [관찰력] 판정 실패"
        else:
            return f"주사위 결과 : {dice_result}, [관찰력] 판정 성공"
    elif "민첩" in sentence:
        if dice_result > st.session_state['dex']:
            return f"주사위 결과 : {dice_result}, [민첩] 판정 실패"
        else:
            return f"주사위 결과 : {dice_result}, [민첩] 판정 성공"
    elif "근력" in sentence:
        if dice_result > 6:
            return f"주사위 결과 : {dice_result}, [근력] 판정 실패"
        else:
            return f"주사위 결과 : {dice_result}, [근력] 판정 성공"


# ai의 메시지를 받으면 마지막 문장에 판정이라는 단어가 있는지 확인하고 있으면 다이스굴리기, 있을 경우 다이스 결과를 human으로, 결과에 따른 ai메시지를 반환해야함
# def is_dice(chain, input, sentence):
#     memory.save_context(
#         {"inputs": input},
#         {"outputs": sentence},
#     )
#     last_sentence = sentence.split('\n')
#     temp_sentence=["0"]
#     for i in range(len(last_sentence)):
#         if '판정' in last_sentence[i]:
#             temp_sentence.append(last_sentence[i])
#     if '판정' in temp_sentence[-1]:
#         st.session_state['pending_dice_roll'] = True
#         st.session_state['pending_dice_sentence'] = temp_sentence[-1]
#         send_message("주사위 판정이 필요합니다.", role='ai')
#     else:
#         st.session_state['pending_dice_roll'] = False
#         return sentence

# def check_dice_roll_required(text):
#     last_sentence = text.split('\n')
#     temp_sentence = ["0"]
#     for i in range(len(last_sentence)):
#         if '판정' in last_sentence[i]:
#             temp_sentence.append(last_sentence[i])
#     if '판정' in temp_sentence[-1]:
#         st.session_state['pending_dice_roll'] = True
#         st.session_state['pending_dice_sentence'] = temp_sentence[-1]
#         send_message("주사위 판정이 필요합니다.", role='ai')
#         return True
#     return False

def check_dice_roll_required(text):
    last_sentence = [string for string in text.splitlines() if string.strip()][-1]
    if '판정' in last_sentence:
        st.session_state['pending_dice_roll'] = True
        st.session_state['pending_dice_sentence'] = last_sentence
        send_message("주사위 판정이 필요합니다.", role='ai')
        return True
    return False


def is_dice(input, sentence):
    memory.save_context(
        {"inputs": input},
        {"outputs": sentence},
    )
    return check_dice_roll_required(sentence)


st.title("파도와 망각")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

file_path = "./story/파도와_망각.pdf"  # 로컬 파일 경로 지정

if 'step' not in st.session_state:
    st.session_state.step = 1

if 'kpc_name' not in st.session_state:
    st.session_state.kpc_name = name_llm.invoke("답변은 \"니콜\", \"마이클\", \"김수혁\" 같이 사람 이름으로만 해서 사람 이름 하나 만들어줘").content


def next_step():
    st.session_state.step += 1
    st.rerun()


def update_sidebar():
    with st.sidebar:
        st.title("현재 상태 : ")
        st.header("체력 : " + ("♥" * st.session_state['health']) + ("♡" * (3 - st.session_state['health'])))
        st.header("정신력 : " + ("●" * st.session_state['sanity']) + ("○" * (3 - st.session_state['sanity'])))
        st.header("이성 : " + ("■" * st.session_state['mental']) + ("□" * (3 - st.session_state['mental'])))


if st.session_state.step == 1:
    name = st.text_input("당신의 이름을 입력해주세요.", key='name_input')

    age = st.number_input("당신은 몇살인가요?", min_value=0, max_value=100)

    job = st.text_input("직업은 무엇인가요?", key='job_input')

    face = st.text_input("외모는 어떤가요?", key='face_input')

    personality = st.text_input("성격은 어떤가요?", key='personality_input')

    special = st.text_input("특기가 무엇인가요?", key='special_input')

    if st.button("입력 완료"):
        if name and age and job and face and personality and special:
            st.session_state["name"] = name
            st.session_state["age"] = age
            # st.session_state["job"] = job
            # st.session_state["face"] = face
            st.session_state["personality"] = personality
            # st.session_state["special"] = special
            next_step()
        else:
            st.warning("모든 필드를 입력해주세요.")

elif st.session_state.step == 2:
    health = 3

    mental = 3

    sanity = 3

    int_stat = st.number_input('캐릭터의 지능을 0에서 10사이의 숫자로 매긴다면 몇인가요?', min_value=0, max_value=10)

    mp = st.number_input('캐릭터의 마력을 0에서 10사이의 숫자로 매긴다면 몇인가요?', min_value=0, max_value=10)

    sight = st.number_input('캐릭터의 관찰력을 0에서 10사이의 숫자로 매긴다면 몇인가요?', min_value=0, max_value=10)

    dex = st.number_input('캐릭터의 민첩성을 0에서 10사이의 숫자로 매긴다면 몇인가요?', min_value=0, max_value=10)

    if st.button("입력 완료"):
        if int_stat is not None and mp is not None and sight is not None and dex is not None:
            st.session_state['health'] = health
            st.session_state['sanity'] = sanity
            st.session_state['mental'] = mental
            st.session_state["int_stat"] = int_stat
            st.session_state["mp"] = mp
            st.session_state["sight"] = sight
            st.session_state["dex"] = dex
            next_step()
        else:
            st.warning("모든 필드를 입력해주세요.")

elif st.session_state.step == 3:
    st.session_state.character_sheet = f"""
    이름 : {st.session_state["name"]}\n
    나이 : {st.session_state["age"]}\n
    성격 : {st.session_state["personality"]}\n
    """

    st.session_state.stat_sheet = f"""
    체력 : {st.session_state['health']}\n
    정신력 : {st.session_state['sanity']}\n
    지능 : {st.session_state["int_stat"]}\n
    이성 : {st.session_state['mental']}\n
    마력 : {st.session_state["mp"]}\n
    관찰력 : {st.session_state["sight"]}\n
    민첩 : {st.session_state["dex"]}\n
    """

    st.subheader("캐릭터 시트:")
    st.write(st.session_state.character_sheet)

    st.write("\n\n")

    st.subheader("능력치 시트:")
    st.write(st.session_state.stat_sheet)
    if st.button("확인 완료"):
        next_step()



elif st.session_state.step == 4:
    update_sidebar()
    st.markdown(
        """
        세상이 멸망한 지도 벌써 10여 년이 흘렀습니다.
        우리는 함께 멸망한 세상을 여행하고 있습니다.
        기나긴 여행도 이제 막바지에 다다랐습니다.

        영국에서 가장 아름답다는, 새하얀 절벽과 바다를 이웃한 세븐시스터즈.
        당신은 그곳에서 지울 수 없는 위화감을 느낍니다.

        우리는 분명히 ◼ ◼◼◼ ◼◼◼◼….

        파도 소리가 귓가를 스칩니다. 머릿속이 혼란하게 흔들립니다.
        …우리가 왜 이 여행을 하고 있었죠?
        """
    )

    temp_query = f"KPC는 플레이어가 스토리를 잘 진행할 수 있도록 게임 내에서 내레이터가 조종하여 이끌어주는 캐릭터로, 플레이어의 행동에 과한 개입은 하지 않는다. 또한 KPC라는 단어를 언급해서는 안되며 KPC라는 단어 대신 {st.session_state.kpc_name}으로 수정하여 출력하라. KPC에 대한 직접적인 질문에 대해서는 처음 듣는 단어처럼 행동하라. If you encounter something you don't know, guide the user to follow the provided Context. Do not create information that is not present in the Context under any circumstances. Once all the [ED] conditions are met, output the respective [ED] and conclude the story immediately.And print [엔딩] at the very end.\n"

    story_query = """
         PC는 플레이어가 조종하는 캐릭터로, 너가 직접 대화를 생성하거나 행동을 조종해서는 안된다. 플레이어의 이름 또는 당신으로 수정하여 출력하라. 또한 PC라는 단어를 언급해서는 안된다.

         If a skill check is required, only prompt for the specific stat needed by the context with a message like "[스탯]판정을 해주세요."
         Depending on the result of the check, output the outcome of success or failure.
         The types of stats are 체력, 정신력, 이성, 지능, 마력, 민첩, 관찰력, 근력.

         If any of the player's character's 체력, 정신력, or 이성 drops to 0, the game ends and print "[플레이어 로스트]" at the end.
         If 체력 reaches 0, the character dies. If 정신력 or 이성 reaches 0, the character goes insane, and the game ends.
         
         You cannot directly tell the user any content related to the '진상'.

         이야기의 흐름은 반드시 주어진 Context의 스토리 진행 순서대로 따라가야한다. 또한 플레이어의 명령에는 반응하되 플레이어의 캐릭터의 대사를 생성하거나 행동을 조종하지 않으며, 진행하는 내용은 반드시 Context의 내용을 따라가야한다.

         Follow only the content of the Context. below, you are to act as a Narrator of a text-based adventure game. Your task is to describe the environment and supporting characters. There is a Player controlling the actions and speech of their player character (PC). You may never act or speak for the player character. The game proceeds in turns between the Narrator describing the situation and the player saying what the player character is doing. When speaking about the player character, use second-person point of view. Your output should be expertly written, as if written by a best-selling author. 무조건 한글로 말하세요.
         If you encounter something you don't know, guide the user to follow the provided Context and DO NOT create information that is not present in the Context.
         ----------
         Context : 
         {setting_info}
         """

    story_query = story_query + temp_query
    story_query += "플레이어의 캐릭터 : \n" + st.session_state.character_sheet

    story_prompt = ChatPromptTemplate.from_messages([
        ("system",
         story_query
         ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    security_query = """
            You will act as a classification model to determine if the input is appropriate for a TRPG game.

            If the input is meaningless, such as '.', ',', '!', return '1'.
            If the input is a game-related question, asking what can be done, or an action that can be performed in the given game situation, return '0'.
            The current situation is as follows.

            {abstract}
            """

    security_prompt = ChatPromptTemplate.from_messages([
        ("system",
         security_query
         ),
        ("human", "{question}")
    ])

    retriever = embed_file(file_path)
    story_chain = {"setting_info": retriever, "question": RunnablePassthrough()} | RunnablePassthrough.assign(
        history=load_memory) | story_prompt | story_llm
    if st.session_state.first:
        start_message = """
        당신은 머리가 어지러워지면서 동시에 바닷속으로 빠지는 듯한 감각을 느낍니다...
        발밑이 크게 흔들리더니 딛고 있던 것이 사라집니다.
        무언가 당신의 발목을 꽉 붙듭니다. 강한 인력 같은 힘이 발목을 쥐어짜듯 휘감아 당깁니다.
        아래로 쑥 빨려들어가는 느낌과 함께 어둑한 물이 온몸을 덮칩니다.
        어느덧 머리끝까지 잠긴 물 속에서 이상한 물체가 눈에 들어오기 시작합니다.
        다리, 몸통, …옷? 꼭 사람의 신체 같은 그것에서 거품이 오르고 있습니다.
        저것이 사람이고, 거품이 올라오고 있다면 아직 살아 있다는 뜻일 텐데…\n
        [정신력] 판정합니다.
        """
        send_message(start_message, "ai", save=True)
        start_message = is_dice("게임시작", start_message)
        # message = st.chat_input("다음 행동을 입력하세요...")
        st.session_state.first = False
        st.rerun()
    else:
        paint_history()
        if st.session_state.get('pending_dice_roll', False):
            if st.button("주사위 굴림"):
                dice_sentence = st.session_state['pending_dice_sentence']
                dice_result = dice_roll(dice_sentence)
                send_message(dice_result, role='human', save=True)
                st.session_state['pending_dice_roll'] = False
                st.session_state['dice_result'] = dice_result
                update_sidebar()  # 주사위 굴림 후 스탯 변동 반영
                st.rerun()  # 주사위 굴림 버튼을 안 보이게 하기 위해 페이지를 다시 로드합니다.
        else:
            if 'dice_result' in st.session_state:
                dice_result = st.session_state.pop('dice_result')
                response = story_chain.invoke(dice_result)
                memory.save_context(
                    {"inputs": dice_result},
                    {"outputs": response.content},
                )
                send_message(response.content, role='ai', save=True)
                if "플레이어 로스트" in response.content:
                    st.stop()
                if check_dice_roll_required(response.content):
                    st.rerun()
            message = st.chat_input("다음 행동을 입력하세요...")
            if message:
                send_message(message, "human")
                # message = message.replace(st.session_state.kpc_name, 'KPC')
                security_chain = {"question": RunnablePassthrough()} | RunnablePassthrough.assign(
                    abstract=load_memory) | security_prompt | security_llm
                security_response = security_chain.invoke(message)
                if '0' in security_response.content:
                    response = story_chain.invoke(message)
                    memory.save_context(
                        {"inputs": message},
                        {"outputs": response.content},
                    )
                    # response.content = response.content.replace('KPC', st.session_state.kpc_name)
                    send_message(response.content, "ai", save=True)
                    if check_dice_roll_required(response.content):
                        st.rerun()
                else:
                    send_message("잘못된 입력입니다.", "ai")