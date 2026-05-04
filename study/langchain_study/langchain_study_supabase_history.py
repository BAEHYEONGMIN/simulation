import os
import sys

# 상위 폴더의 config 및 DB 접속 정보 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CHAT_MODEL, GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY

from supabase import create_client, Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# ============== [핵심 1] 커스텀 DB 연결 비서 만들기 ==============
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class SupabaseChatMessageHistory(BaseChatMessageHistory):
    """
    랭체인이 이해할 수 있도록 Supabase DB '읽기/쓰기' 방식을 정의해두는 전용 클래스입니다.
    이 클래스 하나 덕분에, 앞으로 챗봇 중심 코드(chat.py)에서 지저분한 SQL 코드들이 모조리 사라집니다!
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        # 파이썬 객체가 생성될 때 DB에 한 번 딱 연결해 둡니다.
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.table_name = "chat_messages" # 회원님의 실제 테이블명

    @property
    def messages(self) -> list[BaseMessage]:
        """
        [조회(SELECT) 자동화 구역] 
        랭체인이 "야, 이전 기록 좀 싹 다 내놔봐" 할 때 발동되는 함수.
        """
        # 현재 우리가 쓰던 chat.py의 수동 조회 로직이 여기로 이사 온 겁니다.
        response = self.supabase.table(self.table_name).select("*").eq("history_uid", self.session_id).order("created_at").execute()
        
        # 💡 [포인트] 단순히 문자열('\n')로 합치지 않고, 랭체인 전용 객체로 예쁘게 포장해서 넘깁니다.
        langchain_messages = []
        for row in response.data:
            if row["speaker_id"] == "user_baemin": # 예시: 유저일 경우
                langchain_messages.append(HumanMessage(content=row["content"]))
            elif row["speaker_id"] == "sua_001":   # 예시: AI일 경우
                langchain_messages.append(AIMessage(content=row["content"]))
        
        return langchain_messages

    def add_message(self, message: BaseMessage) -> None:
        """
        [삽입(INSERT) 자동화 구역]
        유저가 질문을 던졌을 때 한 번, AI가 답변을 완성했을 때 또 한 번, 
        랭체인이 몰래 이 함수를 호출해서 DB에 흔적을 남깁니다.
        """
        # 지금 파이프라인에서 넘어온 메세지가 유저꺼인지, AI꺼인지 판단
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        speaker_id = "user_baemin" if role == "user" else "sua_001"
            
        data = {
            "history_uid": self.session_id,
            "speaker_id": speaker_id,
            "content": message.content
            # created_at 등은 DB에서 디폴트로 찍힘
        }
        # 개발자가 INSERT 코드를 깜빡할 확률이 0%로 줄어듭니다!
        self.supabase.table(self.table_name).insert(data).execute()

    def clear(self) -> None:
        # DB 내용을 날려버리는 기능 (보통 잘 안 씁니다)
        pass

# 매핑을 위해 감싸주는 헬퍼 함수
def get_supabase_history(session_id: str) -> BaseChatMessageHistory:
    return SupabaseChatMessageHistory(session_id)


# ============== [핵심 2] 비즈니스 로직(AI 뇌 구조) ==============
# 이제 DB 코드는 완전히 잊어버리고 오직 '프롬프트'에만 집중합니다.
# ==============================================================

def test_real_db_wrapping():
    print("\n--- [실전] Supabase 연동 자동화 맛보기 ---")
    
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=GEMINI_API_KEY)
    
    # 🎯 프롬프트 조립: DB 코드가 단 한 줄도 없습니다!
    prompt = ChatPromptTemplate.from_messages([
        ("system", "너는 따뜻하고 친구 같은 AI야. 짧고 명확하게 대답해줘."),
        MessagesPlaceholder(variable_name="chat_history"), # 빈 방: 위에서 만든 클래스의 SELECT 결과가 여기에 알아서 꽂힘
        ("human", "{question}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # 🎯 뇌 + 비서 결합!
    wrapped_chain = RunnableWithMessageHistory(
        chain,
        get_supabase_history, # 우리가 방금 심혈을 기울여 만든 그 Supabase 전용 비서!
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    
    # 🚀 이제 DB를 찌르는 대신, 세션 아이디만 던지고 쿨하게 질문합니다.
    config = {"configurable": {"session_id": "test_auto_history_001"}}
    
    print("🗨️ 질문 1 (SELECT 조회 0건 -> 프롬프트 조립 -> AI 답변생성 -> 유저/AI 2건 INSERT 숨어서 진행중...)")
    res1 = wrapped_chain.invoke({"question": "안녕! 나는 사과를 참 좋아해."}, config=config)
    print(f"🤖 AI: {res1}\n")
    
    print("🗨️ 질문 2 (SELECT 2건 조회 -> 프롬프트 넣음 -> 대답 -> 2건 INSERT 또 진행함)")
    res2 = wrapped_chain.invoke({"question": "나 무슨 과일 좋아하게?"}, config=config)
    print(f"🤖 AI: {res2}")
    
    print("\n✅ 방금 일어난 일: 아무 신경도 안 썼는데, DB에는 4개의 대화 행(Row)이 예쁘게 INSERT 되었습니다!")


if __name__ == "__main__":
    test_real_db_wrapping()
