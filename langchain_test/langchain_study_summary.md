# 🚀 LangChain & RAG 챗봇 구축 스터디 총정리

이 문서는 기초적인 DB 연결부터 최고급 에이전트 구축 및 스트리밍 제어, 시맨틱 검색 고도화까지 시간 순서대로 진행된 7개의 파이썬 스크립트에 대한 **역할과 기술적 교훈**을 정리한 마크다운 문서입니다.

---

## 1. `test.py` : 초기 통신 및 프롬프트 조립 뼈대
* **역할:** Supabase 데이터베이스와 처음으로 연결을 시도하고, 날것의 데이터를 가져와 랭체인의 프롬프트 템플릿에 집어넣는 "Hello World" 같은 테스트 파일입니다.
* **배운 기술/기능:**
  * `create_client`를 이용한 Supabase 기본 통신 (`select`, `eq`, `limit` 등)
  * `format_documents` 등의 파이썬 함수로 DB 배열을 순수한 문자열(Text)로 포맷팅하는 로직
  * `ChatPromptTemplate`을 통해 System / Human 메세지 구조를 짜고, 변수(`{documents}`, `{history}`)를 `invoke()`로 깔끔하게 맵핑해보기.

## 2. `input_version.py` : 데이터 삽입과 수동 RAG 검색기
* **역할:** 터미널에서 채팅을 치면, 그걸 DB에 저장하고 동시에 임베딩 벡터로 변환해 저장하는 **'입력 전용 파이프라인'**입니다. 
* **배운 기술/기능:**
  * Google GenAI API를 순수하게 호출하여 문자열을 768차원의 숫자로 변환 (`models.embed_content`)
  * `insert_chat_message`와 `insert_document`를 분리하여 원본 로그와 RAG용 검색 데이터를 따로 관리하는 아키텍처 설계
  * Supabase 내부 RPC 함수(`match_documents_gemini`)를 쌩 코드로 호출하여, **순수한 유사도 수치(Similarity Score)** 를 눈으로 직접 확인하기.

## 3. `chat.py` : 수동 기어(생코딩) 챗봇의 최종 완성본
* **역할:** RAG(결과 검색) + 히스토리 + LLM 답변 생성을 하나의 `while True` 루프에 묶어낸 **실시간 캐릭터 챗봇의 완성형**입니다. (현재 프로젝트의 주력 아키텍처)
* **배운 기술/기능:**
  * **환각 방지(Threshold) 필터링:** 유사도 점수가 `0.75` 이하인 쓰레기 데이터는 LLM에 들어가지 못하도록 파이썬 `if`문으로 쳐내기.
  * **페르소나 주입:** "너는 수아다" 라는 시스템 프롬프트를 맨 꼭대기에 박아두고 캐릭터성 유지시키기.
  * 생코딩의 자유로움 덕분에 랭체인의 버그나 의존성 충돌로부터 100% 안전한, 완벽한 통제권(제어권)을 갖는 방법 터득.

## 4. `supabase_vector_store.py` : 랭체인 생태계와 블랙박스의 명암
* **역할:** 대용량 문서(PDF 등)를 밀어 넣기 위해 랭체인이 공식 지원하는 `SupabaseVectorStore` 모듈을 써보고, 그 장단점을 뼈저리게 체험한 스크립트입니다.
* **배운 기술/기능:**
  * `.from_documents` 와 `.as_retriever` 를 통한 초간편 검색(LCEL) 기능. Multi-Query나 MMR 같은 고급 검색 알고리즘으로 쉽게 넘어갈 수 있는 확장성.
  * **블랙박스의 치명적 단점 체험:** 
    * "지 마음대로 임베딩 차원을 3072로 뻥튀기함" (`output_dimensionality` 버그)
    * "PK 컬럼이 숫자인 테이블에 강제로 문자열 UUID를 밀어 넣어서 서버를 다운시킴" (`invalid input syntax` 에러)
    * "Supabase 내부 코드를 마음대로 해킹해서 쓰다가 코드가 부서짐" (`params` 에러)
  * **결론:** 대용량 외부 문서를 자를 때(TextSplitter)만 랭체인을 쓰고, 수동 DB 제어는 `chat.py`로 간다는 아키텍처 결정의 계기가 됨.

## 5. `langchain_study_advanced.py` : Agentic AI와 구조화된 파싱 (끝판왕)
* **역할:** 단순한 RAG 질문봇을 넘어, 스스로 함수를 호출하는 '에이전트'와 무조건 JSON만 뱉는 '파서' 기술을 배운 스크립트입니다.
* **배운 기술/기능:**
  * **`with_structured_output` (구조화된 출력 강제):** 수동 프롬프트의 한계(마크다운 섞임 등)를 부수고, `memory_schema.txt`의 까다로운 제약(`Literal` 범주, 길이 제한 등)을 구글 서버 엔진 단에서 물리적으로 강제하여 100% 깔끔한 JSON만 받아내는 엔터프라이즈급 파싱 기술.
  * **`bind_tools` (도구 호출 에이전트):** 파이썬 함수(`calculate_salary`)를 도구로 던져주어, LLM이 수학이나 외부 API를 스스로 꺼내 쓰도록 진화시키는 기술.
  * **에이전트의 한계와 꿀팁:** LLM은 뇌(메모리)가 없어서 매번 엄청난 길이의 도구 설명서(docstring)를 API에 꽉 채워서 통신하므로 비용/토큰이 낭비된다는 점. 그리고 너무 몰입하면 냅다 계산기부터 들이미는 병(오지랖)에 걸리므로, 수동 프롬프트 통제가 필수불가결하다는 교훈.

## 6. `langchain_study_lcel.py` : LCEL 문법과 실시간 스트리밍(Streaming) 제어
* **역할:** 랭체인 프레임워크의 꽃인 "파이프 연산자(|)"를 통한 체인 조립 방식과, 사용자 UX(속도 체감)를 극대화하는 실시간 출력 기법을 학습한 스크립트입니다.
* **배운 기술/기능:**
  * **LCEL 파이프 체인 (`prompt | llm | parser`):** 파이썬 객체들을 리눅스 명령어처럼 한 줄의 `|` 로 조립하여, 데이터가 투입구부터 배출구까지 폭포수처럼 흘러가게 만드는 랭체인의 선언형 프로그래밍 문법.
  * **스트리밍을 통한 병목 현상(Latency) 은폐:** 대부분의 시간 소모(1~3초)는 파이썬 코드가 아니라 구글 LLM 엔진의 "첫 단어 생성 시간(Time To First Token)"과 "생성/통신"에서 발생함. `.invoke()` 대신 `.stream()`을 써서, LLM이 단어 하나를 계산할 때마다 조각(chunk) 단위로 즉시 화면에 뿌려 대기 시간을 없애버리는 UX 기술을 습득.
  * **`flush=True` 의 OS 레벨 버퍼링 통제:** 스트리밍 시 터미널 화면에 줄바꿈 없이(`end=""`) 실시간으로 글자를 뱉어내기 위해, 파이썬 기본 출력 장치의 장바구니 쌓기(Buffering)를 무시하고 즉시 출력 하드웨어로 강제 발사(`flush=True`)시키는 시스템 최적화 구문. (단, 일반 로그 출력 시 무분별하게 남발하면 심각한 성능 저하 초래)

## 7. `langchain_study_retriever.py` : 고급 검색 기술 (문맥 보존 & AI 자동 필터링)
* **역할:** 단순한 텍스트 쪼개기(Chunk) 검색이 가지는 '앞뒤 문맥 단절 문제'와 '의미 기반 엉뚱한 로그 검색 문제'를 소프트웨어적으로 해결하는 랭체인의 고급 검색기(Retriever) 스터디.
* **배운 기술/기능:**
  * **Parent Document Retriever (검색은 조각으로, 대답은 통째로):** 문서 원본(Parent)을 2개의 창고로 쪼개어 관리하는 기술. `documents_gemini`처럼 잘게 쪼갠 조각(Child)으로 단어 명중률을 극한으로 올리고, 실제 LLM에게는 그 조각이 소속되어 있던 원래의 긴 문단(Parent)을 던져주어 AI가 앞뒤 맥락(Context)을 완벽하게 파악하게 하는 아키텍처. (이때 `chunk_size`는 반드시 `chunk_overlap`보다 커야 한다는 텍스트 분할기의 룰을 배움.)
  * **Self-Querying Retriever:** 사용자의 자연어 질문("어제 배민이가 한 말 찾아줘")을 LLM이 분석하여, DB 쿼리의 `WHERE` 절 조건(`created_at = '어제' AND speaker_id = 'baemin'`)으로 스스로 깎아낸 뒤 메타데이터 필터링을 조작하게 만드는 자율 검색 기술.
  * **💡 [테마 6] 예제가 에러 난 이유 (기술적 교훈):** 파이썬 램(RAM) 위에 띄워둔 임시 보관소(`InMemoryVectorStore`)는 단순히 비슷한 글자만 매칭할 줄 아는 '멍텅구리 창고'임. Self-Querying은 AI가 만들어주는 고도의 메타데이터 필터링(SQL WHERE 조건문 같은 것)을 척척 받아먹을 수 있는 **지능형 Vecto DB(Supabase, Pinecone 등)** 엔진과 결합되어야만 동작할 수 있는 고급 기술이라는 플랫폼의 한계를 깨달음.

---

### 🎯 총평 (Next Step)
이 7개의 파이프라인 파일을 거치며 **"언제 랭체인의 꿀(?)만 빨아먹고, 언제 손수 짠 생코딩 뼈대(`chat.py`)를 고수해야 하는가"** 에 대한 완벽한 아키텍처 기준이 세워졌습니다. 이제 이 지식들을 활용해 `memory_schema.py(장기 기억 추출)` 와 `conversation_summaries.py(요약 블록)` 를 구축하는 것이 다음 목표입니다!

---

## 📚 주요 사용 파이썬 모듈 및 함수 사전

### 1. Supabase (데이터베이스 통신)
* `create_client(url, key, options)`: Supabase DB 접속용 클라이언트 생성 객체
* `supabase.schema("스키마명").table("테이블명")`: 타겟 DB 스키마와 테이블을 명시적으로 수동 지정
* `.insert(payload)`, `.select(...)`, `.eq(...)`, `.order(...)`: 데이터 조회 및 삽입용 기본 문법
* `.rpc("RPC함수명", {"파라미터": 값})`: Supabase 내부의 저장 프로시저(벡터 코사인 검색 함수 등)를 호출
* `.execute()`: 세팅된 쿼리를 최종적으로 DB 서버로 전송

### 2. Google GenAI 공식 SDK (수동 모델 제어)
* `google.genai.Client(...)`: 구글이 제공하는 오리지널 SDK 클라이언트 선언
* `genai_client.models.embed_content(...)`: 텍스트를 순수하게 고정된 길이(예: `output_dimensionality=768`)의 벡터(숫자 배열)로 변환
* `genai_client.models.generate_content_stream(...)`: 실시간 타이핑 효과(스트리밍)를 위한 LLM 텍스트 생성

### 3. LangChain 
* `ChatGoogleGenerativeAI`: 랭체인의 파이프라인(명령어 규격)에 완벽하게 호환되도록 감싼 제미나이 챗봇 객체
* `GoogleGenerativeAIEmbeddings`: 랭체인의 문서 분할 및 벡터 DB 인서트에 호환되도록 감싼 임베딩 객체
* `ChatPromptTemplate.from_messages([...])`: "system", "human" 등 역할별로 대화를 예쁘게 포장하는 프롬프트 템플릿
* `SupabaseVectorStore.from_documents(...)`: 대량의 문서(Document) 리스트를 쪼개어 임베딩하고 DB에 뭉텅이로 Bulk Insert 하는 블랙박스 메소드
* `vectorstore.similarity_search(query, k)`: 인서트 대신 검색 기능만 쓸 때 던지는 랭체인의 자체 검색 체인
* `@tool`: 그냥 평범한 파이썬 `def` 함수 머리 위에 씌우면, 랭체인 LLM이 메뉴판처럼 인식할 수 있는 `BaseTool` 객체로 강제 진화시키는 데코레이터
* `.bind_tools(tools)`: LLM 인스턴스에 사용 가능한 `@tool` 함수들의 파라미터와 설명을 묶어주는 기능 (스스로 도구 선택 가능)
* `.with_structured_output(PydanticModel)`: LLM이 딴소리를 하지 못하게 Pydantic 구조(JSON)로만 결과물을 타공해서 뱉도록 서버 엔진 단에서 족쇄를 거는 마법의 함수
* `chain = prompt | llm | parser` (LCEL): 각 부품 객체를 파이프 기호(`|`)로 결합해, 입력을 던지면 최종 출력까지 자동으로 흐르는 컨베이어 벨트를 구축하는 문법
* `.stream({"변수": "값"})`: 체인을 통해 결과물 조각(chunk)을 실시간(generator 형태)으로 쪼개서 반환받아, 답변 생성 지연시간(Latency)을 마스킹하는 비동기급 함수
* `StrOutputParser()`: LLM이 반환하는 두꺼운 AI객체(메타데이터 포함) 덩어리에서, 사람이 읽을 수 있는 순수한 텍스트 원문(String)만 쏙 뽑아주는 LCEL 전용 깔때기 부품
* `ParentDocumentRetriever`: 문서 검색 시 정교함(작은 단어)을 얻기 위해 Child 조각으로 찾고, 대답의 퀄리티(맥락)를 얻기 위해 원문인 Parent를 낚아채 오는 고급 검색 엔진
* `SelfQueryRetriever`: 사용자의 자연어 질문을 역분석하여 DB 필터링용 쿼리 조건(Metadata WHERE)으로 자동 변환해 주는 똑똑한 검색 엔진 (단, 이 필터를 처리할 수 있는 실제 Vector DB 엔진이 필요함)

### 4. Pydantic (데이터 검증 모듈)
* `BaseModel`: 구조화된 데이터 껍데기(클래스)를 선언할 때 반드시 상속받아야 하는 기둥
* `Field(description="...")`: 변수의 역할을 친절하게 설명하여 LLM에게 "여긴 이런 거 넣어" 라고 가이드해주는 기능 (최소값, 최대값 방어 등 혼용 가능)
* `Literal["단어1", "단어2"]`: 파이썬의 타입 힌트로, "조건에 명세한 영단어 외에는 절대 통과시키지 마라"고 엄격하게 제한하는 기능
