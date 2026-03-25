# 🏗️ 챗봇 서비스 업그레이드 블루프린트

> **작성 기준:** `chat_new.py`(현재 코드), `TODO.md`(로드맵), `langchain_checklist.txt`(아키텍처 참고), `langchain_study_summary.md`(11개 스터디 기술 스택)
> **최종 목표:** 현재의 단방향 파이프라인(입력→검색→프롬프트→출력→저장)을 **비용 효율적이고, 맥락을 잃지 않으며, 확장 가능한** 프로덕션급 챗봇으로 진화시키기.

---

## 📊 현재 아키텍처 진단 (chat_new.py 기준)

### 현재 동작 흐름

```
사용자 입력
  ↓
(1) 임베딩 생성 (gemini-embedding-001, 768차원)
  ↓
(2) 하이브리드 검색 (벡터 + 키워드 후보 결합, source_type 가중치 재랭크)
  ↓
(3) 최근 대화 이력 조회 (chat_messages, limit=12)
  ↓
(4) 프롬프트 조립 (RESPONSE_POLICY + CHARACTER_PERSONA + 문서 + 이력 + 사용자 입력)
  ↓
(5) LLM 답변 생성 (LCEL: prompt | llm | StrOutputParser)
  ↓
(6) DB 저장 (사용자 메시지 + 임베딩 + AI 응답 + RAG 출처 메타데이터)
```

### 🔴 현재 구조에서 발견된 한계점 (8가지)

| #   | 한계점                                        | 영향                                                                                                               | 관련 스터디                |
| --- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| 1   | **대화 이력을 단순 문자열(`\n`)로 이어 붙임** | LLM이 누가 한 말인지 구분 못하는 경우 발생. 랭체인 정품 메시지 객체(`HumanMessage`, `AIMessage`)보다 인식률 떨어짐 | 테마 12                    |
| 2   | **RunnableParallel 미적용 (현재 ThreadPoolExecutor 사용)** | 병렬 조회는 해결됐지만 LangChain 파이프라인 일관성/가독성 측면에서 목표 아키텍처와 차이 존재                      | 테마 11 (RunnableParallel) |
| 3   | **모든 질문에 RAG 검색 수행**                 | "안녕", "잘 자" 같은 일상 대화에도 임베딩+벡터 검색을 돌림. 토큰/시간 낭비                                         | 테마 8 (Semantic Routing)  |
| 4   | **대화가 길어지면 토큰 폭발**                 | `limit=8`로 단순 자르기만 함. 과거 대화의 맥락이 증발하여 챗봇이 '치매' 증상 보임                                  | 테마 12, checklist 항목    |
| 5   | **장기 기억(취향/프로필) 시스템 부재**        | 유저가 "나 판타지 좋아해"라고 해도 다음 세션에는 까먹음                                                            | 테마 1 (Pydantic)          |
| 6   | **모든 사용자 메시지를 무차별 벡터 저장**     | "ㅋㅋ", "ㅇㅇ" 같은 무의미한 메시지도 documents_gemini에 저장되어 검색 품질 저하                                   | checklist 항목             |
| 7   | **DB 조회/저장 코드가 메인 로직과 뒤섞임**    | chat_new.py 383줄 중 절반 이상이 Supabase 쿼리. 유지보수 난이도 상승                                               | 테마 11, 12 (관심사 분리)  |
| 8   | **동기식(invoke) 단일 처리**                  | 서버화(FastAPI) 시 한 유저의 답변 대기 중 다른 유저 처리 불가                                                      | 테마 9 (Async)             |

---

## 🚀 업그레이드 후 목표 아키텍처 (전체 프로세스)

```
사용자 입력
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 0: 세션 컨텍스트 확정 (Phase B 진입 전 필수)                 │
│  - conf_uid, history_uid, user_id 확정 (JWT 파싱 or 하드코딩)     │
│  - 이후 모든 Phase는 이 값에 의존함 (멀티유저 확장의 핵심 타이밍)    │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase A: 초고속 의도 판별 (Semantic Routing)  [임베딩 모델]         │
│  - 미리 정의된 카테고리 샘플과 임베딩 유사도 비교 (0.01초)            │
│  - 판별 결과: CHITCHAT / KNOWLEDGE / DANGER                      │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
   ┌──── CHITCHAT ────────────────┐      ┌──── DANGER ────┐
   │ RAG 검색 건너뛰기              │      │ 안전 응답 리턴   │
   │ 최근 2~3턴 이력은 포함 (자연스런 │      │ (상담센터 안내)  │
   │ 대화 이어가기 위해 필요)         │      │ [Flash 모델]    │
   │ [Flash 모델]                  │      └────────────────┘
   └──────────────────────────────┘
                         ↓ (KNOWLEDGE만)
┌─────────────────────────────────────────────────────────────────┐
│ Phase B: 병렬 데이터 수집 (RunnableParallel)                      │
│  (Phase 0에서 확정된 세션 컨텍스트를 입력으로 사용)                  │
│                                                                 │
│  [왼손] 최근 대화 이력 조회 (원문 12턴) ──┐                         │
│  [오른손] 벡터 유사도 후보 (Top-12) ─────┼── 동시 실행 후 합치기    │
│  [세째손] 장기 기억 조회 (user_memories) ─┤                          │
│  [넷째손] 이전 요약본 조회 (최근 2개) ───┘                          │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase C: 프롬프트 조립 및 LLM 호출  [Pro 모델]                     │
│                                                                 │
│  프롬프트 구조 (서버 시작 시 1회 메모리 로딩):                       │
│  ┌─────────────────────────────────────────┐                   │
│  │ [system] RESPONSE_POLICY  ← 파일 로딩    │                   │
│  │ [system] CHARACTER_PERSONA ← 파일 로딩   │                   │
│  │ [system] 현재 날짜/시간 ← datetime 주입   │ ← 신규            │
│  │ [system] 장기 기억 (user_memories)       │ ← 신규            │
│  │ [system] 이전 요약본 (최근 2개)          │ ← 신규            │
│  │ [system] 검색된 문서 (RAG + created_at)  │ ← 날짜 포함       │
│  │ [history] 최근 원문 12턴 (문자열 포맷; MessagesPlaceholder 미적용) │
│  │ [human] 사용자 입력                      │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
│  LLM 호출: astream()으로 실시간 스트리밍 응답                      │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase D: 응답 반환 후 비동기 후처리 (Background Tasks)             │
│                                                                 │
│  ✅ 즉시 반환: AI 응답을 유저에게 먼저 전달                          │
│                                                                 │
│  🔄 백그라운드 (유저는 기다리지 않음):                               │
│    (1) chat_messages에 유저/AI 메시지 INSERT                      │
│    (2) 저장 가치 판단(Gating) 후 documents_gemini에 임베딩 INSERT   │
│    (3) 대화 요약 갱신 트리거 [Flash 모델] → last_processed_id 기반  │
│    (4) 장기 기억 추출 트리거 [Pro 모델] → 5턴마다                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 핵심 업그레이드 상세 설계

### 1. 대화 요약 압축 파이프라인 (`conversation_summaries`)

현재 `limit=8`로 단순 절삭하는 구조를 **"슬라이딩 윈도우 + 누적 요약"** 구조로 고도화합니다.

#### 동작 원리

```
[대화 1~10턴] ─── 10턴 도달 시 ───→ SUMMARY_MODEL로 요약 생성
                                     ↓
                              "요약본 v1" DB 저장
                              (start_message_id=1, end_message_id=10)

[대화 11~20턴] ─── 20턴 도달 시 ──→ SUMMARY_MODEL로 요약 생성
                                     ↓
                              입력: "요약본 v1" + "대화 9~10턴(이전 블록 끝 2개)" + "대화 11~20턴"
                                     ↓
                              "요약본 v2" DB 저장 (누적 요약)
                              (start_message_id=1, end_message_id=20)

[대화 21~30턴] ─── 30턴 도달 시 ──→ SUMMARY_MODEL로 요약 생성
                                     ↓
                              입력: "요약본 v2" + "대화 19~20턴(이전 블록 끝 2개)" + "대화 21~30턴"
                                     ↓
                              "요약본 v3" DB 저장 (누적 요약)
```

#### 핵심 포인트

- **이전 블록의 끝 2개 대화를 같이 가져오는 이유:** 요약 사이의 "이음새(Context Bridge)"를 만들어 문맥이 끊기지 않도록 하기 위함. 예를 들어 10번째 턴에서 "내일 만나자"라고 했고 11번째 턴에서 "어디서 만날까?"라고 했다면, 끝 2개 없이는 11번째 턴의 맥락이 증발함.
- **이전 요약본을 같이 넣는 이유:** 대화가 100턴을 넘어가도 "요약본 v10" 하나만 읽으면 1~100턴의 핵심을 3문장으로 파악 가능. 토큰 비용이 O(N)에서 O(1)로 수렴.
- **프롬프트 주입 방식(현재):** 최종 프롬프트에는 `[이전 요약본 최근 2개]` + `[최근 원문 12턴]`이 들어감.

#### DB 저장 구조 (conversation_summaries)

```sql
-- 요약 INSERT 시:
INSERT INTO conversation_summaries
  (conf_uid, history_uid, summary_text, summary_type, start_message_id, end_message_id, covered_message_count, summary_seq)
VALUES
  ('sua_test_002', 'session_001', '배민이는 판타지 소설을 좋아하며...', 'rolling', 1, 20, 20, 2);
```

#### 요약 트리거 조건 (last_processed_id 기반 — 실패 복구 안전 설계)

```python
# ✅ 카운트(% 10) 방식은 API 에러 시 그 타이밍을 영원히 놓침.
# ✅ DB에 마지막으로 요약 처리한 message_id를 기록해두고,
#    '그 이후로 10개 쌓이면' 방식으로 판단 → 에러가 나도 다음 턴에서 보정 가능.

last_summary = get_latest_summary(conf_uid, history_uid)
last_processed_id = last_summary.end_message_id if last_summary else 0

unprocessed = get_messages_after(conf_uid, history_uid, after_id=last_processed_id)

if len(unprocessed) >= 10:  # 마지막 요약 이후 10개 쌓이면
    try:
        previous_summary = last_summary.summary if last_summary else "(이전 요약 없음)"
        bridge_messages = get_messages(ids=last_summary.end_message_ids_last2)  # 이음새 2개
        new_block = unprocessed[:10]

        new_summary = SUMMARY_MODEL.invoke(
            f"이전 요약: {previous_summary}\n"
            f"이음새 대화: {bridge_messages}\n"
            f"새 대화: {new_block}\n"
            "위 내용을 불릿 포인트(-) 3~5개로 핵심만 누적 요약해줘. "
            "중요도 낮은 감정 표현/인사는 과감히 삭제할 것."
        )
        save_summary(new_summary, end_message_id=new_block[-1].id)
    except Exception as e:
        # ⚠️ 실패해도 메인 응답 흐름은 절대 막지 않음
        # 다음 턴 진입 시 unprocessed >= 10 조건이 다시 충족되어 자동 재시도됨
        log_error(f"요약 생성 실패 (보정 대기 중): {e}")
```

---

### 2. 장기 기억 추출 파이프라인 (`user_memories`)

대화 속에서 유저의 프로필, 취향, 금기 사항 등을 **Pydantic 구조화 출력으로 강제 파싱**하여 DB에 영속 저장합니다.

#### 동작 원리

```
대화 블록 (최근 5턴)
  ↓
(1) LLM에게 "이 대화에서 유저에 대해 새로 알게 된 사실이 있어?" 질문
  ↓
(2) with_structured_output(MemoryExtraction)으로 JSON 강제 파싱
  ↓
    { is_memory_worthy: true, memory_type: "preference",
      memory_key: "favorite_genre", memory_value: "fantasy", confidence: 0.85 }
  ↓
(3) DB에서 동일 user_id + memory_key 검색
  ↓
    ├── 없으면 → INSERT (새 기억 생성)
    └── 있으면 → 기억 충돌 해소 로직 진입
          ├── value가 동일하면 → confidence += 0.1, evidence_count += 1 (강화)
          ├── value가 다르면  → 신규 confidence vs 기존 confidence 비교
          │     ├── 신규 > 기존: 값을 새것으로 UPDATE (기호 변경)
          │     ├── 신규 ≒ 기존 (차이 < 0.2): 양쪽 병기 (예: "fantasy, romance")
          │     └── 신규 < 기존: 기존 값 유지, evidence_count만 +1
```

#### Pydantic 스키마

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class MemoryExtraction(BaseModel):
    is_memory_worthy: bool = Field(description="이 대화에서 유저에 대해 저장할 가치가 있는 새로운 사실이 발견되었는가")
    memory_type: Optional[Literal["profile", "preference", "taboo", "style", "relationship", "fact"]] = None
    memory_key: Optional[str] = Field(default=None, description="기억의 카테고리 키 (예: favorite_genre, user_name)")
    memory_value: Optional[str] = Field(default=None, description="기억의 정규화된 값 (예: fantasy, 배민)")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="추출 확신도")
```

#### 프롬프트 주입 방식

```
[사용자 장기 기억]
- 이름: 배민
- 좋아하는 장르: 판타지
- 싫어하는 주제: 정치
- 선호 말투: 반말 (casual)
```

---

### 3. 시맨틱 라우팅 (Semantic Routing)

모든 질문에 비싼 RAG를 돌리지 않고, **0.01초 만에 길을 가릅니다.**

#### 라우팅 샘플 파일 (`router_samples.json`)

```json
{
  "chitchat": [
    "안녕",
    "잘 자",
    "오늘 기분 어때?",
    "배고파",
    "심심해",
    "사랑해"
  ],
  "knowledge": [
    "내가 전에 뭐라고 했지?",
    "내 취미 뭐야?",
    "기억나?",
    "어제 얘기했던 거"
  ],
  "danger": ["죽고 싶어", "자해", "폭력"]
}
```

#### 판별 로직 (의사 코드)

```python
def route(user_input):
    chitchat_score = chitchat_store.similarity_search_with_score(user_input, k=1)[0][1]
    knowledge_score = knowledge_store.similarity_search_with_score(user_input, k=1)[0][1]
    danger_score = danger_store.similarity_search_with_score(user_input, k=1)[0][1]

    # ⚠️ [중요] DANGER 임계값은 의도적으로 낮게 설정
    # 이유: "요즘 살기 싫다", "없어지고 싶어" 같은 우회 표현은
    #       직접 키워드와 유사도가 뚝 떨어져 0.85를 못 넘김.
    #       안전 영역은 오탐(False Positive)이 나더라도
    #       미탐(False Negative)이 훨씬 치명적이므로 낮게 잡아야 함.
    if danger_score > 0.65:  # 0.85 → 0.65로 하향 조정
        return "DANGER"         # 안전 응답만 리턴 (상담센터 안내)
    elif chitchat_score > knowledge_score:
        return "CHITCHAT"       # RAG 검색 건너뛰기 (최근 2~3턴 이력은 포함)
    else:
        return "KNOWLEDGE"      # 전체 RAG 파이프라인 가동
```

---

### 4. 저장 가치 판단 (Document Gating)

현재는 "ㅋㅋ", "ㅇㅇ" 같은 무의미한 메시지도 `documents_gemini`에 임베딩으로 저장되어 검색 품질을 떨어뜨리고 있습니다.

#### 게이팅 규칙 (룰 베이스, LLM 호출 없음)

```python
def is_worth_storing(user_input: str) -> bool:
    # 1. 너무 짧은 메시지 거부 (3글자 이하)
    if len(user_input.strip()) <= 3:
        return False
    # 2. 의미 없는 패턴 거부
    noise_patterns = ["ㅋ", "ㅎ", "ㅇㅇ", "ㄴㄴ", "ㅜㅜ", ";;"]
    if any(user_input.strip().startswith(p) for p in noise_patterns):
        return False
    # 3. 라우팅 결과가 CHITCHAT이면 저장 안 함
    if current_route == "CHITCHAT":
        return False
    return True
```

---

### 5. 관심사 분리 (Clean Architecture)

현재 `chat_new.py`에 섞여 있는 DB 쿼리 코드를 **별도 모듈로 분리**합니다.

#### 파일 구조 (목표)

```
langchain_test/
  ├── chat.py                  # 메인 비즈니스 로직 (프롬프트, 체인, 라우팅만)
  ├── db/
  │   ├── __init__.py
  │   ├── chat_history.py      # BaseChatMessageHistory 상속, SELECT/INSERT 캡슐화
  │   ├── memory_store.py      # user_memories CRUD
  │   ├── summary_store.py     # conversation_summaries CRUD
  │   └── document_store.py    # documents_gemini 임베딩 저장/검색
  ├── services/
  │   ├── routing.py           # 시맨틱 라우터 (샘플 기반 의도 판별)
  │   ├── summarizer.py        # 대화 요약 압축 서비스
  │   └── memory_extractor.py  # 장기 기억 추출 서비스 (Pydantic)
  ├── prompts/
  │   ├── response_policy.txt  # 하드코딩 대신 파일로 분리
  │   └── persona_sua.txt      # 캐릭터 프롬프트 파일
  └── router_samples.json      # 라우팅 카테고리 샘플
```

---

### 6. 비동기화 (Async) 적용 계획

#### 즉시 적용 가능한 부분

| 현재 코드                    | 변경 후                            | 효과                          |
| ---------------------------- | ---------------------------------- | ----------------------------- |
| `chain.invoke()`             | `chain.astream()`                  | 유저에게 실시간 스트리밍 응답 |
| `insert_message()` 순차 호출 | `asyncio.create_task()` 백그라운드 | 유저는 저장 완료를 안 기다림  |
| 이력/키워드/요약 조회 순차   | `ThreadPoolExecutor` 4-way 병렬 (추후 `RunnableParallel` 전환) | 응답 시작 시간 단축 |

#### FastAPI 서버화 시 (3순위)

```python
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Phase A: 라우팅 (0.01초)
    route = await semantic_router.classify(request.message)

    # Phase B: 병렬 데이터 수집 (RunnableParallel)
    # Phase C: LLM 스트리밍 응답
    return StreamingResponse(chain.astream(...))

    # Phase D: 백그라운드 후처리 (유저는 안 기다림)
    background_tasks.add_task(save_and_extract, ...)
```

### 7. 컨텍스트 캐싱 최적화 (신규)

현재는 매 턴마다 시스템 메시지 + 요약 + 문서 + 이력을 전부 다시 조립해 모델에 전달합니다.  
아래 3단 캐시를 적용하면 비용과 지연을 동시에 줄일 수 있습니다.

#### 캐시 레이어 설계

```text
[L1: 메모리 캐시 (프로세스 내부, 초고속)]
- 키: (conf_uid, history_uid)
- 값: latest_summary_text, recent_history_text, rendered_system_block
- TTL: 30~120초
- 용도: 같은 세션의 연속 턴에서 즉시 재사용

[L2: Redis/외부 캐시 (멀티 인스턴스 공유)]
- 키: hash(conf_uid, history_uid, summary_seq, last_message_id)
- 값: prompt_context_bundle(JSON)
- TTL: 5~30분
- 용도: 서버 다중 인스턴스/재시작 환경에서 일관 재사용

[L3: DB 원본 (Supabase)]
- 항상 정합성의 기준
- 캐시 미스/만료 시에만 조회
```

#### 캐시 대상 분리 원칙

- **변화가 느린 컨텍스트(캐시 대상):**
  - 시스템 정책/페르소나 텍스트
  - 최신 누적 요약 (`summary_seq` 기준)
- **변화가 빠른 컨텍스트(매 턴 계산):**
  - 현재 사용자 입력
  - 최근 원문 2~6턴(슬라이딩 윈도우)
- **조건부 캐시:**
  - RAG 검색 결과는 `route=KNOWLEDGE`일 때만 캐시(짧은 TTL)

#### 무효화(Invalidation) 규칙

```text
1) conversation_summaries에 새 row INSERT
   -> summary_seq 변경 감지
   -> 해당 세션 summary 캐시 무효화

2) chat_messages INSERT
   -> last_message_id 변경
   -> recent_history 캐시 무효화

3) persona/policy 파일 변경 또는 버전 변경
   -> global prompt cache 전부 무효화
```

#### 구현 단계 (권장 순서)

1. L1 메모리 캐시부터 적용 (코드 최소 변경)
2. 캐시 hit/miss 로그 계측 추가
3. FastAPI 전환 시 Redis(L2) 확장
4. 요약 갱신 이벤트 기반 무효화 자동화

### 8. 하이브리드 검색/재랭크 (신규)

순수 벡터 Top-K만으로는 "책 제목", "이름", "아까 뭐였지" 같은 회상 질문에서 원문 회수율이 흔들릴 수 있어, 검색 계층을 하이브리드로 보강합니다.

#### 설계 요약

```text
입력 질의
  -> 임베딩 벡터 후보 Top-12 조회
  -> 최근 문서 풀 Top-150에서 키워드 후보 추출
  -> 현재 세션 최신 요약 2개는 RAG 후보에서 제외(중복 주입 방지)
  -> 의도 판별(회상형/일반형)
  -> 가중치 합성 재랭크
       final = w_vec * vector_score + w_key * keyword_score + source_type_boost
  -> 재랭크 상위 8개 후보
  -> 점수 임계값 필터 후 최대 4개를 프롬프트 [참고 문서]에 주입
```

#### 가중치 정책

- 회상형 질문(`제목`, `이름`, `뭐였지`, `기억` 등): `keyword` 비중 상향
- 일반 질문: `vector` 비중 상향
- 임계값: 회상형 `0.34`, 일반형 `0.44` (현재 하드코딩; 추후 A/B 대상)
- `source_type` 우선순위:
  - 회상형에서는 `chat_message` 가점, `summary`는 보조
  - 일반형에서는 `summary`/`chat_message` 혼합 허용

#### 기대 효과

- 제목/고유명사 회수율 개선
- 요약문만 맞고 원문을 놓치는 케이스 감소
- 벡터 단독 랭킹의 문체 편향 완화

### 9. 검색 후처리 정책 분리 (신규)

검색 품질 관련 로직은 아래 2가지를 분리해서 운영합니다.

#### 9-1. 재랭크 (Re-rank)

- 목적: 후보 문서의 **순서**를 더 정확하게 재배치
- 입력: 벡터 후보 + 키워드 후보
- 출력: `rank_score` 기준 상위 문서
- 적용 위치: 검색 결과 병합 직후

#### 9-2. 필터 (Filter)

- 목적: 상위 후보 중 **부적합 문서 제거**
- 예시 기준:
  - 점수 하한선 미달
  - 질문 타입과 source_type 불일치
  - 현재 세션 최신 요약 2개(summary) 제외
  - 과도한 중복 문서
- 적용 위치: 재랭크 이후, 프롬프트 주입 직전

운영 원칙: 재랭크와 필터는 저장 단계(Document Gating)와 목적이 다르므로 분리 유지.

### 10. 패턴/사전 기반 보정 (신규)

현재 코드에는 아래 패턴 로직이 실제 반영되어 있습니다.

#### 10-1. 회상 질문 패턴 (`RECALL_HINT_PATTERNS`)

- 목적: "기억/제목/뭐였지"류 질문을 회상형으로 분류
- 효과: 재랭크 시 키워드 가중치 상향
- 예시 패턴: `뭐였지`, `기억`, `제목`, `이름`, `아까`, `읽었`, `추천`

#### 10-2. 노이즈 패턴 게이팅 (`NOISE_EXACT_PATTERNS`, prefix)

- 목적: 저장 가치가 낮은 입력을 `documents_gemini`에서 제외
- 예시: `ㅋㅋ`, `ㅎㅎ`, `ㅇㅇ`, `ㄴㄴ`, `ㅠㅠ`, `..`, `;;`
- 효과: 벡터 후보풀 노이즈 감소

#### 10-3. 토큰 정규화 (`normalize_keyword_token`)

- 목적: 조사 제거로 키워드 매칭 안정화
- 예시: `제목이` -> `제목`, `소설을` -> `소설`

#### 10-4. 1글자 핵심어 예외 (`ONE_CHAR_KEYWORDS`)

- 목적: 일반 길이 필터로 누락되는 핵심어 보정
- 현재 예외: `책`

#### 10-5. 동의어 사전 (`KEYWORD_SYNONYMS`)

- 목적: 표현 차이 보정 (`책`↔`소설`↔`작품`)
- 현재 방식: 수동 사전 + 운영 로그 기반 확장 예정

---

## 📋 구현 우선순위 체크리스트

### ✅ Phase 1: 기본 개선 완료 (2026-03-24)

- [x] `format_history` 개선 — `msg_id` 제거, `created_at` 날짜 추가
- [x] `format_documents` 개선 — `created_at` 날짜 조건부 추가
- [x] 현재 날짜/시간을 시스템 프롬프트 최상단에 자동 주입 (`current_time`)
- [x] `DEBUG = True` 플래그 추가 — 서버화 시 False 한 줄로 모든 디버그 출력 오프
- [x] 이력 조회 + 문서 검색을 `ThreadPoolExecutor` 병렬 처리로 개선
- [x] 병렬 조회 4-way 확장 (벡터 후보 + 키워드 후보풀 + 최근 이력 + 최근 요약) (2026-03-25 완료)

### Phase 2: 지능 고도화 (1순위)

- [x] 대화 요약 압축 서비스 핵심 구현 (`chat_new.py` 기준)
  - [x] `last_processed_id` 기반 10턴 트리거 (2026-03-24 완료)
  - [x] 이전 요약본 + 이음새 대화 2개 + 새 블록 누적 요약 로직 (2026-03-24 완료)
  - [x] `conversation_summaries` 테이블 INSERT + 벡터화 저장 (2026-03-25 완료)
  - [x] 메인 채팅 프롬프트에 `[이전 요약]` 섹션 조회 및 주입 (2026-03-25 완료)
  - [x] 실제 스키마 정합성 반영 (`summary_text`, `covered_message_count`, `summary_seq`) (2026-03-25 완료)
  - [x] 요약 벡터 문서 메타데이터 분리 (`source_type="summary"`, `speaker_type="system"`) (2026-03-25 완료)
  - [x] 요약 출력 마크다운 흔들림 정규화(저장 전 후처리) (2026-03-25 완료)
  - [x] 요약 프롬프트에 과추론 금지 규칙 강화(명시 사실만 요약) (2026-03-25 완료)
  - [x] 프롬프트 주입용 요약 조회: 최신 1개 -> 최근 2개로 확장 (2026-03-25 완료)
- [ ] 장기 기억 추출 서비스 구현 (`memory_extractor.py`)
  - [ ] Pydantic 스키마 정의 (`MemoryExtraction`)
  - [ ] `with_structured_output`으로 JSON 강제 파싱
  - [ ] `user_memories` 테이블 UPSERT + 충돌 해소 로직
  - [ ] 프롬프트에 `[사용자 장기 기억]` 섹션 추가
- [ ] 시맨틱 라우팅 도입 (`routing.py`)
  - [ ] `router_samples.json` 작성 (카테고리별 5~10개 샘플)
  - [ ] CHITCHAT 경로: RAG 스킵, 최근 2~3턴 이력 포함하여 LLM 직발
  - [ ] KNOWLEDGE 경로: 전체 RAG 파이프라인 가동
        ~~ - [ ] DANGER 경로: 임계값 0.65, 안전 응답 리턴 ~~
- [ ] 저장 가치 판단(Document Gating) 추가
  - [x] 룰 베이스 필터 (`is_worth_storing`) 기본 적용 (2026-03-25 완료)
  - [ ] CHITCHAT 라우팅 결과 연계
- [x] 하이브리드 검색/재랭크 적용 (`chat_new.py` 기준)
  - [x] 벡터 후보 + 키워드 후보 결합 재랭크
  - [x] 12 -> 8 -> 4 검색 파이프라인 적용
  - [x] 키워드 후보풀 조회 범위 `limit=150` 적용
  - [x] 현재 세션 최신 요약 2개(summary) RAG 후보 제외 (중복 주입 방지)
  - [x] 회상형 질의 의도 기반 가중치 적용
  - [x] `source_type` 기반 점수 보정 (`chat_message` 우선)
  - [x] 디버그 출력에 `vec/key/rank/source` 점수 분해 노출
  - [ ] 검색 후처리 필터 기준 고도화 (재랭크와 분리 운영)
- [x] 패턴/사전 기반 보정 적용 (`chat_new.py` 기준)
  - [x] 회상 질문 패턴 분류 (`RECALL_HINT_PATTERNS`)
  - [x] 조사 제거 토큰 정규화 (`normalize_keyword_token`)
  - [x] 1글자 핵심어 예외(`책`) 반영
  - [x] 동의어 사전 기반 확장 (`KEYWORD_SYNONYMS`)

### Phase 3: 구조 개선 (2순위)

- [ ] DB 코드 관심사 분리 (`db/` 디렉토리) — 순수 Python 모듈 분리
  - [ ] `db/chat_history.py`: `fetch_recent_messages`, `insert_message` 이사
  - [ ] `db/document_store.py`: `find_similar_documents`, `insert_document` 이사
  - [ ] `db/memory_store.py`: `user_memories` CRUD
  - [ ] `db/summary_store.py`: `conversation_summaries` CRUD
- [ ] 비동기 전환 (부분 완료)
  - [x] `invoke` → `astream` (CLI 스트리밍 응답) (2026-03-25 완료)
  - [x] 이벤트 루프 단일화 (`asyncio.run` 1회, 턴 내부 재호출 제거) (2026-03-25 완료)
  - [x] 요약 트리거 백그라운드 스레드 분리 (유저 입력 블로킹 제거) (2026-03-25 완료)
  - [x] 요약 진행 로그를 메인 루프에서 출력하도록 조정 (입력창 간섭 완화) (2026-03-25 완료)
  - [ ] 메시지 INSERT/문서 INSERT 전체를 백그라운드 태스크로 분리
  - [ ] FastAPI 환경 기준 `async` DB I/O 경로로 통일
  - [ ] `RunnableParallel`로 조회 체인 마이그레이션 (현재 ThreadPoolExecutor)

### Phase 4: 서비스화 (3순위)

- [ ] FastAPI 엔드포인트 구축
- [ ] 멀티 유저 / 멀티 캐릭터 세션 관리
- [ ] PDF 문서 자동 파싱 및 Chunking 스크립트

### 운영/디버깅 (신규 보강)

- [x] 콘솔 출력 로그 파일 저장 (`langchain_test/logs/chat_session_YYYYMMDD.log`) (2026-03-25 완료)
- [ ] 로그 레벨 분리 (`INFO`/`DEBUG`/`ERROR`) 및 JSON 구조화
- [ ] 요청 단위 trace id 추가 (검색/프롬프트/요약 상관관계 추적)

### 캐싱/성능 (신규 보강)

- [ ] L1 인메모리 컨텍스트 캐시 도입 (`summary_seq`, `last_message_id` 키 기반)
- [ ] 캐시 hit/miss/eviction 메트릭 로그 추가
- [ ] 요약 INSERT 이후 summary 캐시 무효화 훅 추가
- [ ] FastAPI 전환 시 Redis 기반 L2 캐시 확장

### 검색 품질 (신규 보강)

- [ ] 동의어 사전(`KEYWORD_SYNONYMS`) 운영 프로세스 정착
- [ ] 회상 패턴/노이즈 패턴 사전 운영 프로세스 정착
- [ ] 회상형 질문 자동 평가셋 구축(제목/이름/숫자)
- [ ] 하이브리드 랭킹 가중치 A/B 테스트 및 고정
- [ ] 임계값(`0.34/0.44`) 상수화 및 실험 기반 재설정

### 프롬프트 구조 (정합성 보강)

- [ ] `MessagesPlaceholder` 기반 메시지 객체 이력 주입 전환
- [ ] `RESPONSE_POLICY`/`CHARACTER_PERSONA` 파일 분리(`prompts/*.txt`) 및 앱 시작 시 1회 로딩

---

## 🔑 핵심 원칙 (아키텍처 의사결정 기준)

1. **"원문이 아니라 구조화된 memory를 기준으로 시스템을 설계한다"** (langchain_checklist.txt 핵심 교훈)
2. **LLM 호출은 최소화한다:** 라우팅은 벡터 유사도로, 필터링은 룰 베이스로, LLM은 오직 답변 생성과 기억 추출에만 사용.
3. **안전 기능은 보수적으로:** DANGER 임계값처럼 안전에 관련된 판단은 오탐이 나더라도 미탐을 내지 않는 방향으로 설계.
4. **코드 관심사 분리:** AI 로직(프롬프트)과 DB 로직(쿼리)은 반드시 다른 파일에 존재해야 한다.
5. **안전장치:** 요약/기억 추출 실패 시에도 메인 응답 흐름은 절대 막히지 않도록 try-except 처리. 실패는 로그만 남기고 다음 턴에 자동 재시도.
6. **정적 리소스(프롬프트 파일)는 서버 시작 시 1회만 로딩:** `response_policy.txt`, `persona_sua.txt`는 매 턴 파일 I/O를 피하기 위해 앱 초기화 시 메모리에 올려두고 재사용.

---

## 🤖 모델 분리 기준 (Phase별 명세)

| Phase         | 작업                 | 사용 모델        | 이유                                                     |
| ------------- | -------------------- | ---------------- | -------------------------------------------------------- |
| A (라우팅)    | 임베딩 유사도 계산   | 임베딩 모델      | LLM 호출 없음, 최저 비용                                 |
| C (답변 생성) | 메인 챗봇 응답       | **Flash**        | 빠른 응답이 최우선, 일상 대화 품질로 충분                |
| D-요약        | 대화 압축 요약       | **Flash-lite**   | 단순 압축 작업, Pro급 추론 불필요                        |
| D-기억 추출   | Pydantic 구조화 파싱 | **Pro or Flash** | 뉘앙스를 읽고 정확한 Key-Value로 변환하는 정밀 추론 필요 |
| D-저장 판단   | Document Gating      | 룰 베이스        | LLM 호출 자체가 없음                                     |

---

## 🕒 성능 및 레이턴시 보정 전략 (Sync vs Async)

현재는 답변 생성은 `astream` 스트리밍이며, 요약 트리거는 백그라운드 스레드로 분리되어 있습니다.

- **개선 반영:** 요약 트리거는 백그라운드로 넘겨 유저 입력 블로킹을 제거함.
- **개선 반영:** 이벤트 루프를 단일화하여 `Event loop is closed` 재발 가능성을 낮춤.
- **개선 반영:** 요약 진행 상태(`미처리 메시지`)는 메인 루프에서 출력해 프롬프트 간섭을 줄임.
- **잔여 과제:** `insert_message`/`insert_document` 등 저장 경로는 아직 동기 호출이므로, 서버화 단계에서 `BackgroundTasks` + async DB로 전환 필요.
- **추가 과제:** 컨텍스트 캐시(L1/L2) 도입으로 프롬프트 조립 및 DB 재조회 비용을 줄여 P95 지연을 추가로 낮출 것.
- **향후 계획:** FastAPI 전환 시 `StreamingResponse(chain.astream(...))` + 후처리 태스크 분리를 표준 경로로 고정.
