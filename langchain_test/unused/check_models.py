"""
사용 가능한 임베딩 모델 목록 확인 스크립트
실행: py -3.12 check_models.py
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from google import genai as google_genai
from config import GEMINI_API_KEY, GEMINI_API_KEY_FREE

def check_embedding_models(label: str, api_key: str):
    print(f"\n{'='*55}")
    print(f"[{label}] 키: {api_key[:12]}...")
    print(f"{'='*55}")

    for api_version in ["v1", "v1beta"]:
        print(f"\n--- API Version: {api_version} ---")
        try:
            client = google_genai.Client(
                api_key=api_key,
                http_options={"api_version": api_version}
            )
            models = client.models.list()
            embed_models = [
                m for m in models
                if "embed" in m.name.lower()
            ]
            if embed_models:
                for m in embed_models:
                    print(f"  ✅ {m.name}")
            else:
                print("  (임베딩 모델 없음)")
        except Exception as e:
            print(f"  ❌ 오류: {e}")

    # 실제 임베딩 호출 테스트
    print(f"\n--- 실제 embedContent 호출 테스트 ---")
    for api_version, model in [("v1beta", "models/embedding-001"), ("v1", "models/text-embedding-004")]:
        try:
            client = google_genai.Client(
                api_key=api_key,
                http_options={"api_version": api_version}
            )
            result = client.models.embed_content(
                model=model,
                contents="테스트"
            )
            dim = len(result.embeddings[0].values)
            print(f"  ✅ {api_version} / {model} → {dim}차원 성공!")
        except Exception as e:
            print(f"  ❌ {api_version} / {model} → {str(e)[:80]}")


if __name__ == "__main__":
    check_embedding_models("무료 키 (FREE)", GEMINI_API_KEY_FREE)
    check_embedding_models("유료 키 (1티어)", GEMINI_API_KEY)
