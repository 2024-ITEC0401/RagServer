from google.cloud import bigquery, aiplatform
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from typing import List
from config import load_prompt
from vertexai.generative_models import GenerativeModel
import re
import json

# 1. Vertex AI 임베딩 모델 사용
def embed_texts(texts: List[str], project_id: str, location: str, model_name: str = "textembedding-gecko@003") -> List[
    List[float]]:
    """
    Vertex AI Text Embedding 모델을 사용하여 텍스트 데이터를 임베딩 벡터로 변환
    """
    aiplatform.init(project=project_id, location=location)
    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text) for text in texts]
    embeddings = model.get_embeddings(inputs)
    return [embedding.values for embedding in embeddings]


# 2. BigQuery에서 코사인 유사도 계산 쿼리 실행
def query_similar_embeddings(project_id: str, dataset_id: str, table_id: str, user_embedding: List[float],
                             top_n: int = 5):
    """
    BigQuery에서 사용자 임베딩과 데이터베이스의 임베딩 간 코사인 유사도를 계산하여 상위 N개 결과 반환
    """
    client = bigquery.Client(project=project_id)

    # 사용자 임베딩을 문자열 형태로 변환
    user_embedding_str = ", ".join(map(str, user_embedding))

    query = f"""
    CREATE TEMP FUNCTION cosine_similarity(vec1 ARRAY<FLOAT64>, vec2 ARRAY<FLOAT64>) AS (
      (
        SELECT SUM(v1 * v2)
        FROM UNNEST(vec1) AS v1 WITH OFFSET i
        JOIN UNNEST(vec2) AS v2 WITH OFFSET j ON i = j
      ) /
      (
        SQRT(
          (SELECT SUM(POW(v, 2)) FROM UNNEST(vec1) AS v)
        ) *
        SQRT(
          (SELECT SUM(POW(v, 2)) FROM UNNEST(vec2) AS v)
        )
      )
    );

    WITH user_embedding AS (
      SELECT ARRAY[{user_embedding_str}] AS embedding
    )
    SELECT
      codi_json,
      cosine_similarity(user_embedding.embedding, table_embedding.embedding) AS similarity
    FROM `{project_id}.{dataset_id}.{table_id}` AS table_embedding
    CROSS JOIN user_embedding
    ORDER BY similarity DESC
    LIMIT {top_n};
    """

    query_job = client.query(query)
    return query_job.result()

def recommend_codi_to_gemini(user_codi, rag_data):
    multimodal_model = GenerativeModel(model_name="gemini-1.5-flash-002")

    prompt = load_prompt("../prompt/codi_recommend_prompt.txt")

    prompt = prompt.replace("{{USER_CLOTHES}}", user_codi).replace("{{RECOMMENDED_OUTFITS}}", rag_data)

    # 이미지 URI와 프롬프트 전송
    response = multimodal_model.generate_content(
        [
            prompt
        ],
        generation_config={
            "temperature": 1,  # temperature 설정
        }
    )

    # 불필요한 ```json 구문 제거 및 JSON 파싱
    codis = response.text if response else "No response text found"
    json_match = re.search(r"\{.*\}", codis, re.DOTALL)  # 중괄호로 시작하는 JSON 부분 추출

    if json_match:
        json_str = json_match.group(0)  # JSON 부분만 추출
    else:
        json_str = "{}"  # JSON 부분이 없을 때 빈 객체 반환

    try:
        codis_json = json.loads(json_str) if codis else {}
    except json.JSONDecodeError as e:
        codis_json = {"error": "Invalid JSON format received"}

    return codis_json

# 3. Main 함수
def main():
    # BigQuery 및 Vertex AI 설정
    project_id = "gen-lang-client-0935527998"
    dataset_id = "vector_search"
    table_id = "vector_test_table"
    location = "us-central1"
    model_name = "textembedding-gecko@003"

    # 사용자 옷
    user_codi = load_prompt("../codi_recommend_response.json")

    # 사용자 코디 데이터를 Vertex AI 임베딩 모델을 사용해 임베딩 벡터로 변환
    texts = [user_codi]
    user_embedding = embed_texts(texts, project_id, location, model_name)

    # BigQuery에서 코사인 유사도 계산 및 상위 N개 결과 가져오기
    top_results = query_similar_embeddings(project_id, dataset_id, table_id, user_embedding[0], top_n=5)
    rag_data = ""
    # 결과 출력
    for row in top_results:
       rag_data += row['codi_json']

    print(recommend_codi_to_gemini(user_codi, rag_data))

if __name__ == "__main__":
    main()
