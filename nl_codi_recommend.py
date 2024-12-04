import json
import re
import os
from typing import List
from dotenv import load_dotenv
from flask import Blueprint, request, Response, jsonify
from vertexai.generative_models import GenerativeModel
from config import load_prompt
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.cloud import bigquery, aiplatform

# .env 파일 로드
load_dotenv()

# Blueprint 생성
nl_codi_recommend_bp = Blueprint('nl_codi_recommend', __name__)

# GCP 설정
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = "us-central1"
MODEL_NAME = "textembedding-gecko@003"
DATASET_ID = "vector_search"
TABLE_ID = "vector_test_table"

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

def recommend_codi_to_gemini(user_codi, rag_data, natural_language):
    multimodal_model = GenerativeModel(model_name="gemini-1.5-flash-002")

    prompt = load_prompt("prompt/nl_codi_recommend_prompt.txt")

    prompt = prompt.replace("{{USER_CLOTHES}}", user_codi).replace("{{RECOMMENDED_OUTFITS}}", rag_data).replace("{{USER_REQUIREMENT}}", natural_language)
    print("***** prompt = ", prompt)
    # 이미지 URI와 프롬프트 전송
    response = multimodal_model.generate_content(
        [
            prompt
        ],
        generation_config={
            "temperature": 0.8,  # temperature 설정
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

@nl_codi_recommend_bp.route("/get_nl_codi", methods=["POST"])
def get_codi():

    # 사용자 옷
    nl_codi_request = request.get_json()

    # 사용자 코디 데이터를 Vertex AI 임베딩 모델을 사용해 임베딩 벡터로 변환
    natural_language = nl_codi_request.get('natural_language')
    clothing = request.get_data(as_text=True)

    clothing_list = [clothing]
    natural_language_list = [natural_language]
    clothing_embedding = embed_texts(clothing_list, PROJECT_ID, LOCATION, MODEL_NAME)
    natural_language_embedding = embed_texts(natural_language_list, PROJECT_ID, LOCATION, MODEL_NAME)

    # BigQuery에서 코사인 유사도 계산 및 상위 N개 결과 가져오기
    a_result = query_similar_embeddings(PROJECT_ID, DATASET_ID, TABLE_ID, clothing_embedding[0], top_n=3)
    b_result = query_similar_embeddings(PROJECT_ID, DATASET_ID, TABLE_ID, natural_language_embedding[0], top_n=3)

    rag_data = ""
    for row in a_result:
        rag_data += row['codi_json']
    for row in b_result:
        rag_data += row['codi_json']

    response = recommend_codi_to_gemini(clothing, rag_data, natural_language)
    try:
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500