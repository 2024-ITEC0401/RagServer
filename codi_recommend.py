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
codi_recommend_bp = Blueprint('codi_recommend', __name__)

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

def recommend_codi_to_gemini(user_codi, rag_data):
    multimodal_model = GenerativeModel(model_name="gemini-1.5-flash-002")

    prompt = load_prompt("codi_recommend_prompt.txt")

    prompt = prompt.replace("{{USER_CLOTHES}}", user_codi).replace("{{RECOMMENDED_OUTFITS}}", rag_data)

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

@codi_recommend_bp.route("/get_codis", methods=["POST"])
def get_codis():
    """
    코디 추천 API
    ---
    tags:
      - Recommendation
    summary: Generate outfit recommendations based on user clothing
    description: This endpoint takes user clothing data as input and returns recommended outfits based on the input data.
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        description: 옷장 데이터 전부. 특히 clothing_id가 중요함
        schema:
          type: object
          properties:
            clothing:
              type: array
              items:
                type: object
                properties:
                  clothing_id:
                    type: integer
                    example: 1
                  baseColor:
                    type: string
                    example: "파랑"
                  description:
                    type: string
                    example: "부드러운 촉감의 남성용 울 니트 스웨터"
                  mainCategory:
                    type: string
                    example: "상의"
                  name:
                    type: string
                    example: "남성 울 니트"
                  pattern:
                    type: string
                    example: "무지"
                  pointColor:
                    type: string
                    nullable: true
                    example: null
                  season:
                    type: string
                    example: "가을"
                  style:
                    type: string
                    example: "데일리"
                  subCategory:
                    type: string
                    example: "니트"
                  textile:
                    type: string
                    example: "니트/울"
    responses:
      200:
        description: 코디를 1~10개 알아서 llm이 생성함. 몇개인지 특정 못함 괜찮나? 원하면 바꿀 수 있음
        schema:
          type: object
          properties:
            codis:
              type: array
              items:
                type: object
                properties:
                  clothing_ids:
                    type: array
                    items:
                      type: integer
                    example: [1, 2, 8]
                  description:
                    type: string
                    example: "부드러운 울 니트와 블랙 와이드 데님의 편안한 가을 남친룩."
                  hashtags:
                    type: array
                    items:
                      type: string
                    example: ["남친룩", "가을코디", "캐주얼", "데일리룩", "편안함"]
                  name:
                    type: string
                    example: "가을 남친룩"
      400:
        description: Invalid input
    """
    # 사용자 옷
    user_closet = request.get_data(as_text=True)
    # 사용자 코디 데이터를 Vertex AI 임베딩 모델을 사용해 임베딩 벡터로 변환
    texts = [user_closet]
    user_embedding = embed_texts(texts, PROJECT_ID, LOCATION, MODEL_NAME)

    # BigQuery에서 코사인 유사도 계산 및 상위 N개 결과 가져오기
    top_results = query_similar_embeddings(PROJECT_ID, DATASET_ID, TABLE_ID, user_embedding[0], top_n=5)

    rag_data = ""
    for row in top_results:
        rag_data += row['codi_json']

    response = recommend_codi_to_gemini(user_closet, rag_data)
    print("***** response = ", response)
    try:
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500