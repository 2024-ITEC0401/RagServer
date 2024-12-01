from google.cloud import bigquery, aiplatform
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from typing import List, Generator


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
      (SELECT SUM(v1 * v2) FROM UNNEST(vec1) AS v1 WITH OFFSET i
      JOIN UNNEST(vec2) AS v2 WITH OFFSET j ON i = j) /
      (SQRT(SUM(POW(v, 2)) FROM UNNEST(vec1) AS v) *
       SQRT(SUM(POW(v, 2)) FROM UNNEST(vec2) AS v))
    );

    WITH user_embedding AS (
      SELECT [{user_embedding_str}] AS embedding
    ),
    similarities AS (
      SELECT
        codi_json,
        cosine_similarity(user_embedding.embedding, embedding) AS similarity
      FROM `{project_id}.{dataset_id}.{table_id}`
    )
    SELECT codi_json, similarity
    FROM similarities
    ORDER BY similarity DESC
    LIMIT {top_n};
    """

    query_job = client.query(query)
    return query_job.result()


# 3. Main 함수
def main():
    # BigQuery 및 Vertex AI 설정
    project_id = "gen-lang-client-0935527998"
    dataset_id = "vector_search"
    table_id = "vector_test_table"
    location = "us-central1"
    model_name = "textembedding-gecko@003"

    # 사용자 코디 데이터
    user_codi = '''{
      "name": "캠퍼스 여름 캐주얼 코디",
      "clothes": [
        {
          "category": "아우터",
          "subCategory": "조끼",
          "baseColor": "검정",
          "pointColor": "검정",
          "season": "여름",
          "styles": "데일리",
          "textile": "나일론",
          "pattern": "무지"
        },
        {
          "category": "바지",
          "subCategory": "반바지",
          "baseColor": "연회색",
          "pointColor": "연회색",
          "season": "여름",
          "styles": "데일리",
          "textile": "데님",
          "pattern": "워싱"
        },
        {
          "category": "신발",
          "subCategory": "슬립온",
          "baseColor": "검정",
          "pointColor": "흰색",
          "season": "여름",
          "styles": "데일리",
          "textile": "가죽",
          "pattern": "무지"
        },
        {
          "category": "가방",
          "subCategory": "백팩",
          "baseColor": "검정",
          "pointColor": "검정",
          "season": "여름",
          "styles": "데일리",
          "textile": "나일론",
          "pattern": "무지"
        },
        {
          "category": "악세서리",
          "subCategory": "우산",
          "baseColor": "검정",
          "pointColor": "나무색",
          "season": "여름",
          "styles": "데일리",
          "textile": "나일론",
          "pattern": "무지"
        }
      ],
      "hashtags": [
        "#미니멀",
        "#비",
        "#캠퍼스",
        "#심볼",
        "#워싱",
        "#여름",
        "#캐주얼",
        "#비코디",
        "#코디맵"
      ]
    }'''

    # 사용자 코디 데이터를 Vertex AI 임베딩 모델을 사용해 임베딩 벡터로 변환
    texts = [user_codi]
    user_embedding = embed_texts(texts, project_id, location, model_name)

    # BigQuery에서 코사인 유사도 계산 및 상위 N개 결과 가져오기
    top_results = query_similar_embeddings(project_id, dataset_id, table_id, user_embedding[0], top_n=5)

    # 결과 출력
    print("Top similar outfits from BigQuery:")
    for row in top_results:
        print(f"Outfit: {row['codi_json']}, Similarity: {row['similarity']:.4f}")


if __name__ == "__main__":
    main()
