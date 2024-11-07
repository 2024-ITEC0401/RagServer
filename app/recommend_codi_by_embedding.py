from traceback import print_tb

from google.cloud import bigquery
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List

from embed_data import embed_texts


# 1. BigQuery에서 코디 임베딩 데이터 불러오기
def load_embeddings_from_bigquery(project_id: str, dataset_id: str, table_id: str) -> List[dict]:
    client = bigquery.Client(project=project_id)
    query = f"SELECT codi_json, embedding FROM `{project_id}.{dataset_id}.{table_id}` LIMIT 1000"
    query_job = client.query(query)
    results = query_job.result()

    embeddings_data = []
    for row in results:
        embeddings_data.append({"codi_json": row.codi_json, "embedding": row.embedding})

    return embeddings_data


# 2. 코사인 유사도 계산 함수
def calculate_cosine_similarity(user_embedding: List[float], embeddings_data: List[dict]) -> List[dict]:
    user_embedding = np.array(user_embedding).reshape(1, -1)  # 사용자 임베딩을 2D 배열로 변환
    similarities = []
    for data in embeddings_data:
        # 코사인 유사도 계산
        coord_embedding = np.array(data["embedding"]).reshape(1, -1)
        similarity = cosine_similarity(user_embedding, coord_embedding)[0][0]
        similarities.append({"codi_json": data["codi_json"], "similarity": similarity})

    return similarities


# 3. 유사도가 가장 높은 코디 선별 함수
def get_top_similar_coord(similarities: List[dict], top_n: int = 5) -> List[dict]:
    # 유사도 순으로 정렬하여 상위 N개의 코디 선택
    sorted_similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
    return sorted_similarities[:top_n]


# Main 함수 - 유사도 계산 및 코디 선별 실행
def main():
    project_id = "gen-lang-client-0935527998"
    dataset_id = "vector_search"
    table_id = "vector_test_table"
    user_codi = ''''{
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



    # 1. BigQuery에서 코디 임베딩 데이터 불러오기
    embeddings_data = load_embeddings_from_bigquery(project_id, dataset_id, table_id)

    # 예시 사용자 임베딩 (여기에 실제 사용자의 옷 데이터를 기반으로 생성된 벡터가 들어감)
    texts = [user_codi]

    user_embedding = embed_texts(texts, project_id, "us-central1")

    # 2. 코사인 유사도 계산
    similarities = calculate_cosine_similarity(user_embedding, embeddings_data)

    # 3. 유사도가 가장 높은 상위 5개의 코디 선별
    top_similar_coords = get_top_similar_coord(similarities, top_n=5)

    # 결과 출력
    print("Top 5 similar outfits based on user embedding:")
    for i, coord in enumerate(top_similar_coords):
        print(f"{i + 1}. Outfit: {coord['codi_json']}, Similarity: {coord['similarity']:.4f}")



if __name__ == "__main__":
    main()
