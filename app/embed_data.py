from google.cloud import bigquery, aiplatform
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from typing import List


# 1. BigQuery에서 데이터 불러오기
def load_data_from_bigquery(project_id: str, dataset_id: str, table_id: str) -> List[str]:
    client = bigquery.Client(project=project_id)
    query = f"SELECT ml_generate_text_llm_result FROM `{project_id}.{dataset_id}.{table_id}` LIMIT 1000"
    query_job = client.query(query)
    results = query_job.result()
    # 텍스트 데이터를 리스트로 추출
    texts = [row.ml_generate_text_llm_result for row in results if row.ml_generate_text_llm_result is not None]

    return texts


# 2. Vertex AI 임베딩 모델 사용 (textembedding-gecko@003)
def embed_texts(texts: List[str], project_id: str, location: str, model_name: str = "textembedding-gecko@003") -> List[
    List[float]]:
    aiplatform.init(project=project_id, location=location)
    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text) for text in texts]
    embeddings = model.get_embeddings(inputs)
    # 각 텍스트에 대한 임베딩 벡터 추출
    return [embedding.values for embedding in embeddings]


# 3. BigQuery에 임베딩 결과 저장
def save_embeddings_to_bigquery(project_id: str, dataset_id: str, table_id: str, texts: List[str],
                                embeddings: List[List[float]]):
    client = bigquery.Client(project=project_id)
    # 임베딩 테이블의 스키마 정의
    schema = [
        bigquery.SchemaField("text", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED")
    ]
    table_ref = bigquery.Table(f"{project_id}.{dataset_id}.{table_id}", schema=schema)
    try:
        client.get_table(table_ref)
    except Exception:
        client.create_table(table_ref)
    rows_to_insert = [
        {"codi_json": text, "embedding": embedding} for text, embedding in zip(texts, embeddings)
    ]
    # 새로운 테이블에 데이터 삽입
    errors = client.insert_rows_json(table_ref, rows_to_insert, row_ids=[None] * len(rows_to_insert))
    if errors:
        print(f"Errors occurred while inserting rows: {errors}")
    else:
        print(f"Data successfully inserted into {table_ref}")


# Main 함수 - 전체 작업 흐름 실행
def main():
    project_id = "gen-lang-client-0935527998"
    dataset_id = "bqml_codi_map"
    vector_dataset_id = "vector_search"
    source_table_id = "codi_map_summary_test"
    destination_table_id = "vector_test_table"
    location = "us-central1"
    # 1. BigQuery에서 데이터 불러오기
    texts = load_data_from_bigquery(project_id, dataset_id, source_table_id)
    # 2. Vertex AI에서 텍스트 임베딩 생성
    embeddings = embed_texts(texts, project_id, location)
    # 3. BigQuery에 임베딩 결과 저장
    save_embeddings_to_bigquery(project_id, vector_dataset_id, destination_table_id, texts, embeddings)


if __name__ == "__main__":
    main()