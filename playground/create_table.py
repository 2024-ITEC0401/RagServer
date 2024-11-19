from google.cloud import bigquery

project_id = "gen-lang-client-0935527998"
vector_dataset_id = "vector_search"
destination_table_id = "vector_test_table"

# BigQuery 클라이언트 초기화
client = bigquery.Client()

# 테이블 스키마 정의
schema = [
    bigquery.SchemaField("codi_json", "STRING"),  # 각 텍스트를 식별할 수 있는 URI나 ID
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED")  # 임베딩 벡터는 FLOAT64 배열로 저장
]

# 테이블 참조 생성
table_ref = bigquery.Table(f"{project_id}.{vector_dataset_id}.{destination_table_id}", schema=schema)

# 테이블 생성
try:
    table = client.create_table(table_ref)  # 테이블 생성
    print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")
except Exception as e:
    print(f"Failed to create table: {e}")
