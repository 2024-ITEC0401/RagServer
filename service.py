import os
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify, current_app
from google.cloud import storage
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from io import BytesIO
from PIL import Image
from config import load_prompt
import uuid

# .env 파일 로드
load_dotenv()

# Blueprint 생성
service_bp = Blueprint('service', __name__)

# GCP 설정
PROMPT_TEXT = load_prompt("prompt.txt")  # 하드코딩된 프롬프트
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = "us-central1"
BUCKET_NAME = os.getenv("BUCKET_NAME")  # Cloud Storage 버킷 이름

# Vertex AI 초기화
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Cloud Storage 클라이언트 초기화
storage_client = storage.Client()


def upload_image_to_gcs(image):
    """이미지를 Cloud Storage에 업로드하고 URI 반환"""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_name = f"images/{uuid.uuid4()}.jpg"  # 고유한 이미지 이름 생성
    blob = bucket.blob(blob_name)

    # 이미지를 Cloud Storage에 업로드
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)
    blob.upload_from_file(buffered, content_type="image/jpeg")

    # 이미지의 공개 URI 생성
    return f"gs://{BUCKET_NAME}/{blob_name}"


def send_to_gemini(image):
    # 이미지를 Cloud Storage에 업로드하여 URI 획득
    image_uri = upload_image_to_gcs(image)

    # 모델 초기화
    multimodal_model = GenerativeModel(model_name="gemini-1.0-pro-vision-001")

    # 이미지 URI와 프롬프트 전송
    response = multimodal_model.generate_content(
        [
            Part.from_uri(image_uri, mime_type="image/jpeg"),  # 이미지 URI 사용
            PROMPT_TEXT  # 하드코딩된 프롬프트
        ]
    )

    # 응답 데이터 파싱
    outfit_info = response.text if response else "No response text found"
    return outfit_info, image_uri  # 응답 텍스트와 이미지 URI 반환


@service_bp.route("/get_outfit_info", methods=["POST"])
def get_outfit_info():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image = Image.open(image_file)

    try:
        # Vertex AI Gemini API에 이미지와 텍스트 프롬프트 전송
        outfit_info, image_uri = send_to_gemini(image)
        return jsonify({"outfit_info": outfit_info, "image_uri": image_uri})
    except Exception as e:
        current_app.logger.error(f"Error calling Gemini API: {e}")
        return jsonify({"error": str(e)}), 500
