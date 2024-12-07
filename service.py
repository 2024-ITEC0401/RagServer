import json
import os
import re
from dotenv import load_dotenv
from flask import Blueprint, request, jsonify, current_app
from google.cloud import storage
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
PROMPT_TEXT = load_prompt("prompt/analysis_prompt.txt")  # 하드코딩된 프롬프트
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = "asia-northeast3"
BUCKET_NAME = os.getenv("BUCKET_NAME")  # Cloud Storage 버킷 이름

# Vertex AI 초기화
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Cloud Storage 클라이언트 초기화
storage_client = storage.Client()


def upload_image_to_gcs(image):
    """이미지를 Cloud Storage에 업로드하고 공개 URL 반환"""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob_name = f"images/{uuid.uuid4()}.jpg"  # 고유한 이미지 이름 생성
    blob = bucket.blob(blob_name)

    # 이미지를 Cloud Storage에 업로드
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)
    blob.upload_from_file(buffered, content_type="image/jpeg")

    # 파일을 공개로 설정
    blob.make_public()

    # 공개 URL 반환
    return blob.public_url


def send_to_gemini(image):
    # 이미지를 Cloud Storage에 업로드하여 URI 획득
    image_uri = upload_image_to_gcs(image)

    # 모델 초기화
    multimodal_model = GenerativeModel(model_name="gemini-1.5-flash-002")

    # 이미지 URI와 프롬프트 전송
    response = multimodal_model.generate_content(
        [
            Part.from_uri(image_uri, mime_type="image/jpeg"),  # 이미지 URI 사용
            PROMPT_TEXT  # 하드코딩된 프롬프트
        ],
        generation_config={
            "temperature": 0.2,  # temperature 설정
        }
    )

    # 불필요한 ```json 구문 제거 및 JSON 파싱
    outfit_info_str = response.text if response else "No response text found"
    json_match = re.search(r"\{.*\}", outfit_info_str, re.DOTALL)  # 중괄호로 시작하는 JSON 부분 추출

    if json_match:
        json_str = json_match.group(0)  # JSON 부분만 추출
    else:
        current_app.logger.error("No JSON data found in the response.")
        json_str = "{}"  # JSON 부분이 없을 때 빈 객체 반환

    try:
        outfit_info_json = json.loads(json_str) if outfit_info_str else {}
    except json.JSONDecodeError as e:
        current_app.logger.error(f"JSON decode error: {e}")
        outfit_info_json = {"error": "Invalid JSON format received"}

    return outfit_info_json, image_uri


@service_bp.route("/get_outfit_info", methods=["POST"])
def get_outfit_info():
    """
        이미지 파일을 받아 Cloud Storage에 이미지 저장 후 URL과 코디 정보를 반환하는 API
        ---
        tags:
          - Outfit
        consumes:
          - multipart/form-data
        parameters:
          - name: image
            in: formData
            type: file
            required: true
            description: 코디 정보를 추출할 이미지 파일
        responses:
          200:
            description: 성공적으로 코디 정보가 생성된 경우
            schema:
              type: object
              properties:
                imageUri:
                  type: string
                name:
                  type: string
                mainCategory:
                  type: string
                subCategory:
                  type: string
                baseColor:
                  type: string
                pointColor:
                  type: string
                textile:
                  type: string
                pattern:
                  type: string
                season:
                  type: string
                style:
                  type: string
                description:
                  type: string
          400:
            description: 이미지 파일이 제공되지 않았거나 잘못된 요청
          500:
            description: 서버 오류 또는 Gemini API 오류
        """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image = Image.open(image_file)

    try:
        # Vertex AI Gemini API에 이미지와 텍스트 프롬프트 전송
        outfit_info, image_uri = send_to_gemini(image)
        # outfit_info의 속성을 최상위로 이동
        if isinstance(outfit_info, dict):
            response_data = {**outfit_info, "imageUri": image_uri}
        else:
            # outfit_info가 dict가 아닌 경우 빈 값 처리
            response_data = {"imageUri": image_uri}

        return jsonify(response_data)
    except Exception as e:
        current_app.logger.error(f"Error calling Gemini API: {e}")
        return jsonify({"error": str(e)}), 500

@service_bp.route("/delete_image", methods=["DELETE"])
def delete_image():

    try:
        # JSON 요청 본문에서 imageUri 추출
        data = request.get_json()
        image_uri = data.get("imageUri")

        if not image_uri:
            return jsonify({"error": "No image URI provided"}), 400

        # URI에서 Blob 이름 추출
        match = re.search(f"/{BUCKET_NAME}/(.+)", image_uri)
        if not match:
            return jsonify({"error": "Invalid image URI format"}), 400

        blob_name = match.group(1)
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)

        # Blob이 존재하는지 확인
        if not blob.exists():
            return jsonify({"error": "Image not found in Cloud Storage"}), 404

        # Blob 삭제
        blob.delete()
        return jsonify({"message": "Image successfully deleted"}), 200

    except Exception as e:
        current_app.logger.error(f"Error deleting image: {e}")
        return jsonify({"error": str(e)}), 500
