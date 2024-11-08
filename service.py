import os

from dotenv import load_dotenv
from flask import Blueprint, request, jsonify, current_app
from google.oauth2 import service_account
from google.cloud import aiplatform
import base64
from io import BytesIO
from PIL import Image
from config import load_prompt


# .env 파일 로드
load_dotenv()
# Blueprint 생성
service_bp = Blueprint('service', __name__)

# GCP 설정
PROMPT_TEXT = load_prompt("prompt.txt")  # 하드코딩된 프롬프트
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
aiplatform.init(credentials=credentials)

def send_to_gemini(image):
    # 이미지 파일을 Base64로 인코딩
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode()

    # GCP 프로젝트 ID 및 모델 설정
    myProjectId = current_app.config["PROJECT_ID"]
    endpointUrl = f"projects/{myProjectId}/locations/us-central1/publishers/google/models/gemini-1.5-flash-002"

    # 멀티모달 입력 데이터 구성
    instances = [
        {
            "content": {
                "text": PROMPT_TEXT
            },
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": encoded_image
            }
        }
    ]

    # 모델 호출
    endpoint = aiplatform.gapic.PredictionServiceClient()
    response = endpoint.predict(
        endpoint=endpointUrl,
        instances=instances,
        parameters={}
    )

    # 응답 데이터 파싱
    if response.predictions:
        return response.predictions[0].get("content", {}).get("text", "No response text found")
    return "No response text found"

@service_bp.route("/get_outfit_info", methods=["POST"])
def get_outfit_info():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image = Image.open(image_file)

    try:
        # Gemini API에 이미지와 텍스트 프롬프트 전송
        outfit_info = send_to_gemini(image)
        return jsonify({"outfit_info": outfit_info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
