from flask import Flask, jsonify

from codi_recommend import codi_recommend_bp
from nl_codi_recommend import nl_codi_recommend_bp
from service import service_bp
from config import Config  # Config 클래스를 import 합니다.
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})
CORS(app, resources={r"/*": {"origins": "http://api.look-4-me.com:8080"}})
app.config.from_object(Config)  # Config 클래스를 적용합니다.
app.register_blueprint(service_bp)
app.register_blueprint(codi_recommend_bp)
app.register_blueprint(nl_codi_recommend_bp)

# Swagger UI 설정
SWAGGER_URL = '/swagger'  # Swagger UI에 접근할 경로
API_URL = '/spec'  # Swagger JSON을 제공할 경로

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Look4Me Rag Server"}
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/spec')
def spec():
    swag = swagger(app)
    swag['info']['title'] = "Look4Me Rag Server API"
    swag['info']['description'] = "RAG, GCP, LLM 관련 API"
    swag['info']['version'] = "1.0.0"
    return jsonify(swag)

@app.route('/')
def hello_world():
    return "Hello World"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

#gcloud config get-value project
