from dotenv import load_dotenv
from flask import Flask
from service import service_bp
from config import Config  # Config 클래스를 import 합니다.

app = Flask(__name__)
app.config.from_object(Config)  # Config 클래스를 적용합니다.
app.register_blueprint(service_bp)

@app.route('/')
def hello_world():
    return "Hello World"

if __name__ == '__main__':
    app.run(debug=True)

#gcloud config get-value project
