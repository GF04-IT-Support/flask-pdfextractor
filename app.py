from flask import Flask
from flask_cors import CORS
from routes.invigilators.routes import invigilators_bp
from routes.exams_schedule.routes import exams_schedule_bp

app = Flask(__name__)
CORS(app, origins="*") 

app.register_blueprint(invigilators_bp, url_prefix='/invigilators')
app.register_blueprint(exams_schedule_bp, url_prefix='/exams-schedule')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
