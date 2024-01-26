from flask import Blueprint, jsonify, request
from .exams_schedule_extractor import exams_main

exams_schedule_bp = Blueprint('exams_schedule', __name__)

@exams_schedule_bp.route('/extract', methods=['POST'])
def extract_exams_schedule():
    try:
        data = request.get_json()
        base64_pdf_data = data.get('base64_pdf_data')
        result = exams_main(base64_pdf_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
