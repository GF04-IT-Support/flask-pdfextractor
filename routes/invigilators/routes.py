from flask import Blueprint, jsonify, request
from .invigilators_extractor import invigilators_main

invigilators_bp = Blueprint('invigilators', __name__)

@invigilators_bp.route('/extract', methods=['POST'])
def extract_invigilators():
    try:
        data = request.get_json()
        base64_pdf_data = data.get('base64_pdf_data')
        result = invigilators_main(base64_pdf_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
