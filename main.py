import cv2
import numpy as np
from flask import Flask, jsonify, request
import base64
from yolo_segmentation import YOLOSegmentation

app = Flask(__name__)

# Initialiser le détecteur de segmentation YOLO avec le modèle "yolov8m-seg.pt"
ys = YOLOSegmentation("yolov8m-seg.pt")

# Définition d'une route pour l'API
@app.route('/api', methods=['POST'])
def get_data():
    try:
        nmbOfPeople = 0

        # Récupération des données depuis le corps de la requête en format JSON
        data = request.get_json()
        base64_string = data.get('base64_string')

        if base64_string:
            # Décodage de la valeur base64 reçue
            decoded_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(decoded_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Redimensionner l'image
            img = cv2.resize(img, None, fx=0.7, fy=0.7)

            # Effectuer la détection d'objets et obtenir les résultats
            bboxes, classes, segmentations, scores = ys.detect(img)

            # Parcourir les résultats de la détection pour chaque objet détecté
            for class_id in classes:
                # Vérifier si la classe détectée correspond à la classe d'indice 0 (remplacer par le bon ID de classe si nécessaire)
                if class_id == 0:
                    nmbOfPeople += 1

            return jsonify({'number_of_people': nmbOfPeople, 'status': 'success'})
        else:
            return jsonify({'message': 'Aucune donnée base64 fournie', 'status': 'error'}), 400

    except Exception as e:
        # En cas d'erreur lors du décodage ou du traitement
        error_message = f'Erreur : {str(e)}'
        error_data = {
            'message': error_message,
            'status': 'error'
        }
        return jsonify(error_data), 400  # Réponse avec un code d'erreur 400 Bad Request

if __name__ == '__main__':
    app.run(debug=True)
