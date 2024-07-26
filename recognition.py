import cv2
import joblib
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN

# Load the trained model and encoder
model = joblib.load('C:\\Users\\User\\Desktop\\rec project\\face_recognition_model.pkl')
encoder = joblib.load('C:\\Users\\User\\Desktop\\rec project\\label_encoder.pkl')
embedder = FaceNet()
detector = MTCNN()

def extract_face(image, required_size=(160, 160)):
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

def recognize_face(face):
    face_embedding = embedder.embeddings(face)
    yhat_class = model.predict(face_embedding)
    yhat_prob = model.predict_proba(face_embedding)
    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100
    predict_names = encoder.inverse_transform(yhat_class)
    return predict_names[0], class_probability

# Real-time face recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract face
    face = extract_face(frame)
    if face is not None:
        # Recognize face
        name, probability = recognize_face(face)
        
        # Draw bounding box and label
        faces = detector.detect_faces(frame)
        if len(faces) > 0:
            x1, y1, width, height = faces[0]['box']
            x2, y2 = x1 + width, y1 + height
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{name} - Safe ({probability:.2f}%)'
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Display the result
    cv2.imshow('Real-Time Face Recognition', frame)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
