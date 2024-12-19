import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle

# Load the trained model and label mapping
model = tf.keras.models.load_model('gesture_model.h5')
with open('label_to_index.pkl', 'rb') as f:
    label_to_index = pickle.load(f)

# Gesture Recognition
def recognize_gesture():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            data = []
            for lm in landmarks.landmark:
                data.append([lm.x, lm.y, lm.z])
            input_data = np.array(data).reshape(1, -1)

            prediction = model.predict(input_data)
            predicted_label = list(label_to_index.keys())[np.argmax(prediction)]
            cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main flow
if __name__ == "__main__":
    recognize_gesture()
