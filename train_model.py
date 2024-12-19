import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle

# Step 1: Data Collection
def collect_data(label, num_samples=100):
    cap = cv2.VideoCapture(0)
    samples = []
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)

    print(f"Collecting data for: {label}")
    
    while len(samples) < num_samples:
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
            samples.append(data)
            cv2.putText(frame, f"Collecting {label}: {len(samples)}/{num_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return np.array(samples)

# Step 2: Prepare Training Data
def prepare_data():
    data = []
    labels = []
    
    for label in ['hi', 'thank you', 'help', 'love']:  # Example labels
        samples = collect_data(label)
        data.extend(samples)
        labels.extend([label] * len(samples))
    
    return np.array(data), np.array(labels)

# Step 3: Train the Model
def train_model(data, labels):
    data = np.array(data)
    data = data.reshape(data.shape[0], -1)

    label_to_index = {label: index for index, label in enumerate(np.unique(labels))}
    indexed_labels = np.array([label_to_index[label] for label in labels])

    X_train, X_test, y_train, y_test = train_test_split(data, indexed_labels, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(label_to_index), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    return model, label_to_index

# Step 4: Save the Model
def save_model(model, label_to_index):
    model.save('gesture_model.h5')
    with open('label_to_index.pkl', 'wb') as f:
        pickle.dump(label_to_index, f)

# Main flow
if __name__ == "__main__":
    data, labels = prepare_data()
    model, label_to_index = train_model(data, labels)
    save_model(model, label_to_index)
