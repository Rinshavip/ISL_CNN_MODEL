import cv2
import numpy as np
import threading
from tensorflow.keras.models import load_model
import pyttsx3
from textblob import TextBlob
import time
from rapidfuzz import fuzz, process
import wordninja  # For splitting joined words

# Load the trained model
model = load_model("D:\RIN\MODELS\ISL_CNN_model_3.h5")

# Define class labels (update this according to your model)
class_labels =  ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] # example labels

# Parameters
confidence_threshold = 0.7
STABILITY_DURATION = 0.7

engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Speak function in a separate thread
speak_lock = threading.Lock()

def speak_text(text):
    def run():
        with speak_lock:
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()


# Text variables
text_output = ""
last_confirmed_char = ""
stable_char = ""
stable_start_time = None


# Define the input size expected by your model
IMG_SIZE = 48  # or 128, depending on your model
CHANNELS = 1   # use 1 for grayscale

# Initialize webcam
cap = cv2.VideoCapture(0)
# Add more detailed webcam check
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define a region of interest (ROI) for prediction
    x1, y1, x2, y2 =100,100,300, 300  # adjust as needed
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    roi_gray = np.expand_dims(roi_gray, axis=-1)    # Add channel dimension
    roi_normalized = roi_gray / 255.0    # Normalize
    roi_input = np.expand_dims(roi_normalized, axis=0)     # Add batch dimension

    # Predict class
    prediction = model.predict(roi_input)
    class_index = np.argmax(prediction)
    pred_label = class_labels[class_index]
    confidence = prediction[0][class_index]  # Get the confidence value

    current_time = time.time()
    if confidence >= confidence_threshold:
        if pred_label == stable_char:
            # Sign is stable
            if stable_start_time and (current_time - stable_start_time) >= STABILITY_DURATION:
                if pred_label != last_confirmed_char:
                    # Confirm character and speak
                    text_output += pred_label
                    print(f"Confirmed: {pred_label}")
                    speak_text(pred_label)
                    last_confirmed_char = pred_label
                    stable_start_time = None  # Reset
        else:
            # New potential stable sign detected
            stable_char = pred_label
            stable_start_time = current_time
    else:
        stable_char = ""
        stable_start_time = None

        # Display predictions and confidence
    display_label = pred_label if confidence >= confidence_threshold else "..."
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Predicted: {pred_label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Confidence: {confidence:.2f}", (50,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)

    cv2.imshow("Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


def load_valid_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]

valid_sentences = load_valid_sentences("valid_sentences.txt")

def correct_with_rapidfuzz(raw_text, valid_options):
    segmented = wordninja.split(raw_text)  # First, split combined letters
    joined = ' '.join(segmented)

    # Find the closest sentence from known phrases
    best_match, score, _ = process.extractOne(joined, valid_options)

    print(f"Segmented: {joined}")
    print(f"Best match from RapidFuzz: {best_match} (Score: {score})")

    if score >= 70:  # You can tune this threshold
        return best_match
    else:
        return joined  # fallback to original segmentation


# Final correction and speech
if text_output:
    print(f"\nFinal raw text: {text_output}")
    corrected_text = correct_with_rapidfuzz(text_output, valid_sentences)
    print(f"Corrected text: {corrected_text}")
    speak_text("The corrected sentence is " + corrected_text)
    time.sleep(7)
else:
    print("No valid input.")