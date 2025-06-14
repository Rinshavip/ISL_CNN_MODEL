import PIL
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, END
from PIL import Image, ImageTk, ImageSequence
from tensorflow.keras.models import load_model
import threading
import time
import pyttsx3
import wordninja
from rapidfuzz import process
import speech_recognition as sr
import os

model = load_model("D:/RIN/MODELS/ISL_CNN_model_3.h5")
class_labels = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

IMG_SIZE = 48
CHANNELS = 1
confidence_threshold = 0.7
STABILITY_DURATION = 0.7
speak_lock = threading.Lock()
engine = pyttsx3.init()
engine.setProperty('rate', 150)

valid_sentences_path = "valid_sentences.txt"

def speak_text(text):
    def run():
        with speak_lock:
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

def load_valid_sentences(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip().lower() for line in f if line.strip()]

def correct_with_rapidfuzz(raw_text, valid_options):
    segmented = wordninja.split(raw_text)
    joined = ' '.join(segmented)
    best_match, score, _ = process.extractOne(joined, valid_options)
    if score >= 70:
        return best_match
    else:
        return joined

valid_sentences = load_valid_sentences("valid_sentences.txt")

op_dest = "D:/RIN/PRJCT/gif_dataset/words/"
alpha_dest = "D:/RIN/PRJCT/gif_dataset/alphabet/"

dirListing = os.listdir(op_dest)

editFiles = []
for item in dirListing:
    if ".webp" in item:
        editFiles.append(item)

# Map words to pre-saved GIFs
def check_sim(i, file_map):
    for item in file_map:
        for word in file_map[item]:
            if i == word:
                return 1, item
    return -1, ""

file_map = {}
for i in editFiles:
    tmp = i.replace(".webp", "")
    tmp = tmp.split()
    file_map[i] = tmp

def func(a):
    all_frames = []
    final = PIL.Image.new('RGB', (380, 260))
    words = a.split()
    for i in words:
        flag, sim = check_sim(i, file_map)
        if flag == -1:
            for j in i:
                im = PIL.Image.open(alpha_dest + str(j).lower() + "_small.gif")
                frameCnt = im.n_frames
                for frame_cnt in range(frameCnt):
                    im.seek(frame_cnt)
                    im.save("tmp.png")
                    img = cv2.imread("tmp.png")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (380, 260))
                    im_arr = PIL.Image.fromarray(img)
                    for itr in range(15):
                        all_frames.append(im_arr)
        else:
            im = PIL.Image.open(op_dest + sim)
            im.info.pop('background', None)
            im.save('tmp.gif', 'gif', save_all=True)
            im = PIL.Image.open("tmp.gif")
            frameCnt = im.n_frames
            for frame_cnt in range(frameCnt):
                im.seek(frame_cnt)
                im.save("tmp.png")
                img = cv2.imread("tmp.png")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (380, 260))
                im_arr = PIL.Image.fromarray(img)
                all_frames.append(im_arr)
    final.save("out.gif", save_all=True, append_images=all_frames, duration=100, loop=0)
    return all_frames

# ------------------ GUI PART -----------------------
class Tk_Manage(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage,VoiceToSign, SignToVoice):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Two Way Sign Language Translator", font=("Verdana", 12))
        label.pack(pady=10, padx=10)
        button = tk.Button(self, text="Voice to Sign", command=lambda: controller.show_frame(VoiceToSign))
        button.pack()
        button2 = tk.Button(self, text="Sign to Voice", command=lambda: controller.show_frame(SignToVoice))
        button2.pack()
        load = PIL.Image.open("Two Way Sign Language Translator.png.gif")
        load = load.resize((620, 450))
        render = ImageTk.PhotoImage(load)
        img = tk.Label(self, image=render)
        img.image = render
        img.place(x=100, y=200)


class SignToVoice(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        label = tk.Label(self, text="Sign to Speech", font=("Verdana", 16))
        label.pack(pady=10)

        self.video_label = tk.Label(self)
        self.video_label.pack()

        self.start_btn = tk.Button(self, text="Start", command=self.start_prediction)
        self.start_btn.pack(pady=5)

        self.back_btn = tk.Button(self, text="Back", command=lambda: controller.show_frame(StartPage))
        self.back_btn.pack()

        self.cap = None
        self.running = False

    def start_prediction(self):
        self.running = True
        self.text_output = ""
        self.last_confirmed_char = ""
        self.stable_char = ""
        self.stable_start_time = None
        self.cap = cv2.VideoCapture(0)
        self.video_stream()

    def video_stream(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        x1, y1, x2, y2 = 100, 100, 300, 300
        roi = frame[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        roi_normalized = roi_gray / 255.0
        roi_input = np.expand_dims(roi_normalized, axis=0)

        prediction = model.predict(roi_input, verbose=0)
        class_index = np.argmax(prediction)
        pred_label = class_labels[class_index]
        confidence = prediction[0][class_index]

        current_time = time.time()
        if confidence >= confidence_threshold:
            if pred_label == self.stable_char:
                if self.stable_start_time and (current_time - self.stable_start_time) >= STABILITY_DURATION:
                    if pred_label != self.last_confirmed_char:
                        self.text_output += pred_label
                        speak_text(pred_label)
                        self.last_confirmed_char = pred_label
                        self.stable_start_time = None
            else:
                self.stable_char = pred_label
                self.stable_start_time = current_time
        else:
            self.stable_char = ""
            self.stable_start_time = None

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Predicted: {pred_label} ({confidence:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.configure(image=img)
        self.video_label.image = img

        if self.running:
            self.after(10, self.video_stream)

    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
        valid_sentences = load_valid_sentences(valid_sentences_path)
        if self.text_output:
            corrected = correct_with_rapidfuzz(self.text_output, valid_sentences)
            speak_text("The corrected sentence is " + corrected)

class VoiceToSign(tk.Frame):
    def __init__(self, parent, controller):
        cnt = 0
        gif_frames = []
        inputtxt = None
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Voice to Sign", font=("Verdana", 12))
        label.pack(pady=10, padx=10)
        gif_box = tk.Label(self)

        button1 = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button1.pack()
        button2 = tk.Button(self, text="Sign to Voice", command=lambda: controller.show_frame(SignToVoice))
        button2.pack()

        def gif_stream():
            nonlocal cnt
            nonlocal gif_frames
            if cnt == len(gif_frames):
                return
            img = gif_frames[cnt]
            cnt += 1
            imgtk = ImageTk.PhotoImage(image=img)
            gif_box.imgtk = imgtk
            gif_box.configure(image=imgtk)
            gif_box.after(50, gif_stream)

        def hear_voice():
            nonlocal inputtxt
            store = sr.Recognizer()
            with sr.Microphone() as s:
                audio_input = store.record(s, duration=10)
                try:
                    text_output = store.recognize_google(audio_input)
                    inputtxt.insert(END, text_output)
                except:
                    print("Error Hearing Voice")
                    inputtxt.insert(END, '')

        def Take_input():
            INPUT = inputtxt.get("1.0", "end-1c")
            print(INPUT)
            nonlocal gif_frames
            gif_frames = func(INPUT)
            nonlocal cnt
            cnt = 0
            gif_stream()
            gif_box.place(x=400, y=160)

        l = tk.Label(self, text="Enter Text or Voice:")
        l1 = tk.Label(self, text="OR")
        inputtxt = tk.Text(self, height=4, width=25)
        voice_button = tk.Button(self, height=2, width=20, text="Record Voice", command=lambda: hear_voice())
        voice_button.place(x=50, y=180)
        Display = tk.Button(self, height=2, width=20, text="Convert", command=lambda: Take_input())
        l.place(x=50, y=160)
        l1.place(x=115, y=230)
        inputtxt.place(x=50, y=250)
        Display.pack()


if __name__ == "__main__":
    app = Tk_Manage()
    app.geometry("800x750")
    app.mainloop()


