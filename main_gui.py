import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import json

# Load model and label map
model = tf.keras.models.load_model("model.h5")
with open("label_map.json", "r") as f:
    reverse_label_map = json.load(f)

# Constants
CANVAS_SIZE = 280
MODEL_INPUT_SIZE = 96  # Matches training image size

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Recognition")

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict)
        self.predict_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear)
        self.clear_button.pack(side=tk.LEFT)

        self.label = tk.Label(root, text="", font=("Helvetica", 24))
        self.label.pack()

        # PIL image to draw on
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw_obj = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw_obj.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw_obj.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=255)
        self.label.config(text="")

    def predict(self):
        # Crop the image to bounding box of drawn area
        bbox = self.image.getbbox()
        if not bbox:
            self.label.config(text="Draw something first!")
            return

        cropped = self.image.crop(bbox)

        # Resize to model input
        resized = cropped.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)

        # Normalize and reshape
        img_array = np.array(resized).astype("float32") / 255.0
        img_array = img_array.reshape(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 1)

        # Predict
        prediction = model.predict(img_array)
        predicted_idx = np.argmax(prediction)
        predicted_char = reverse_label_map[str(predicted_idx)]

        self.label.config(text=f"Prediction: {predicted_char}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
