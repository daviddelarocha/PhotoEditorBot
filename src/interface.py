import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from src.model import train_model, process_image

def open_train_dialog(progress_var, progress_label):
    data_path = filedialog.askdirectory(title="Select Data Directory")
    output_model_path = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("H5 files", "*.h5")])
    if data_path and output_model_path:
        progress_var.set(0)
        progress_label.config(text="Progress: 0%")
        train_model(data_path, output_model_path, progress_var, progress_label)

def open_predict_dialog():
    image_path = filedialog.askopenfilename(title="Select Image to Process", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    model_path = filedialog.askopenfilename(title="Select Trained Model", filetypes=[("H5 files", "*.h5")])
    output_image_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPG files", "*.jpg")])
    if image_path and model_path and output_image_path:
        process_image(image_path, model_path, output_image_path)

def create_gui():
    root = tk.Tk()
    root.title("Instagram Photo Editor")

    train_button = tk.Button(root, text="Train", command=lambda: open_train_dialog(progress_var, progress_label))
    train_button.pack(pady=10)

    predict_button = tk.Button(root, text="Predict", command=open_predict_dialog)
    predict_button.pack(pady=10)

    global progress_var
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
    progress_bar.pack(pady=10, padx=10, fill=tk.X)

    global progress_label
    progress_label = tk.Label(root, text="Progress: 0%")
    progress_label.pack(pady=5)

    root.mainloop()
