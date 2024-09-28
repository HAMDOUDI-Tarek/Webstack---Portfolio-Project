
from pathlib import Path
import tkinter
# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import numpy as np
import tensorflow 
from tensorflow.keras.models import load_model
import os
import wfdb
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LeakyReLU 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk, messagebox
import datagui 


folder_path = None

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\Expert Info\Desktop\build\assets\frame0")

import sys
sys.path.append('C:/Users/Expert Info/Desktop/build/arrhythmia')
def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def set_folder_path(path):
    global folder_path
    folder_path = path

# Import the Data GUI Window
def open_new_window():
    datagui.create_new_window(window, callback=set_folder_path)

# Preprocessing ECG
def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size), 'valid') / window_size

def preprocess_signal(show_message=True):
    global folder_path

    record = wfdb.rdrecord(os.path.join(folder_path, '100'))
    annotation = wfdb.rdann(folder_path, '/100.atr')
    signal = record.p_signal

    window_size = 3 # Define the window size for the moving average filter
    smoothed_signal1 = moving_average(signal[:, 0], window_size)
    smoothed_signal2 = moving_average(signal[:, 1], window_size)
    smoothed_ecg_signal = np.column_stack((smoothed_signal1, smoothed_signal2))

    N = 3 # Order of filter
    Wn = 25  # Cutoff frequency in Hz
    b, a = butter(N, Wn, fs=360, btype='low', analog=False) # Create the Butterworth filter
    filtered_signal = filtfilt(b, a, smoothed_ecg_signal, axis=0) # Apply the filter to the ECG signal

    kernel_size = 1 # Define the kernel size for the median filter
    filtered_2_signal = medfilt(filtered_signal, kernel_size) # Apply the median filter to the ECG signal

    # Normalize ECG signal
    normalized_signal_lead1 = (filtered_2_signal[:, 0] - np.min(signal[:, 0])) / (np.max(signal[:, 0]) - np.min(signal[:, 0])) * 2 - 1
    normalized_signal_lead2 = (filtered_2_signal[:, 1] - np.min(signal[:, 1])) / (np.max(signal[:, 1]) - np.min(signal[:, 1])) * 2 - 1
    normalized_signal = np.column_stack((normalized_signal_lead1, normalized_signal_lead2))

    if show_message:
        tkinter.messagebox.showinfo('Preprocessing', "ECG Preprocessed succesfully")

        # Calculate the number of samples for 5 seconds
        fs = 360
        num_samples = 10 * fs

        # Create a time vector
        time = [i/fs for i in range(num_samples)]
        filtered_signal = filtered_2_signal
        ax.clear()
        ax.plot(filtered_signal[:num_samples, 0], color='k')
        ax.set_title('Processed Signal for Patient')
        ax.set_ylabel('Amplitude', labelpad=10, fontsize=12)
        ax.set_xlabel('Time (s)')
        ax.grid(which='both', linewidth=0.5, linestyle='--')
        ax.set_ylim(-1, 1)
        ax.tick_params(axis='y', labelsize=10)
        canva.draw()

    return signal, normalized_signal, annotation

class_names = {
    0: 'Paced beats',
    1: 'Atrial Premature beats',
    2: 'Left Bundle Branch Block beats',
    3: 'Normal beats',
    4: 'Right Bundle Branch Block beats',
    5: 'Premature ventricular contraction',
    6: 'Unclassifiable beats'
}

# Arrhythmia Prediction Function
def predict_arrhythmias(heartbeats_list):
    # Load deep learning model
    model = load_model('C:/Users/Expert Info/Desktop/build/arrhythmia/ArrhythClassModel.h5', 
                       custom_objects={'LeakyReLU': LeakyReLU})

    # Run the prediction
    predictions = model.predict(heartbeats_list)

    # Get the predicted class indices (0, 1, 2, 3, 4, 5)
    predicted_classes = np.argmax(predictions, axis=1)

    # Count the occurrences of each arrhythmia type
    arrhythmia_counts = np.bincount(predicted_classes)

    # Return the counts as a dictionary with class names
    return {class_names[i]: count for i, count in enumerate(arrhythmia_counts) if count > 0}

# Run the Prediction Function
def predict_arrhythmias_button_clicked():

    # Get the signal and annotation 
    _, processed_signal, annotation = preprocess_signal(show_message=False)

    # Get the heartbeats list
    heartbeats_list = get_heartbeats_list(processed_signal, annotation)

    # Call the predict_arrhythmias function
    arrhythmia_counts = predict_arrhythmias(heartbeats_list)

    # Display the arrhythmia counts in the text widget
    display_arrhythmia_counts(arrhythmia_counts)

# Get Data Function
def get_heartbeats_list(ecg_signals, annotation):
    beat_ann=['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'F', 'e', 'j', 'E', '/', 'f', 'Q']
    wanted_symbol=['N', 'L', 'R', 'A', 'V', '/']

    def filter_annotations(annotation, wanted_symbols):
        filtered_record_annotations = []
        filtered_samples = []
        filtered_aux_notes = []
        for i, symbol in enumerate(annotation.symbol):
            if symbol in wanted_symbols:
                filtered_record_annotations.append(symbol)
                filtered_samples.append(annotation.sample[i])
                filtered_aux_notes.append(annotation.aux_note[i])
        return wfdb.Annotation(record_name=annotation.record_name,
                            extension=annotation.extension,
                            sample=filtered_samples,
                            symbol=filtered_record_annotations,
                            aux_note=filtered_aux_notes)

    f_annotations = filter_annotations(annotation, beat_ann)

    def segment_ecg_signals(ecg_signals, f_annotations):
        segments = []
        for loc in f_annotations.sample:
            start = max(0, loc - 60)
            end = min(len(ecg_signals), loc + 90)
            segment = ecg_signals[start:end]
            segments.append(segment)
        return segments

    segmented_beats = segment_ecg_signals(ecg_signals, f_annotations)

    def filter_beats(segmented_beats, f_annotations, wanted_symbols):
        filtered_beats = [beat for beat, symbol in zip(segmented_beats, f_annotations.symbol) if symbol in wanted_symbols]
        return filtered_beats

    up_segmented_beats = filter_beats(segmented_beats, f_annotations, wanted_symbol)

    # Separate the leads
    X_lead1 = []
    X_lead2 = []

    for beat in up_segmented_beats:
        lead1, lead2 = beat[:, 0], beat[:, 1]
        X_lead1.append(lead1)
        X_lead2.append(lead2)

    # Pad the sequences to the maximum length
    X_lead1_padded = pad_sequences(X_lead1, dtype='float32', padding='post')
    X_lead2_padded = pad_sequences(X_lead2, dtype='float32', padding='post')

    # Convert the padded lists of lists into 3D numpy arrays
    X_lead1 = np.array(X_lead1_padded)
    X_lead2 = np.array(X_lead2_padded)

    # Stack lead1 and lead2 along the last dimension
    X = np.stack((X_lead1, X_lead2), axis=-1)

    return X

def display_arrhythmia_counts(arrhythmia_counts):
    text2.delete("1.0", "end")  # Clear the text widget
    for arrhythmia, count in arrhythmia_counts.items():
        text2.insert("end", f"{arrhythmia}: {count}\n")

def predict_arrhythmias_button_2_clicked():

    # Get the signal and annotation 
    _, processed_signal, annotation = preprocess_signal(show_message=False)

    # Get the heartbeats list
    heartbeats_list = get_heartbeats_list(processed_signal, annotation)

    arrhythmia_counts = predict_arrhythmias(heartbeats_list)
    total_beats = sum(arrhythmia_counts.values())
    arrhythmia_percentages = {class_name: count/total_beats * 100 for class_name, count in arrhythmia_counts.items()}
    diagnosis = get_diagnosis(arrhythmia_percentages)
    display_diagnosis(arrhythmia_percentages, diagnosis)

def get_diagnosis(arrhythmia_percentages):
    if arrhythmia_percentages['Normal beats'] > 90:
        return 'Normal'
    elif arrhythmia_percentages['Atrial Premature beats'] > 5:
        return 'Atrial Premature beats'
    elif arrhythmia_percentages['Premature ventricular contraction'] > 5:
        return 'Premature ventricular contraction'
    else:
        return 'Unclassifiable'

def display_diagnosis(arrhythmia_percentages, diagnosis):
    text3.delete("1.0", "end")
    for arrhythmia, percentage in arrhythmia_percentages.items():
        text3.insert("end", f"{arrhythmia}: {percentage:.2f}%\n")
    text3.insert("end", f"Diagnosis: {diagnosis}\n")

window = Tk()

window.geometry("1000x600")
window.configure(bg = "#2C852F")


canvas = Canvas(
    window,
    bg = "#2C852F",
    height = 600,
    width = 1000,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    19.0,
    17.0,
    981.0,
    583.0,
    fill="#FFFFFF",
    outline="")

canvas.create_rectangle(
    630.0,
    31.0,
    871.0,
    63.0,
    fill="#FFFDFD",
    outline="")

canvas.create_text(
    741.0,
    39.0,
    anchor="nw",
    text="home",
    fill="#100F0F",
    font=("Inter", 12 * -1)
)

canvas.create_text(
    861.0,
    39.0,
    anchor="nw",
    text="support",
    fill="#100F0F",
    font=("Inter", 12 * -1)
)

canvas.create_text(
    798.0,
    39.0,
    anchor="nw",
    text="account",
    fill="#100F0F",
    font=("Inter", 12 * -1)
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    939.0,
    46.0,
    image=image_image_1
)

# 'Import Signal' Button
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=open_new_window,
    relief="flat"
)
button_1.place(
    x=97.0,
    y=110.0,
    width=169.0,
    height=44.0
)

# 'Diagnosis' Button
button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=predict_arrhythmias_button_2_clicked,
    relief="flat"
)
button_2.place(
    x=672.0,
    y=312.0,
    width=169.0,
    height=44.0
)

# 'Analysis' Button
button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=predict_arrhythmias_button_clicked,
    relief="flat"
)
button_3.place(
    x=209.0,
    y=312.0,
    width=169.0,
    height=44.0
)

# 'Preprocess' Button
button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=preprocess_signal,
    relief="flat"
)
button_4.place(
    x=97.0,
    y=201.0,
    width=169.0,
    height=44.0
)

canvas.create_rectangle(
    41.5,
    61.5,
    956.0197143554688,
    63.0,
    fill="#000000",
    outline="")

canvas.create_rectangle(
    59.5,
    298.4999999999998,
    940.0,
    300.0,
    fill="#7E7B7B",
    outline="")

# Logo 
image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas.create_image(
    97.0,
    40.0,
    image=image_image_2
)

canvas.create_rectangle(
    574.0,
    368.0,
    940.0,
    561.0,
    fill="#D9D9D9",
    outline="")

# Signal Display


# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(7, 3))

# Create a FigureCanvasTkAgg object to display the plot in Tkinter
canva = FigureCanvasTkAgg(fig, master=window)  # window is the Tkinter Tk object
canva.draw()
canva.get_tk_widget().place(x=326.0, y=86.0, width=587.0, height=193.0)

# Create a Text widget for the Analysis Display
text2 = tkinter.Text(canvas, bg="#D9D9D9", bd=0, font=("Helvetica", 14))
text2.place(x=61.0, y=368.0, width=466.0, height=193.0)

# Create a Text widget for the Diagnosis Display
text3 = tkinter.Text(canvas, bg="#D9D9D9", bd=0, font=("Helvetica", 14))
text3.place(x=574.0, y=368.0, width=366.0, height=193.0)

window.resizable(False, False)
window.mainloop()
