import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
from transformers import AutoProcessor, BarkModel
import numpy as np
import scipy.io.wavfile

# Load model and processor
model = BarkModel.from_pretrained("suno/bark-small")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
processor = AutoProcessor.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"
max_chunk_size = 22

# Function to chunk text
##
# This function chunks the input text into smaller pieces, 
# each containing up to a specified number of words.
# The chunks are returned as a list of strings, which can be 
# processed individually by the TTS model.
#
# @param text The input text to be split into chunks.
# @param max_chunk_size The maximum number of words allowed in 
# each chunk (default is 22).
# @return A list of text chunks, each containing a subset of 
# the original text.
def chunk_text_by_words(text, max_chunk_size=max_chunk_size):
    words = text.split()
    chunks = [" ".join(words[i:i + max_chunk_size]) 
              for i in range(0, len(words), max_chunk_size)]
    return chunks

# Function to process text and generate speech
##
# This function generates speech from the input text 
# using the Bark TTS model.
# It processes the text by splitting it into chunks, 
# converting each chunk into speech, and concatenating 
# the results. The final output is saved as a WAV audio file.
#
# @param text The input text to be converted to speech.
# @param output_path The path where the generated audio 
# will be saved as a WAV file.
def generate_speech(text, output_path):
    try:
        text_chunks = chunk_text_by_words(text)
        speech_outputs = []

        # Loop through text chunks and generate 
        # speech for each
        for i, chunk in enumerate(text_chunks):
            progress_message.set(
                f"Processing chunk {i + 1}/{len(text_chunks)}: {chunk}")
            # Update the GUI to reflect the current progress
            root.update()  

            # Process input chunk for TTS model
            inputs = processor(chunk, voice_preset=voice_preset)
            # Generate speech output from model
            speech_output = model.generate(**inputs.to(device)) 
            # Convert to numpy array and store in list
            speech_outputs.append(speech_output[0].cpu().numpy()) 

        # Concatenate all speech outputs into a single audio
        concatenated_output = np.concatenate(speech_outputs)
        sampling_rate = model.generation_config.sample_rate

        # Save the generated speech to a WAV file
        scipy.io.wavfile.write(output_path, rate=sampling_rate, 
                               data=concatenated_output)

        progress_message.set("Audio generation complete! Saved to: " + output_path)
        messagebox.showinfo("Success", f"Audio saved to {output_path}")
    except Exception as e:
        progress_message.set("Error during processing.")
        messagebox.showerror("Error", str(e))

# Function to handle generate button click
##
# This function is triggered when the user clicks the 
# "Generate Speech" button.
# It retrieves the input text, prompts the user to choose 
# a save location, and calls the generate_speech function.
#
# @return None
def on_generate_click():
    text = input_text.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Required", 
                               "Please enter some text to generate speech.")
        return

    # Prompt user to select the location to save the 
    # generated audio
    output_path = filedialog.asksaveasfilename(
        defaultextension=".wav",
        filetypes=[("WAV files", "*.wav")],
        title="Save Audio File"
    )

    if output_path:
        progress_message.set("Starting audio generation...")
        # Update the GUI before processing
        root.update()  
        # Generate the speech and save it
        generate_speech(text, output_path)  

# Create GUI
##
# This section sets up the graphical user interface using Tkinter. 
# It contains a text input field for entering text,
# a "Generate Speech" button to trigger the speech generation, 
# and a progress label to display the current status.
root = tk.Tk()
root.title("Text-to-Speech Generator")
root.geometry("1000x600") 
# Dark theme background
root.configure(bg="#121212")  

# Input text area
input_label = ttk.Label(root, text="Enter Text:", background="#121212", 
                        foreground="white", font=("Arial", 14))
input_label.pack(pady=10)

input_text = tk.Text(root, wrap=tk.WORD, height=15, font=("Arial", 12), 
                     bg="#1E1E1E", fg="white", insertbackground="white")
input_text.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

# Generate button
generate_button = ttk.Button(root, text="Generate Speech", 
                             command=on_generate_click, style="Custom.TButton")
generate_button.pack(pady=20)

# Progress message
progress_message = tk.StringVar()
progress_label = ttk.Label(root, textvariable=progress_message, 
                           background="#121212", foreground="lightblue", 
                           font=("Arial", 12, "italic"))
progress_label.pack(pady=10)

# Style configuration for custom button appearance
style = ttk.Style()
style.configure("Custom.TButton", font=("Arial", 14), 
                background="#323232", foreground="white")
style.map("Custom.TButton", background=[("active", "#505050")])

# Start GUI loop
##
# This starts the Tkinter event loop, 
# which allows the GUI to run and respond to user actions.
root.mainloop()
