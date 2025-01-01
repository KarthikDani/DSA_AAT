"""
PROBLEM STATEMENT FOR DSA AAT:
The task involves converting a text prompt into speech audio using a pre-trained deep learning model.
__Challenges:
1. Handling text chunks larger than the model's input capacity.
2. Processing and combining multiple outputs to produce a single long length human like audio narration.

"""


import scipy
import torch
from transformers import AutoProcessor, BarkModel
import numpy as np

# Load model and processor
model = BarkModel.from_pretrained("suno/bark-small")
device = "cpu" # "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
processor = AutoProcessor.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

text_prompt = """
B.M.S. College of Engineering (BMSCE) Bengaluru has the unique distinction of being the first private engineering college established in the country. 
"""

# Parameters for chunking
max_chunk_size = 22  # Maximum number of words per chunk

def chunk_text_by_words(text, max_chunk_size=max_chunk_size):
    """Function to chunk text by words with a specified max chunk size"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_chunk_size):
        chunk = words[i:i + max_chunk_size]
        chunks.append(" ".join(chunk))  # Join the words into a string

    return chunks

# Process and generate speech for each chunk
speech_outputs = []
text_chunks = chunk_text_by_words(text_prompt)

for i, chunk in enumerate(text_chunks):
    print(f"Processing chunk {i + 1}/{len(text_chunks)}: {chunk}")  # Debug statement to track progress
    inputs = processor(chunk, voice_preset=voice_preset)
    
    # Generate speech for the current chunk
    speech_output = model.generate(**inputs.to(device))

    # Append the generated speech output directly
    speech_outputs.append(speech_output[0].cpu().numpy())

# Concatenate audio outputs
concatenated_output = np.concatenate(speech_outputs)

# Save to a wav file
sampling_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out_new.wav", rate=sampling_rate, data=concatenated_output)
