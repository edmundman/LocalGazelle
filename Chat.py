import torch
import transformers
import torchaudio
import keyboard
import sounddevice as sd
import numpy as np
import pyttsx3
from gazelle import (
    GazelleConfig,
    GazelleForConditionalGeneration,
    GazelleProcessor,
)

model_id = "tincans-ai/gazelle-v0.2"
config = GazelleConfig.from_pretrained(model_id)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
device = "cpu"
dtype = torch.float32

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
    print(f"Using {device} device")
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
    print(f"Using {device} device")

model = GazelleForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
).to(device, dtype=dtype)
model = model.to(dtype=dtype)

audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

def inference_collator(audio_input, prompt="Reply to  the following  : \n<|audio|>"):
    audio_values = audio_processor(
        audio=audio_input, return_tensors="pt", sampling_rate=16000
    ).input_values
    msgs = [
        {"role": "user", "content": prompt},
    ]
    labels = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True
    )
    return {
        "audio_values": audio_values.squeeze(0).to(model.device).to(model.dtype),
        "input_ids": labels.to(model.device),
    }

def record_audio():
    fs = 16000
    duration = 5  # Recording duration in seconds
    print("Recording started. Hold down the space bar to record.")
    
    audio_data = []
    recording = False
    while True:
        if keyboard.is_pressed('space'):
            if not recording:
                print("Recording...")
                recording = True
                audio_data.append(sd.rec(int(duration * fs), samplerate=fs, channels=1))
        else:
            if recording:
                print("Recording stopped.")
                recording = False
                sd.stop()
                break
    
    audio_data = np.concatenate(audio_data, axis=0)
    audio_data = audio_data.reshape(1, -1)  # Reshape to a 2D tensor
    
    # Save the recorded audio as a WAV file
    output_path = "recorded_audio.wav"
    torchaudio.save(output_path, torch.from_numpy(audio_data), fs)
    
    return output_path

def process_audio(audio_path):
    # Load the recorded audio
    test_audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        test_audio = torchaudio.transforms.Resample(sr, 16000)(test_audio)

    # Run the audio through the LLM
    inputs = inference_collator(test_audio)
    response = tokenizer.decode(model.generate(**inputs, max_new_tokens=64)[0])
    
    # Remove the unwanted tags from the response
    response = response.replace("<s> [INST]", "").replace("[/INST]", "").replace("</s>", "").strip()
    
    # Extract the audio content from the response
    audio_content = response.split("<|audio|>")[-1].strip()
    
    print("Response:", response)
    
    # Convert the audio content to speech using pyttsx3
    engine = pyttsx3.init()
    engine.say(audio_content)
    engine.runAndWait()
    
    return response

while True:
    # Record audio and save it as a WAV file
    audio_path = record_audio()
    
    # Process the recorded audio and get the response
    response = process_audio(audio_path)
