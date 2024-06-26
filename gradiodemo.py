"""Gradio demo of Gazelle.

This demo interfaces with Gazelle, a joint speech-language model by Tincans.

We utilize a single environment with GPU and ML frameworks to serve the Gradio app and the model.
"""

import os
import time
from threading import Thread

import gradio as gr
import numpy as np
import torch
import torchaudio
from gazelle import GazelleConfig, GazelleForConditionalGeneration
from transformers import AutoProcessor, AutoTokenizer, TextIteratorStreamer

MODEL_NAME = "tincans-ai/gazelle-v0.2"
AUDIO_MODEL_NAME = "facebook/wav2vec2-base-960h"
MODEL_DIR = "/model"


def download_model():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_DIR,
    )
    move_cache()


def download_samples():
    import concurrent.futures
    import os

    import requests

    remote_urls = [
        "test6.wav",
        "test21.wav",
        "test26.wav",
        "testnvidia.wav",
        "testdoc.wav",
        "testappt3.wav",
    ]

    def download_file(url):
        base_url = "https://r2proxy.tincans.ai/"
        full_url = base_url + url
        filename = os.path.basename(url)
        if not os.path.exists(filename):
            response = requests.get(full_url)
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {filename}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        executor.map(download_file, remote_urls)


class GazelleModel:
    def __init__(self):
        t0 = time.time()
        print("Loading model...")

        config = GazelleConfig.from_pretrained(MODEL_NAME)

        self.model = GazelleForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            config=config,
            torch_dtype=torch.bfloat16,
        )

        print(f"Model loaded in {time.time() - t0:.2f}s")

        self.audio_processor = AutoProcessor.from_pretrained(AUDIO_MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        self.model.config.use_cache = True
        self.model.cuda()
        self.model.eval()

    def generate(self, input="", audio=None, history=[]):
        if input == "" and not audio:
            return

        if "<|audio|>" in input and not audio:
            raise ValueError(
                "Audio input required if '<|audio|>' token is present in input"
            )

        if audio and "<|audio|>" not in input:
            input = "<|audio|> \n\n" + input

        t0 = time.time()

        assert len(history) % 2 == 0, "History must be an even number of messages"

        if audio:
            sr, audio_data = audio
            if audio_data.dtype == "int16":
                audio_data_float = audio_data.astype(np.float32) / 32768.0
                audio_data = torch.from_numpy(audio_data_float)
            elif audio_data.dtype == "int32":
                audio_data_float = audio_data.astype(np.float32) / 2147483648.0
                audio_data = torch.from_numpy(audio_data_float)
            else:
                audio_data = torch.from_numpy(audio_data)

            if sr != 16000:
                # resample
                print("Resampling audio from {} to 16000".format(sr))
                audio_data = torchaudio.transforms.Resample(sr, 16000)(audio_data)
            # print(audio_data)
            print(audio_data.shape)
            audio_values = self.audio_processor(
                audio=audio_data, sampling_rate=16000, return_tensors="pt"
            ).input_values
            audio_values = audio_values.to(dtype=torch.bfloat16, device="cuda")

        messages = []
        for i in range(0, len(history), 2):
            messages.append({"role": "user", "content": history[i]})
            messages.append({"role": "user", "content": history[i + 1]})

        messages.append({"role": "user", "content": input})
        print(messages)
        tokenized_chat = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).cuda()

        generation_kwargs = dict(
            inputs=tokenized_chat,
            audio_values=audio_values if audio else None,
            streamer=self.streamer,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.2,
            max_new_tokens=256,
        )

        # Run generation on separate thread to enable response streaming.
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        results = []
        first_token_time = None
        for new_text in self.streamer:
            yield new_text
            if not first_token_time:
                first_token_time = time.time()
            results.append(new_text)
        thread.join()

        ttft = time.time() - first_token_time
        total_time = time.time() - t0
        print(f"Output generated. TTFT: {ttft:.2f}s, Total: {total_time:.2f}s")


def main(input: str):
    model = GazelleModel()
    for val in model.generate(input):
        print(val, end="", flush=True)


if __name__ == "__main__":
    download_model()
    download_samples()

    gz = GazelleModel()

    def gen_(input, mic_audio, upload_audio):
        final_str = ""
        audio = None
        if mic_audio:
            audio = mic_audio
        elif upload_audio:
            audio = upload_audio
        if mic_audio and upload_audio:
            raise ValueError("Only one audio input is allowed")

        for result in gz.generate(input, audio):
            final_str += result
            yield final_str

    examples = [
        ["", None, os.path.join(os.path.dirname(__file__), "test6.wav")],
        ["", None, os.path.join(os.path.dirname(__file__), "test26.wav")],
        [
            "You are a professional with no available time slots for the rest of the week.",
            None,
            os.path.join(os.path.dirname(__file__), "testappt3.wav"),
        ],
        [
            "You are an expert diagnostic doctor.",
            None,
            os.path.join(os.path.dirname(__file__), "testdoc.wav"),
        ],
        [
            "Translate the previous statement to French.",
            None,
            os.path.join(os.path.dirname(__file__), "test6.wav"),
        ],
        [
            "Why would the Chinese government increase social spending?",
            None,
            os.path.join(os.path.dirname(__file__), "test21.wav"),
        ],
        [
            "What is Nvidia's new generation of chips called? When will they ship?",
            None,
            os.path.join(os.path.dirname(__file__), "testnvidia.wav"),
        ],
        [
            "Translate the previous statement to Chinese.",
            None,
            os.path.join(os.path.dirname(__file__), "testnvidia.wav"),
        ],
    ]

    gr_theme = gr.themes.Default(
        font=[gr.themes.GoogleFont("Space Grotesk"), "Arial", "sans-serif"]
    )

    interface = gr.Interface(
        fn=gen_,
        theme=gr_theme,
        inputs=[
            "textbox",
            gr.Audio(source="microphone"),
            gr.Audio(source="upload"),
        ],
        outputs="textbox",
        title="🦌 Gazelle v0.2",
        description="""Gazelle is a joint speech-language model by [Tincans](https://tincans.ai) 🥫 - for more details and prompt ideas, see our [v0.2 announcement](https://tincans.ai/slm3). This is an *early research preview* -- please temper expectations!
    Gazelle can take in text and audio as input (interchangeably) and generates text as output.
    You can further synthesize the text output into audio via a TTS provider (not implemented here). Some example tasks include transcribing audio, answering questions, or understanding spoken audio. This approach will be superior for business use cases where latency and conversational quality matter - such as customer support, outbound sales, and more.
    
    Known limitations exist! The model was only trained on English audio and is not expected to work well with other languages. Similarly, the model does not handle accents well yet. The gradio demo may have bugs with sample rate for audio. We also only accept a single audio input (microphone or upload).

    Feedback? [Twitter](https://twitter.com/hingeloss) | [email](hello@tincans.ai) | [GitHub](https://github.com/tincans-ai/gazelle)
    """,
        examples=examples,
    )

    interface.queue()
    interface.launch()