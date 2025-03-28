# launch the offline engine
import asyncio
import io
from multiprocessing import freeze_support
import os
import pickle

from PIL import Image
import requests
import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge

if is_in_ci():
    import patch


def main():
    llm = sgl.Engine(model_path="/root/models/deepseek-v2-lite-text", trust_remote_code=True, disable_cuda_graph=True)

    prompts = [
        "Write a short, neutral self-introduction",
        "for a fictional character.",
        "Hello, my name is",
    ]

    sampling_params = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_new_tokens": 0,
    }

    print("\n=== Testing synchronous streaming generation with overlap removal ===\n")

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        merged_output = stream_and_merge(llm, prompt, sampling_params)
        print("Generated text:", merged_output)
        print()

if __name__ == "__main__":
    freeze_support()
    main()
