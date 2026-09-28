import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel

from .prompt_builder import build_prompt


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# For local testing, set this environment variable
# to the location of qwen_lora_output.
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

ADAPTER_PATH = os.getenv(
    "QLORA_ADAPTER_PATH",
    os.path.join(BASE_DIR, "models", "qwen_lora_output")
)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH
    )

    return model, tokenizer


model = None
tokenizer = None


def initialize_model():
    global model, tokenizer

    if model is None or tokenizer is None:
        print("Loading QLoRA model...")
        model, tokenizer = load_model()
        print("QLoRA model loaded.")

    return model, tokenizer


def generate_narrative(insights: dict) -> str:

    model, tokenizer = initialize_model()

    prompt = build_prompt(insights)

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()