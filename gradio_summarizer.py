import json
from typing import Dict

import gradio as gr
import requests


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/auto"


def is_text_too_short(text: str) -> bool:
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < 30:
        return True
    if len(stripped.split()) < 6:
        return True
    return False


def build_payload(text: str) -> Dict:
    return {
        "model": DEFAULT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text concisely in Persian.",
            },
            {
                "role": "user",
                "content": f"لطفاً متن زیر را به‌صورت کوتاه و روان خلاصه کن. فقط خود خلاصه را برگردان:\n\n{text}",
            },
        ],
        "temperature": 0.3,
        "max_tokens": 256,
    }


def call_openrouter(api_key: str, text: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = build_payload(text)

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=(10, 60),
        )
    except requests.exceptions.RequestException as exc:  # network or timeout
        raise gr.Error(f"خطای شبکه یا زمان‌سنجی درخواست: {exc}")

    if not response.ok:
        try:
            data = response.json()
            error_message = (
                (data.get("error") or {}).get("message")
                or data.get("message")
                or response.text
            )
        except Exception:
            error_message = response.text
        raise gr.Error(f"خطا از سرور خلاصه‌ساز: {error_message}")

    try:
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise gr.Error("پاسخی از مدل دریافت نشد.")
        content = (choices[0].get("message") or {}).get("content")
        if not content:
            raise gr.Error("خلاصه‌ای در پاسخ یافت نشد.")
        return content.strip()
    except (ValueError, KeyError) as exc:
        raise gr.Error(f"ساختار پاسخ نامعتبر بود: {exc}")


def summarize(text: str, api_key: str) -> str:
    if not api_key or not api_key.strip():
        raise gr.Error("کلید API خالی است. لطفاً کلید معتبر وارد کنید.")
    if is_text_too_short(text):
        raise gr.Error("متن برای خلاصه‌سازی خیلی کوتاه است. لطفاً متن بلندتری وارد کنید.")

    return call_openrouter(api_key.strip(), text)


with gr.Blocks(title="خلاصه‌ساز متنی محلی") as demo:
    gr.Markdown("## خلاصه‌ساز متن\nمتن را وارد کنید، کلید API را بزنید و روی «خلاصه کن» کلیک کنید.")

    with gr.Row():
        input_text = gr.Textbox(
            label="متن ورودی",
            placeholder="اینجا متن خود را وارد کنید...",
            lines=10,
        )

    with gr.Row():
        api_key_tb = gr.Textbox(
            label="API Key",
            placeholder="کلید OpenRouter یا OpenAI (به‌صورت سازگار با OpenAI)",
            type="password",
        )

    with gr.Row():
        summarize_btn = gr.Button("خلاصه کن", variant="primary")

    output_box = gr.Textbox(label="خلاصه", lines=10)

    summarize_btn.click(fn=summarize, inputs=[input_text, api_key_tb], outputs=output_box)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
