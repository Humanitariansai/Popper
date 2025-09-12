
import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class GroqAgent:
    def __init__(self, model="llama-3.3-70b-versatile"):
        # Groq models: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it
        self.model = model
        self.api_key = GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")

    def ask(self, prompt, system_prompt=None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            try:
                err = response.json()
                return f"[Groq API Error {response.status_code}: {err.get('error', {}).get('message', str(err))}]"
            except Exception:
                return f"[Groq API Error {response.status_code}: {response.text}]"
        try:
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Groq API Response Error: {e}]"
