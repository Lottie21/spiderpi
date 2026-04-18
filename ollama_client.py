import json
import urllib.request


class OllamaClient:
    def __init__(self, host: str, model: str, timeout: int = 60):
        # host example: "192.168.1.113"
        # IMPORTANT: no extra braces here
        self.base = f"http://{host}:11434"
        self.model = model
        self.timeout = timeout

    def chat(self, user_text: str) -> str:
        url = f"{self.base}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Reply briefly"},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
        }

        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return data.get("message", {}).get("content", "")


def ollama_generate(prompt: str, host: str, model: str, timeout_s: int = 60) -> str:
    """
    Backwards-compatible helper for existing scripts that call:
        from ollama_client import ollama_generate
    """
    client = OllamaClient(host=host, model=model, timeout=timeout_s)
    return client.chat(prompt)