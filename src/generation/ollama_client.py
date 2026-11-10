"""
Minimal client for a local Ollama server (https://ollama.com).
Uses only the standard library, so no extra dependency is needed.

Setup on the machine running the project:
    ollama pull llama3.2:3b      # or qwen2.5:3b / qwen2.5:1.5b / phi3:mini
    ollama serve                 # usually already running after install
"""

import json
import urllib.error
import urllib.request
from typing import Dict, List

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        temperature: float = 0.1,
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def _post(self, path: str, payload: Dict) -> Dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            if e.code == 404 and "not found" in body.lower():
                raise OllamaError(
                    f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"
                ) from e
            raise OllamaError(f"Ollama returned HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}. Is it running? (ollama serve)"
            ) from e

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """messages: [{'role': 'system'|'user'|'assistant', 'content': str}, ...]"""
        data = self._post("/api/chat", {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        })
        return data.get("message", {}).get("content", "").strip()

    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
            names = {m.get("name", "") for m in tags.get("models", [])}
            return self.model in names or f"{self.model}:latest" in names
        except Exception:
            return False
