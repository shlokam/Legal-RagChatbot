import os

# Simple manual .env loader
def load_env(path=".env"):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            os.environ[key] = value

def get_api_key():
    return os.environ.get("groq_api_key")
