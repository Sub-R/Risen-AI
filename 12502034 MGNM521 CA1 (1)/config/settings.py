import os
from dotenv import load_dotenv, set_key

# Load environment variables from .env file
load_dotenv()

class Config:
    ENV_PATH = ".env"

    @staticmethod
    def get_openrouter_key():
        return os.getenv("OPENROUTER_API_KEY", "")

    @staticmethod
    def get_huggingface_key():
        return os.getenv("HUGGINGFACE_API_KEY", "")

    @staticmethod
    def update_key(key_name, value):
        # Dynamically updates the .env file from the UI Settings panel
        if not os.path.exists(Config.ENV_PATH):
            open(Config.ENV_PATH, 'w').close()
        set_key(Config.ENV_PATH, key_name, value)
        os.environ[key_name] = value