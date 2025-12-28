import json
from cryptography.fernet import Fernet
import os

class SecureLogger:
    def __init__(self, key_path, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Generate key if it doesn't exist
        if not os.path.exists(key_path):
            key = Fernet.generate_key()
            with open(key_path, "wb") as f:
                f.write(key)
        else:
            with open(key_path, "rb") as f:
                key = f.read()
        self.fernet = Fernet(key)

    def log(self, data: dict):
        try:
            json_str = json.dumps(data)
            encrypted = self.fernet.encrypt(json_str.encode())
            with open(self.log_file, "ab") as f:
                f.write(encrypted + b"\n")
        except Exception as e:
            print("[LOG ERROR]", e)
