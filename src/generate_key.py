from cryptography.fernet import Fernet
from pathlib import Path
import os

def generate_key():
    # ✅ always go to project root
    base_dir = Path(__file__).resolve().parents[1]   # <-- this finds: KF Directional Tracking System
    
    key_path = base_dir / "security" / "encryption" / "key.key"

    key_path.parent.mkdir(parents=True, exist_ok=True)

    if key_path.exists():
        print("Key already exists:", key_path)
        return

    key = Fernet.generate_key()
    key_path.write_bytes(key)
    print("✅ Key generated at:", key_path)

if __name__ == "__main__":
    generate_key()
