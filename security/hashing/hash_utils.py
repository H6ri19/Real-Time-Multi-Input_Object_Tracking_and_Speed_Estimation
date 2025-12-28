import hashlib
def sha256_file(path):
    with open(path,'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()
