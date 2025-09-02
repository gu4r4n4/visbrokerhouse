import os, shutil, hashlib, time
from typing import Tuple

def save_temp_upload(content: bytes, filename: str) -> str:
    name = f"{int(time.time())}_{hashlib.md5(filename.encode()).hexdigest()}_{filename}"
    path = os.path.join("/tmp", name)
    with open(path, "wb") as f:
        f.write(content)
    return path
