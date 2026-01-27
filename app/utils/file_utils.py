# app/utils/file_utils.py

import os
import uuid
from fastapi import UploadFile

UPLOAD_DIR = "static/uploads"


async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Saves uploaded file to disk with a UUID filename.
    Returns absolute path.
    """

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR, exist_ok=True)

    ext = os.path.splitext(upload_file.filename)[1]
    new_filename = f"{uuid.uuid4()}{ext}"

    file_path = os.path.join(UPLOAD_DIR, new_filename)

    with open(file_path, "wb") as f:
        f.write(await upload_file.read())

    return file_path
