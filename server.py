###########server.py###############
import numpy as np
import sys, os
from fastapi import FastAPI, UploadFile, File
from starlette.requests import Request
import io
import cv2
import pytesseract
import re
from pydantic import BaseModel


def read_img(img):
    # Deploy
    pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'

    # Local
    # pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/4.1.1/bin/tesseract'

    # Window
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(img, lang="kor")
    return (text)


app = FastAPI()


class ImageType(BaseModel):
    url: str


@app.post("/predict/")
def prediction(request: Request, file: bytes = File(...)):
    if request.method == "POST":
        image_stream = io.BytesIO(file)
        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        label = read_img(frame)
        return label
    else:
        return "No post request found"
