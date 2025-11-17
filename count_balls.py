# ativar cron
import os
import json
import base64
import numpy as np
import cv2

from google.oauth2 import service_account
from googleapiclient.discovery import build

FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID")

def authenticate():
    service_account_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_KEY"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=credentials)

def get_latest_image(drive):
    query = f"'{FOLDER_ID}' in parents and mimeType contains 'image/' and trashed = false"

    results = drive.files().list(
        q=query,
        orderBy="modifiedTime desc",
        pageSize=1,
        fields="files(id, name)"
    ).execute()

    files = results.get("files", [])
    if not files:
        return None, None
    return files[0]["id"], files[0]["name"]

def download_image(drive, file_id):
    request = drive.files().get_media(fileId=file_id)
    data = request.execute()
    img_array = np.frombuffer(data, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def count_balls(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )
    if circles is None:
        return 0
    return len(circles[0])

def main():
    drive = authenticate()

    file_id, filename = get_latest_image(drive)
    if not file_id:
        print("Nenhuma imagem encontrada.")
        return

    img = download_image(drive, file_id)
    count = count_balls(img)

    print("Imagem:", filename)
    print("Bolas detectadas:", count)

if __name__ == "__main__":
    main()
