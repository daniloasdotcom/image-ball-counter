import os
import json
import numpy as np
import cv2
from ultralytics import YOLO
from google.oauth2 import service_account
from googleapiclient.discovery import build

FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID")

# modelo leve, ideal para CI/CD
model = YOLO("yolov8n.pt")

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
    data = drive.files().get_media(fileId=file_id).execute()
    img_array = np.frombuffer(data, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def count_balls(img):
    results = model(img)[0]
    count = 0

    for b in results.boxes:
        cls = int(b.cls[0])
        label = model.names[cls].lower()
        if "ball" in label or label in ["sports ball", "tennis ball"]:
            count += 1

    return count

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
