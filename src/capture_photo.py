import os
import uuid
import time
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
import cv2

app = FastAPI()

@app.get("/{photo_id}")
async def read_photo(photo_id: str):
    # Check if the ID is valid
    if photo_id is None:
        return {"message": "Error: No ID specified"}, 400

    # Check if the photo exists
    photo_path = os.path.join("photos", f"{photo_id}.png")
    if not os.path.exists(photo_path):
        return {"message": "Error: Photo not found"}, 404
    return FileResponse(photo_path)

@app.get("/")
async def save():
    # Capture a photo using the default camera
    cap = cv2.VideoCapture(0)
    time.sleep(3)
    ret, frame = cap.read()
    cap.release()

    # Generate a unique identifier for the photo
    photo_id = str(uuid.uuid4())

    # Save the photo to a file in the photos directory
    cv2.imwrite(os.path.join('photos', f'{photo_id}.png'), frame)

    # Return the photo ID as plain text
    return {"photo_id": photo_id}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)