from ultralytics import YOLO
import os

model = YOLO("best.pt")

def predict_image(image):
    # Convert PIL → save temp file (no cv2 needed)
    temp_path = "temp.jpg"
    image.save(temp_path)

    results = model.predict(temp_path, verbose=False)

    for r in results:
        probs = r.probs.data.tolist()
        names = r.names

    os.remove(temp_path)

    top3 = sorted(zip(names.values(), probs), key=lambda x: x[1], reverse=True)[:3]

    return top3
