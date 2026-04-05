from ultralytics import YOLO

# Load model (only once)
model = YOLO("best.pt")

def predict_image(image):
    results = model.predict(image, verbose=False)

    for r in results:
        probs = r.probs.data.tolist()
        names = r.names

    # Get top 3 predictions
    top3 = sorted(zip(names.values(), probs), key=lambda x: x[1], reverse=True)[:3]

    return top3