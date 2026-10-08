from ultralytics import YOLO
import cv2

camera = cv2.VideoCapture("rose.mp4")
model = YOLO("runs/classify/train/weights/best.pt")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # prediction
    result = model(frame, verbose=False)
    data = result[0]
    label = data.names[data.probs.top1]
    conf = data.probs.top1conf.item()

    cv2.putText(frame, f"{label} ({conf:.2%})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("YOLOv8 Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()