from ultralytics import YOLO
import cv2

# Load models
person_model = YOLO("yolov8n.pt")   # person detection
id_model = YOLO("best.pt")          # ID card detection (your trained model)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run models
    person_results = person_model(frame)
    id_results = id_model(frame)

    person_detected = False
    id_detected = False

    # Check person
    for r in person_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = person_model.names[cls]
            if label == "person":
                person_detected = True

    # Check ID card
    for r in id_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = id_model.names[cls]
            if label == "id_card":
                id_detected = True

    # FINAL LOGIC
    if person_detected and id_detected:
        print("Wearing ID Card ")
    elif person_detected and not id_detected:
        print("Not Wearing ID Card ")

    # Show camera
    frame = person_results[0].plot()
    cv2.imshow("ID Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()