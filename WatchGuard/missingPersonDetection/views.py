from django.shortcuts import render, redirect
from django.http import HttpResponse
import os
import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
from django.conf import settings

def home(request):
    return render(request, 'index.html')

def mp(request):
    if request.method == 'POST':
        person_name = request.POST['person_name']
        return detect_missing_person(person_name)
    return render(request, 'missing_person.html')

def fr(request):
    if request.method == 'POST':
        return detect_faces()
    return render(request, 'face_recognition.html')

def contacts(request):
    return render(request, 'contact.html')

def abouts(request):
    return render(request, 'about.html')

def detect_missing_person(person_name):
    base_path = settings.BASE_DIR
    
    protoPath = os.path.join(base_path, "face_detection_model/deploy.prototxt")
    modelPath = os.path.join(base_path, "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    embedder = cv2.dnn.readNetFromTorch(os.path.join(base_path, "openface_nn4.small2.v1.t7"))

    recognizer_path = os.path.join(base_path, "output/recognizer.pickle")
    le_path = os.path.join(base_path, "output/le.pickle")

    if not os.path.isfile(recognizer_path):
        return HttpResponse(f"File not found: {recognizer_path}", status=404)
    if not os.path.isfile(le_path):
        return HttpResponse(f"File not found: {le_path}", status=404)

    recognizer = pickle.loads(open(recognizer_path, "rb").read())
    le = pickle.loads(open(le_path, "rb").read())

    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    fps = FPS().start()

    while True:
        frame = vs.read()
        if frame is None:
            break

        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                if name == person_name:
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        fps.update()

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    fps.stop()
    vs.stop()

    cv2.destroyAllWindows()

    return HttpResponse("Detection completed")

def detect_faces():
    base_path = settings.BASE_DIR
    
    protoPath = os.path.join(base_path, "face_detection_model/deploy.prototxt")
    modelPath = os.path.join(base_path, "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    embedder = cv2.dnn.readNetFromTorch(os.path.join(base_path, "openface_nn4.small2.v1.t7"))

    recognizer_path = os.path.join(base_path, "output/recognizer.pickle")
    le_path = os.path.join(base_path, "output/le.pickle")

    if not os.path.isfile(recognizer_path):
        return HttpResponse(f"File not found: {recognizer_path}", status=404)
    if not os.path.isfile(le_path):
        return HttpResponse(f"File not found: {le_path}", status=404)

    recognizer = pickle.loads(open(recognizer_path, "rb").read())
    le = pickle.loads(open(le_path, "rb").read())

    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    fps = FPS().start()

    while True:
        frame = vs.read()
        if frame is None:
            break

        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]

                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        fps.update()

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    fps.stop()
    vs.stop()

    cv2.destroyAllWindows()

    return HttpResponse("Face recognition completed")
