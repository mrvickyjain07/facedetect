import cv2
import time
import os

dataset = "dataset"
name = "sample"
path = os.path.join(dataset, name)

if not os.path.exists(dataset):
    os.makedirs(dataset)
if not os.path.exists(path):
    os.makedirs(path)

alg = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + alg)


cam = cv2.VideoCapture(0)
time.sleep(1)

count = 1
while count<50:
    print(count)
    _, img = cam.read()
    text = "No person detected"
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray_img, 1.3, 4)
    
    for (x, y, w, h) in face:
        text2 = "Person Detected"
        print(text2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, text2, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        face_only = gray_img[y:y + h, x:x + w]
        resize_img = cv2.resize(face_only, (130, 100))
        cv2.imwrite(f"{path}/{count}.jpg", resize_img)
        count += 1
    
    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Face Detection", img)
    
    key = cv2.waitKey(10)
    if key == ord("q"): 
        break


cam.release()
cv2.destroyAllWindows()