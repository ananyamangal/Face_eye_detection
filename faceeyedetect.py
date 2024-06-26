import cv2
cap = cv2.VideoCapture(0)
cap.set(3, 640)  
cap.set(4, 480) 

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

while True:
    success, img = cap.read()
    if not success:
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    faces = faceCascade.detectMultiScale(imgGray, scaleFactor=1.3, minNeighbors=5)


    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cv2.putText(img, 'Face', (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

   
        roi_gray = imgGray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

 
        eyes = eyeCascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            cv2.putText(roi_color, 'Eye', (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


    cv2.imshow('face_detect', img)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()