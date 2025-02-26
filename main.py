import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
           
            left_eye = [face_landmarks.landmark[i] for i in [33, 133]]
            right_eye = [face_landmarks.landmark[i] for i in [362, 263]]

            
            left_eye_x = np.mean([p.x for p in left_eye])
            right_eye_x = np.mean([p.x for p in right_eye])

            
            if left_eye_x > 0.6 and right_eye_x > 0.6:
                text = "Looking Right"
            elif left_eye_x < 0.4 and right_eye_x < 0.4:
                text = "Looking Left"
            else:
                text = "Looking Forward"

            
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            
            if text != "Looking Forward":
                cv2.putText(frame, "WARNING: Stay Focused!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    
    cv2.imshow("Eye Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
