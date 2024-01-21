import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import subprocess

def set_volume(volume_level):
    # Utilizza osascript per eseguire uno script AppleScript che imposta il volume
    script = f'set volume output volume {volume_level}'
    subprocess.run(['osascript', '-e', script])



cap = cv2.VideoCapture(0)
pTime = 0

detector = htm.HandDetector(detection_con=0.7)
vol = 0
vol_bar = 400

while True:
    success, img = cap.read()
    img = detector.find_hands(img, draw=True)
    lm_list = detector.find_position(img, draw=False)
    if len(lm_list) != 0:
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        
        # calculating the center of the line 
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10 , (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10 , (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10 , (0, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2 - x1, y2 - y1)

        vol = np.interp(length, [50, 300], [0, 100])
        vol_bar = np.interp(length, [50, 300], [400, 150])
        

        set_volume(vol)

        if length < 50:
            cv2.circle(img, (cx, cy), 10 , (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), 3)
    cv2.putText(img, f'{int(vol)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:  # Press 'Esc' key to exit
        break