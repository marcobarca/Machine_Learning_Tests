import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm


def main():
    p_time = 0
    c_time = 0

    video_path = "./bass.mp4"
    detector = htm.HandDetector()
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        lm_list = detector.findPosition(img)
        if len(lm_list) != 0:
            print(lm_list[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == 27:  # Press 'Esc' key to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
