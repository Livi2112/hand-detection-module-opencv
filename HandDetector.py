import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector():
    def __init__(
        self,
        num_hands = 2, 
        min_hand_detection_confidence=0.5, 
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        # Set options
        base_options = python.BaseOptions(
            model_asset_path="hand_landmarker.task"
        )
        options = vision.HandLandmarkerOptions(
            base_options = base_options,
            num_hands = num_hands,
            min_hand_detection_confidence = min_hand_detection_confidence,
            min_hand_presence_confidence = min_hand_presence_confidence,
            min_tracking_confidence = min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def findHands(self, frame, draw=True):
        # Change image format for mediapipe
        imgRGB = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=imgRGB
        )

        # Run detector
        hands = self.detector.detect(mp_image)
        if draw:
            # Draw hand skeleton
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),         # Pointer
                (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Little
                (5, 9), (9, 13), (13, 17)               # Palm 
            ]
            for hand_landmarks in hands.hand_landmarks:
                # Draw points
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Draw lines
                for start, end in connections:
                    x1 = int(hand_landmarks[start].x * frame.shape[1])
                    y1 = int(hand_landmarks[start].y * frame.shape[0])

                    x2 = int(hand_landmarks[end].x * frame.shape[1])
                    y2 = int(hand_landmarks[end].y * frame.shape[0])

                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return hands
    

def main():
    handDetector = HandDetector()

    cTime = 0
    pTime = 0

    # Select video capture source
    cap = cv2.VideoCapture(0)

    # Main capture loop
    while True:
        success, frame = cap.read()

        # Get FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display fps
        cv2.putText(frame, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        if not success:
            break

        if cv2.waitKey(1) == 27:
            break;

        handDetector.findHands(frame)

        cv2.imshow('HAND_DETECTION', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()