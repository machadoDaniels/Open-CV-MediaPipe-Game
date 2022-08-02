import cv2
import mediapipe as mp
import numpy as np
from time import sleep


def motion_detection(k, km, ld):
    for data_point in ld:
        km.append({
            'X': data_point.x,
            'Y': data_point.y,
            'Z': data_point.z,
        })

    if len(k) == 0:
        for data_point in ld:
            k.append({
                'X': data_point.x,
                'Y': data_point.y,
                'Z': data_point.z,
            })
    else:
        for c in range(0, len(k)):
            if k[c]['X'] - 0.08 < km[c]['X'] < k[c]['X'] + 0.08 and \
                    k[c]['Y'] - 0.1 < km[c]['Y'] < k[c]['Y'] + 0.1 and \
                    k[c]['Z'] - 0.4 < km[c]['Z'] < k[c]['Z'] + 0.4:
                k[c]['X'] = km[c]['X']
                k[c]['Y'] = km[c]['Y']
                k[c]['Z'] = km[c]['Z']
                return False
            else:
                return True


mp_drawing1 = mp.solutions.drawing_utils
mp_pose1 = mp.solutions.pose
mp_drawing2 = mp.solutions.drawing_utils
mp_pose2 = mp.solutions.pose

# VIDEO FEED
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# Setup mediapipe instance
keypoints1 = []
keypoints2 = []

with mp_pose1.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.2) as pose1, \
        mp_pose2.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.2) as pose2:
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Recolor image to RGB
        image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        image1.flags.writeable = False

        image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        image2.flags.writeable = False

        results1 = pose1.process(image1)
        results2 = pose2.process(image2)

        # Recolor back to BGR
        image1.flags.writeable = True
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        image2.flags.writeable = True
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        landmarks1 = results2.pose_landmarks.landmark
        landmarks2 = results2.pose_landmarks.landmark

        # Render detections
        mp_drawing1.draw_landmarks(image1, results1.pose_landmarks, mp_pose1.POSE_CONNECTIONS,
                                   mp_drawing1.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                   mp_drawing1.DrawingSpec(color=(66, 245, 66), thickness=2, circle_radius=2))

        mp_drawing2.draw_landmarks(image2, results2.pose_landmarks, mp_pose2.POSE_CONNECTIONS,
                                   mp_drawing2.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                   mp_drawing2.DrawingSpec(color=(66, 245, 66), thickness=2, circle_radius=2))

        # Motion detection
        keypoints_movel1 = []
        keypoints_movel2 = []

        if motion_detection(keypoints1, keypoints_movel1, landmarks1) is True:
             print('O JOGADOR 1 PERDEU!!')

        if motion_detection(keypoints2, keypoints_movel2, landmarks2) is True:
            print('O JOGADOR 2 PERDEU!!')


        sleep(0.5)
        cv2.imshow('Mediapipe Feed1', image1)
        cv2.imshow('Mediapipe Feed2', image2)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap1.release()
cap2.release()
cv2.destroyAllWindows()
