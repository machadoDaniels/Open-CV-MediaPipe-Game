import cv2
import mediapipe as mp
import numpy as np
from time import sleep

mp_drawing1 = mp.solutions.drawing_utils
mp_pose1 = mp.solutions.pose
mp_drawing2 = mp.solutions.drawing_utils
mp_pose2 = mp.solutions.pose

# VIDEO FEED
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)



# Setup mediapipe instance
results = 0
keypoints = []

with mp_pose1.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.2) as pose1, mp_pose2.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.2) as pose2:
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Recolor image to RGB
        image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        image1.flags.writeable = False

        image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        image2.flags.writeable = False

        # Make detection

        # if results == 0:
        #     results = pose.process(image)
        # elif results != pose.process(image):
        #     break
        # else:

        results1 = pose1.process(image1)
        results2 = pose2.process(image2)

        # Recolor back to BGR
        image1.flags.writeable = True
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        image2.flags.writeable = True
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results2.pose_landmarks.landmark
            #print(landmarks[0])
        except:
            pass

        # for lndnmrk in mp_pose1.PoseLandmark:
        #     print(lndnmrk)

        pos_nariz = landmarks[mp_pose1.PoseLandmark.NOSE]

        # if cont == 0:
        #     x_nariz = pos_nariz.x
        # else:
        #     if float(x_nariz) - 0.2 > float(pos_nariz.x) or float(pos_nariz) > float(x_nariz + 0.2):
        #         break

        # print('---------')
        # print(pos_nariz.x)
        # print(x_nariz)
        # print('---------')
        #


        # Render detections
        mp_drawing1.draw_landmarks(image1, results1.pose_landmarks, mp_pose1.POSE_CONNECTIONS,
                                  mp_drawing1.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing1.DrawingSpec(color=(66, 245, 66), thickness=2, circle_radius=2))

        mp_drawing2.draw_landmarks(image2, results2.pose_landmarks, mp_pose2.POSE_CONNECTIONS,
                                  mp_drawing2.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing2.DrawingSpec(color=(66, 245, 66), thickness=2, circle_radius=2))

        cont = 0
        keypoints_movel = []
        sair = False
        for data_point in landmarks:
            keypoints_movel.append({
                'X': data_point.x,
                'Y': data_point.y,
                'Z': data_point.z,
            })

        if len(keypoints) == 0:
            for data_point in landmarks:
                    keypoints.append({
                        'X': data_point.x,
                        'Y': data_point.y,
                        'Z': data_point.z,
                        })
        else:
            for c in range(0, len(keypoints)):
                if keypoints[c]['X'] - 0.08 < keypoints_movel[c]['X'] < keypoints[c]['X'] + 0.08 and keypoints[c]['Y'] - 0.1 < keypoints_movel[c]['Y'] < keypoints[c]['Y'] + 0.1 and keypoints[c]['Z'] - 0.4 < keypoints_movel[c]['Z'] < keypoints[c]['Z'] + 0.4:
                    keypoints[c]['X'] = keypoints_movel[c]['X']
                    keypoints[c]['Y'] = keypoints_movel[c]['Y']
                    keypoints[c]['Z'] = keypoints_movel[c]['Z']
                else:
                    sair = True

        # for c in range(0, len(keypoints)):
        #     if keypoints[c]['X'] != keypoints_movel[c]['X']:
        #         sair = True


        if sair == True:
            break

            # print('-----------')
            # print(keypoints[0]['X'])
            #
            # print(pos_nariz.x)
            #
            # print('-----------')
            # if float(pos_nariz.x) != float(keypoints[0]['X']):
            #     break
            #
            #
            # elif keypoints[cont]['X'] - 0.1 < data_point.x < keypoints[cont]['X'] + 0.1 and keypoints[cont]['Y'] - 0.1 \
            #         < data_point.y < keypoints[cont]['Y'] + 0.1 and keypoints[cont]['Z'] - 0.1 < data_point.z \
            #         < keypoints[cont]['Z'] + 0.1:
            #
            #     keypoints[cont]['X'] = data_point.x
            #     keypoints[cont]['Y'] = data_point.y
            #     keypoints[cont]['Z'] = data_point.z
            #
            #
            #
            # cont += 1
        print('--------------')
        print('keypoint: \t\t', keypoints[0]['X'])
        print('keypoints movel: ', keypoints_movel[0]['X'])
        print('--------------')


        sleep(0.3)
        cv2.imshow('Mediapipe Feed1', image1)
        cv2.imshow('Mediapipe Feqed2', image2)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        cont += 1


cap1.release()
cap2.release()
cv2.destroyAllWindows()
