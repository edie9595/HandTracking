import math
from pythonosc import udp_client
import cv2
import mediapipe as mp
import numpy as np

def lerp(value, start, end):
    return (value - start) / (end - start) if (end - start) != 0 else 0.0

def distance3D(p1, p2):
    v1 = np.array([p1.x, p1.y, p1.z])
    v2 = np.array([p2.x, p2.y, p2.z])
    return np.linalg.norm(v1 - v2)

def angle3D(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle_rad = math.acos(dot_product / (norm_v1 * norm_v2))
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def calculate_fingers(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    hand_size = distance3D(wrist, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC])

    thumb_distance = distance3D(wrist, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP])
    index_finger_distance = distance3D(wrist, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])
    middle_finger_distance = distance3D(wrist, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP])
    ring_finger_distance = distance3D(wrist, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP])
    pinky_distance = distance3D(wrist, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP])

    normalized_thumb_distance = thumb_distance / hand_size
    normalized_index_finger_distance = index_finger_distance / hand_size
    normalized_middle_finger_distance = middle_finger_distance / hand_size
    normalized_ring_finger_distance = ring_finger_distance / hand_size
    normalized_pinky_distance = pinky_distance / hand_size

    openness = [
        normalized_thumb_distance,
        normalized_index_finger_distance,
        normalized_middle_finger_distance,
        normalized_ring_finger_distance,
        normalized_pinky_distance
    ]

    return openness

def sendOSCdata(client, hand, fingerdata):

    """leftThumb (float)
        leftIndex (float)
        leftMiddle (float)
        leftRing (float)
        leftPinky (float)
        rightThumb (float)
        rightIndex (float)
        rightMiddle (float)
        rightRing (float)
        rightPinky (float)
        leftThumbSpread (float)
        leftIndexSpread (float)
        leftMiddleSpread (float)
        leftRingSpread (float)
        leftPinkySpread (float)
        rightThumbSpread (float)
        rightIndexSpread (float)
        rightMiddleSpread (float)
        rightRingSpread (float)
        rightPinkySpread (float)"""

    handstr = "left" if hand == 1 else "right"

    client.send_message(f"/avatar/parameters/{handstr}Thumb", float(fingerdata[0][2]))
    client.send_message(f"/avatar/parameters/{handstr}Index", float(fingerdata[1][2]))
    client.send_message(f"/avatar/parameters/{handstr}Middle", float(fingerdata[2][2]))
    client.send_message(f"/avatar/parameters/{handstr}Ring", float(fingerdata[3][2]))
    client.send_message(f"/avatar/parameters/{handstr}Pinky", float(fingerdata[4][2]))


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

fingers = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]

oscClient = udp_client.SimpleUDPClient("127.0.0.1", 9001)

cap = cv2.VideoCapture(1)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for idx, hand_handedness in enumerate(results.multi_handedness):

                hand_landmarks = results.multi_hand_landmarks[idx]

                hand = 1 if hand_handedness.classification[0].label == "Left" else 0

                openness = calculate_fingers(hand_landmarks)
                for x in range(5):
                    if fingers[hand][x][0] == 0:
                        fingers[hand][x][0] = openness[x]

                    fingers[hand][x][0] = min(openness[x], fingers[hand][x][0])
                    fingers[hand][x][1] = max(openness[x], fingers[hand][x][1])
                    fingers[hand][x][2] = lerp(openness[x], fingers[hand][x][0], fingers[hand][x][1])

                sendOSCdata(oscClient, hand, fingers[hand])

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
