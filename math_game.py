import cv2
import mediapipe as mp
import random
import time

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Finger tip landmark indices
FINGER_TIPS = [4, 8, 12, 16, 20]

# Generate random math question
def generate_question():
    a, b = random.randint(1, 5), random.randint(1, 5)
    return f"{a} + {b}", a + b

# Count fingers (only one hand)
def count_fingers(hand_landmarks, img_height, img_width):
    count = 0
    for tip in FINGER_TIPS:
        tip_y = hand_landmarks.landmark[tip].y * img_height
        pip_y = hand_landmarks.landmark[tip - 2].y * img_height
        if tip != 4:  # all fingers except thumb
            if tip_y < pip_y:
                count += 1
        else:  # Thumb logic
            tip_x = hand_landmarks.landmark[tip].x * img_width
            pip_x = hand_landmarks.landmark[tip - 2].x * img_width
            if tip_x > pip_x:  # right hand
                count += 1
    return count

# Game loop
cap = cv2.VideoCapture(0)
question, answer = generate_question()
last_check = time.time()
feedback = ""

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]  # Only first hand
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        h, w, _ = frame.shape
        fingers = count_fingers(hand_landmarks, h, w)

        if time.time() - last_check > 2:
            if fingers == answer:
                feedback = "✅ Correct!"
                question, answer = generate_question()
            else:
                feedback = f"❌ Try again ({fingers})"
            last_check = time.time()

    # Show question and feedback
    cv2.putText(frame, f"Question: {question}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, feedback, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "Correct" in feedback else (0, 0, 255), 2)

    cv2.imshow("Math with Fingers (One Hand)", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
