import cv2
import mediapipe as mp
import random
import time

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Finger tip landmark indices
FINGER_TIPS = [4, 8, 12, 16, 20]

# Count fingers for one hand
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
            if tip_x > pip_x:  # right hand (basic check)
                count += 1
    return count

# Generate random math question
def generate_question(max_num):
    while True:
        a, b = random.randint(1, max_num), random.randint(1, max_num)
        if a + b <= 10:   # ✅ only allow answers you can show with fingers
            return f"{a} + {b}", a + b


# Game loop
cap = cv2.VideoCapture(0)

difficulty = None
max_num = 5
question, answer = None, None
last_check = time.time()
feedback = ""
score = 0

print("✋ Show 1, 2, or 3 fingers to select difficulty")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape
    fingers_total = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers_total += count_fingers(hand_landmarks, h, w)

        # Step 1: Difficulty Selection
        if difficulty is None:
            cv2.putText(frame, "Select Difficulty:", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "1 Finger = Easy (1-5)", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "2 Fingers = Medium (1-10)", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, "3 Fingers = Hard (1-20)", (50, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            if fingers_total == 1:
                difficulty, max_num = "Easy", 5
            elif fingers_total == 2:
                difficulty, max_num = "Medium", 10
            elif fingers_total == 3:
                difficulty, max_num = "Hard", 20

            if difficulty:
                question, answer = generate_question(max_num)
                feedback = f"🎮 {difficulty} Mode Started!"
                last_check = time.time()

        # Step 2: Play Game
        else:
            cv2.putText(frame, f"Q: {question}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Your Answer: {fingers_total}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 200), 2)
            cv2.putText(frame, f"Score: {score}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            if time.time() - last_check > 2:  # check every 2 sec
                if fingers_total == answer:
                    feedback = "✅ Correct!"
                    score += 1
                    question, answer = generate_question(max_num)
                elif fingers_total != 0:
                    feedback = f"❌ Wrong ({fingers_total})"
                last_check = time.time()

            cv2.putText(frame, feedback, (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if "Correct" in feedback else (0, 0, 255), 2)

    cv2.imshow("Math with Fingers - Education Game", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
