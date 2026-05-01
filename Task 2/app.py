import sys
import mediapipe as mp


LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16


def classify_arm(image_path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
        running_mode=VisionRunningMode.IMAGE
    )

    image = mp.Image.create_from_file(image_path)

    with PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)

    if not result.pose_landmarks:
        return "None"

    landmarks = result.pose_landmarks[0]

    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]
    left_wrist = landmarks[LEFT_WRIST]
    right_wrist = landmarks[RIGHT_WRIST]

    left_arm_up = left_wrist.y < left_shoulder.y
    right_arm_up = right_wrist.y < right_shoulder.y

    if left_arm_up and right_arm_up:
        return "both"
    elif left_arm_up:
        return "left"
    elif right_arm_up:
        return "right"
    else:
        return "None"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <input-image>")
        sys.exit(1)

    image_path = sys.argv[1]
    output = classify_arm(image_path)
    print(output)