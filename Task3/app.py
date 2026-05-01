import sys
import mediapipe as mp


NOSE_TIP = 1
LEFT_FACE = 234
RIGHT_FACE = 454


def classify_direction(image_path):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )

    image = mp.Image.create_from_file(image_path)

    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)

    if not result.face_landmarks:
        return "straight"

    landmarks = result.face_landmarks[0]

    nose = landmarks[NOSE_TIP]
    left_face = landmarks[LEFT_FACE]
    right_face = landmarks[RIGHT_FACE]

    face_center_x = (left_face.x + right_face.x) / 2

    difference = nose.x - face_center_x

    threshold = 0.03

    if difference > threshold:
        return "left"
    elif difference < -threshold:
        return "right"
    else:
        return "straight"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <input-image>")
        sys.exit(1)

    image_path = sys.argv[1]
    output = classify_direction(image_path)
    print(output)