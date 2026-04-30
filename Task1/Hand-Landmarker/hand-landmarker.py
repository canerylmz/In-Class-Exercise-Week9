import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = mp.tasks.BaseOptions
hand_landmarker = mp.tasks.vision.HandLandmarker
hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
vision_running_mode = mp.tasks.vision.RunningMode

model_path = "hand_landmarker.task"
image_path = "image.jpg"
output_path = "hand_output.jpg"


def draw_hand_landmarks(image, detection_result):
    annotated_image = image.copy()

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness

    image_height, image_width, _ = annotated_image.shape

    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    for index, hand_landmarks in enumerate(hand_landmarks_list):
        points = []

        for landmark in hand_landmarks:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            points.append((x, y))

            cv2.circle(
                annotated_image,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        for start_index, end_index in hand_connections:
            cv2.line(
                annotated_image,
                points[start_index],
                points[end_index],
                (255, 0, 0),
                2
            )

        if handedness_list:
            handedness = handedness_list[index][0]
            label = handedness.category_name
            score = handedness.score

            wrist_x, wrist_y = points[0]

            cv2.putText(
                annotated_image,
                f"{label}: {score:.2f}",
                (wrist_x, wrist_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

    return annotated_image


def main():
    options = hand_landmarker_options(
        base_options=base_options(model_asset_path=model_path),
        running_mode=vision_running_mode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    with hand_landmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(image_path)

        detection_result = landmarker.detect(mp_image)

        image = cv2.imread(image_path)

        if image is None:
            print("Error: Could not read the input image.")
            return

        annotated_image = draw_hand_landmarks(image, detection_result)

        cv2.imwrite(output_path, annotated_image)

        hand_count = len(detection_result.hand_landmarks)

        print(f"Detected hands: {hand_count}")
        print(f"Output image saved as: {output_path}")

        for index, handedness in enumerate(detection_result.handedness):
            category = handedness[0]
            print(
                f"Hand {index + 1}: "
                f"{category.category_name}, "
                f"confidence: {category.score:.2f}"
            )

        for hand_index, hand_landmarks in enumerate(detection_result.hand_landmarks):
            print(f"\nHand {hand_index + 1} landmarks:")

            for landmark_index, landmark in enumerate(hand_landmarks):
                print(
                    f"Landmark {landmark_index}: "
                    f"x={landmark.x:.4f}, "
                    f"y={landmark.y:.4f}, "
                    f"z={landmark.z:.4f}"
                )


if __name__ == "__main__":
    main()