import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = mp.tasks.BaseOptions
pose_landmarker = mp.tasks.vision.PoseLandmarker
pose_landmarker_options = mp.tasks.vision.PoseLandmarkerOptions
vision_running_mode = mp.tasks.vision.RunningMode

model_path = "pose_landmarker_full.task"
image_path = "pose.jpg"
output_path = "pose_output.jpg"


def draw_pose_landmarks(image, detection_result):
    annotated_image = image.copy()

    pose_landmarks_list = detection_result.pose_landmarks

    image_height, image_width, _ = annotated_image.shape

    pose_connections = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12),
        (11, 13), (13, 15),
        (15, 17), (15, 19), (15, 21),
        (17, 19),
        (12, 14), (14, 16),
        (16, 18), (16, 20), (16, 22),
        (18, 20),
        (11, 23), (12, 24),
        (23, 24),
        (23, 25), (25, 27),
        (27, 29), (27, 31),
        (29, 31),
        (24, 26), (26, 28),
        (28, 30), (28, 32),
        (30, 32)
    ]

    for pose_landmarks in pose_landmarks_list:
        points = []

        for landmark in pose_landmarks:
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

        for start_index, end_index in pose_connections:
            if start_index < len(points) and end_index < len(points):
                cv2.line(
                    annotated_image,
                    points[start_index],
                    points[end_index],
                    (255, 0, 0),
                    2
                )

    return annotated_image


def main():
    options = pose_landmarker_options(
        base_options=base_options(model_asset_path=model_path),
        running_mode=vision_running_mode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )

    with pose_landmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(image_path)

        detection_result = landmarker.detect(mp_image)

        image = cv2.imread(image_path)

        if image is None:
            print("Error: Could not read the input image.")
            return

        annotated_image = draw_pose_landmarks(image, detection_result)

        cv2.imwrite(output_path, annotated_image)

        pose_count = len(detection_result.pose_landmarks)

        print(f"Detected poses: {pose_count}")
        print(f"Output image saved as: {output_path}")

        for pose_index, pose_landmarks in enumerate(detection_result.pose_landmarks):
            print(f"\nPose {pose_index + 1} landmarks:")

            for landmark_index, landmark in enumerate(pose_landmarks):
                print(
                    f"Landmark {landmark_index}: "
                    f"x={landmark.x:.4f}, "
                    f"y={landmark.y:.4f}, "
                    f"z={landmark.z:.4f}, "
                    f"visibility={landmark.visibility:.4f}"
                )


if __name__ == "__main__":
    main()