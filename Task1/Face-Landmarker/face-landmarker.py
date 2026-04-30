import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "face_landmarker.task"
image_path = "image.png"
output_path = "output.jpg"

def draw_landmarks_on_image(image, detection_result):
    annotated_image = image.copy()

    face_landmarks_list = detection_result.face_landmarks

    image_height, image_width, _ = annotated_image.shape

    for face_landmarks in face_landmarks_list:
        for landmark in face_landmarks:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)

            cv2.circle(
                annotated_image,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

    return annotated_image

def main():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(image_path)

        face_landmarker_result = landmarker.detect(mp_image)

        image = cv2.imread(image_path)

        if image is None:
            print("Error: Could not read the input image.")
            return

        annotated_image = draw_landmarks_on_image(image, face_landmarker_result)

        cv2.imwrite(output_path, annotated_image)

        face_count = len(face_landmarker_result.face_landmarks)
        print(f"Detected faces: {face_count}")
        print(f"Output image saved as: {output_path}")

        if face_landmarker_result.face_blendshapes:
            print("\nTop face blendshapes:")

            blendshapes = face_landmarker_result.face_blendshapes[0]
            sorted_blendshapes = sorted(
                blendshapes,
                key=lambda item: item.score,
                reverse=True
            )

            for blendshape in sorted_blendshapes[:10]:
                print(f"{blendshape.category_name}: {blendshape.score:.4f}")


if __name__ == "__main__":
    main()