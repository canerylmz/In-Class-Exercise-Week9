import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'efficientdet_lite0.tflite'
image_path = 'image.jpg'
output_path = 'result.jpg'

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    max_results=5,
    running_mode=VisionRunningMode.IMAGE)

with ObjectDetector.create_from_options(options) as detector:
    mp_image = mp.Image.create_from_file(image_path)

    detection_result = detector.detect(mp_image)

    image = cv2.imread(image_path)

    for detection in detection_result.detections:
        bbox = detection.bounding_box

        x = bbox.origin_x
        y = bbox.origin_y
        w = bbox.width
        h = bbox.height

        category = detection.categories[0]
        label = category.category_name
        score = category.score

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text = f"{label}: {score:.2f}"
        cv2.putText(
            image,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        print(f"Object: {label}, Confidence: {score:.2f}, Location: x={x}, y={y}, w={w}, h={h}")

    cv2.imwrite(output_path, image)

print(f"Completed. Result is saved: {output_path}")