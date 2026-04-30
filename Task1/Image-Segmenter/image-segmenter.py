import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = mp.tasks.BaseOptions
image_segmenter = mp.tasks.vision.ImageSegmenter
image_segmenter_options = mp.tasks.vision.ImageSegmenterOptions
vision_running_mode = mp.tasks.vision.RunningMode

model_path = "deeplab_v3.tflite"
image_path = "input.jpg"
output_path = "segmentation_output.jpg"
mask_output_path = "segmentation_mask.jpg"


def create_colored_mask(category_mask):
    mask = category_mask.numpy_view()

    normalized_mask = mask.astype(np.uint8)

    colored_mask = cv2.applyColorMap(
        normalized_mask * 10,
        cv2.COLORMAP_JET
    )

    return colored_mask


def blend_image_with_mask(image, colored_mask, alpha=0.5):
    colored_mask = cv2.resize(
        colored_mask,
        (image.shape[1], image.shape[0])
    )

    blended_image = cv2.addWeighted(
        image,
        1 - alpha,
        colored_mask,
        alpha,
        0
    )

    return blended_image


def main():
    options = image_segmenter_options(
        base_options=base_options(model_asset_path=model_path),
        running_mode=vision_running_mode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=False
    )

    with image_segmenter.create_from_options(options) as segmenter:
        mp_image = mp.Image.create_from_file(image_path)

        segmentation_result = segmenter.segment(mp_image)

        image = cv2.imread(image_path)

        if image is None:
            print("Error: Could not read the input image.")
            return

        category_mask = segmentation_result.category_mask

        if category_mask is None:
            print("Error: No category mask was returned.")
            return

        colored_mask = create_colored_mask(category_mask)
        blended_image = blend_image_with_mask(image, colored_mask)

        cv2.imwrite(mask_output_path, colored_mask)
        cv2.imwrite(output_path, blended_image)

        mask_array = category_mask.numpy_view()
        unique_categories = np.unique(mask_array)

        print(f"Output image saved as: {output_path}")
        print(f"Mask image saved as: {mask_output_path}")
        print(f"Detected category indexes: {unique_categories}")


if __name__ == "__main__":
    main()