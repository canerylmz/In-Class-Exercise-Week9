import mediapipe as mp

base_options = mp.tasks.BaseOptions
image_classifier = mp.tasks.vision.ImageClassifier
image_classifier_options = mp.tasks.vision.ImageClassifierOptions
vision_running_mode = mp.tasks.vision.RunningMode

model_path = "efficientnet_lite0.tflite"
image_path = "cat.jpg"


def print_classification_result(classification_result):
    if not classification_result.classifications:
        print("No classification result was returned.")
        return

    for classification_index, classification in enumerate(classification_result.classifications):
        print(f"\nClassification head {classification_index + 1}:")

        for category in classification.categories:
            category_name = category.category_name
            display_name = category.display_name
            score = category.score
            index = category.index

            if display_name:
                label = display_name
            else:
                label = category_name

            print(
                f"Label: {label}, "
                f"Score: {score:.4f}, "
                f"Index: {index}"
            )


def main():
    options = image_classifier_options(
        base_options=base_options(model_asset_path=model_path),
        running_mode=vision_running_mode.IMAGE,
        max_results=5,
        score_threshold=0.0
    )

    with image_classifier.create_from_options(options) as classifier:
        mp_image = mp.Image.create_from_file(image_path)

        classification_result = classifier.classify(mp_image)

        print("Image classification completed.")
        print_classification_result(classification_result)


if __name__ == "__main__":
    main()