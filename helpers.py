import time

import albumentations as A
import cv2
import numpy as np
import onnxruntime as ort


class BackgroundRemoval:
    def __init__(self, model_path, device_inference):
        self.model_path = model_path
        self.device_inference = device_inference
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float16)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float16)
        self.model = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"]
            if self.device_inference == "cpu"
            else ["CUDAExecutionProvider"],
        )
        self.input_height, self.input_width = self.model.get_inputs()[0].shape[2:]

    def preprocessing_image(self, image):
        image_augmentation = A.Compose(
            [
                A.Normalize(self.mean, self.std),
            ]
        )
        image = image_augmentation(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        return image

    def postprocessing_image(self, results, image):
        results = (results[0][0] * 255).astype(np.uint8)
        image_output = image.copy()
        mask_image = cv2.resize(results, image_output.shape[:2][::-1])
        segmented_image = cv2.bitwise_and(image_output, image_output, mask=mask_image)
        return segmented_image, mask_image

    def predict(
        self,
        image_pillow,
    ):
        image = np.array(image_pillow)
        image_input = image.copy()
        image_input = cv2.resize(image, (self.input_width, self.input_height))
        input_value = self.preprocessing_image(image_input)
        output_value = self.model.run(
            None,
            {self.model.get_inputs()[0].name: input_value},
        )[0]
        segmented_image, mask_image = self.postprocessing_image(output_value, image)
        return segmented_image, mask_image


def main(value_parser):
    model = BackgroundRemoval(value_parser.model_path, value_parser.device_inference)
    start_time = time.perf_counter()
    segmented_image, _ = model.predict(value_parser.image_path)
    print(f"Time: {time.perf_counter() - start_time}")
    cv2.imwrite(
        value_parser.output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    )
