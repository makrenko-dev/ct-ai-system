import onnxruntime as ort
import numpy as np
import cv2


class ONNXGradCAM:
    """
    Простая, рабочая реализация Grad-CAM для ONNX моделей типа EfficientNet.
    Работает через ручной захват активаций внутреннего слоя.
    """

    def __init__(self, onnx_path: str, target_layer: str = None):
        self.sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.sess.get_inputs()[0].name

        # usually last conv is "blocks_6/conv" or similar
        # если не указан — берем ПЕРЕД-последний выход графа
        outputs = [o.name for o in self.sess.get_outputs()]
        if target_layer is None:
            self.target_layer = outputs[-2]
        else:
            self.target_layer = target_layer

        self.final_output = outputs[-1]

    def preprocess(self, img):
        img = cv2.resize(img, (224, 224))
        img = img[..., ::-1] / 255.0
        img = (img - 0.485, img - 0.456, img - 0.406)
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    def generate_cam(self, img):
        """
        Возвращает heatmap CAM размера 224x224.
        """

        x = self.preprocess(img)

        # запускаем модель и просим вернуть:
        # 1) активации target_layer
        # 2) финальные логиты
        activations, logits = self.sess.run(
            [self.target_layer, self.final_output],
            {self.input_name: x}
        )

        activations = activations[0]            # [C, H, W]
        weights = logits[0]                     # [num_classes]

        # берём класс максимального логита
        cls = np.argmax(weights)

        # делаем weighted sum
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for c in range(activations.shape[0]):
            cam += activations[c] * weights[cls]

        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

        return cam

    def overlay(self, img, cam, alpha=0.45):
        cam_col = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        out = cv2.addWeighted(cam_col, alpha, img, 1 - alpha, 0)
        return out
