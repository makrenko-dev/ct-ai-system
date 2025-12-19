import cv2
import numpy as np
import onnxruntime as ort

IMG_SIZE = 224
CLASS_NAMES = ["calcification", "mass"]


class LesionClassifierONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, img):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # CHW
        img = np.expand_dims(img, 0)
        return img

    def predict(self, img):
        x = self._preprocess(img)
        logits = self.session.run(None, {self.input_name: x})[0][0]
        probs = self._softmax(logits)

        idx = int(np.argmax(probs))
        return {
            "label": CLASS_NAMES[idx],
            "confidence": float(probs[idx]),
            "probs": {
                CLASS_NAMES[0]: float(probs[0]),
                CLASS_NAMES[1]: float(probs[1]),
            },
        }

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()
