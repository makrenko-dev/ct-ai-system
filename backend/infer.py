import onnxruntime as ort
import numpy as np
import cv2


class YOLOONNX:
    def __init__(
        self,
        model_path,
        imgsz=1024,
        conf_thres=0.25,
        iou_thres=0.45,
    ):
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self._printed = False

    # ---------------- PREPROCESS ----------------
    def preprocess(self, img):
        h0, w0 = img.shape[:2]

        img_rs = cv2.resize(img, (self.imgsz, self.imgsz))

        sx = self.imgsz / w0
        sy = self.imgsz / h0

        img_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
        img_rs = img_rs.astype(np.float32) / 255.0
        img_rs = img_rs.transpose(2, 0, 1)[None]

        meta = {
            "sx": sx,
            "sy": sy,
            "orig_shape": (h0, w0)
        }
        return img_rs, meta

    # ---------------- NMS ----------------
    def _nms(self, boxes, scores):
        if not boxes:
            return []

        boxes = np.array(boxes, np.float32)
        scores = np.array(scores, np.float32)

        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            order = order[np.where(iou <= self.iou_thres)[0] + 1]

        return keep

    # ---------------- POSTPROCESS ----------------
    def postprocess(self, out, meta):
        sx, sy = meta["sx"], meta["sy"]
        h0, w0 = meta["orig_shape"]

        out = out.T
        cx, cy, w, h, scores = out.T

        mask = scores >= self.conf_thres
        cx, cy, w, h, scores = cx[mask], cy[mask], w[mask], h[mask], scores[mask]

        if len(scores) == 0:
            return []

        x1 = (cx - w / 2) / sx
        y1 = (cy - h / 2) / sy
        x2 = (cx + w / 2) / sx
        y2 = (cy + h / 2) / sy

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = self._nms(boxes.tolist(), scores.tolist())

        return [{
            "bbox": [
                int(max(0, boxes[i][0])),
                int(max(0, boxes[i][1])),
                int(min(w0, boxes[i][2])),
                int(min(h0, boxes[i][3])),
            ],
            "conf": float(scores[i])
        } for i in keep]

    # ---------------- PREDICT ----------------
    def predict(self, img):
        inp, meta = self.preprocess(img)
        out = self.session.run(None, {self.input_name: inp})

        if not self._printed:
            print("ONNX output shape:", out[0].shape)
            self._printed = True

        return self.postprocess(out[0][0], meta)
