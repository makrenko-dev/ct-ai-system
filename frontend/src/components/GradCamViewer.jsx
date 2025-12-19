import React, { useEffect, useRef } from "react";

function jet(v) {
  const r = Math.min(255, Math.max(0, 255 * Math.min(4*v - 1.5, -4*v + 4.5)));
  const g = Math.min(255, Math.max(0, 255 * Math.min(4*v - 0.5, -4*v + 3.5)));
  const b = Math.min(255, Math.max(0, 255 * Math.min(4*v + 0.5, -4*v + 2.5)));
  return [r, g, b];
}

function renderCamWithBBox(canvas, image, cam, alpha = 0.6) {
  const ctx = canvas.getContext("2d");
  canvas.width = cam.width;
  canvas.height = cam.height;

  // 1. base image
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  // 2. heatmap
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < cam.values.length; i++) {
    const [r, g, b] = jet(cam.values[i]);
    imgData.data[i*4+0] = imgData.data[i*4+0]*(1-alpha) + r*alpha;
    imgData.data[i*4+1] = imgData.data[i*4+1]*(1-alpha) + g*alpha;
    imgData.data[i*4+2] = imgData.data[i*4+2]*(1-alpha) + b*alpha;
  }
  ctx.putImageData(imgData, 0, 0);

  // 3. bbox overlay (ROI)
  ctx.strokeStyle = "rgba(255, 165, 0, 0.95)"; // orange
  ctx.lineWidth = 2;
  ctx.strokeRect(
    1,
    1,
    canvas.width - 2,
    canvas.height - 2
  );
}

export default function GradcamViewer({ gradcam, cropBase64 }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  useEffect(() => {
    if (!gradcam || !imgRef.current || !canvasRef.current) return;
    renderCamWithBBox(canvasRef.current, imgRef.current, gradcam);
  }, [gradcam]);

  if (!gradcam || !cropBase64) return null;

  return (
    <div className="gradcam-block">
      <h3 className="gradcam-title">Grad-CAM Explanation</h3>

      <img
        ref={imgRef}
        src={`data:image/jpeg;base64,${cropBase64}`}
        style={{ display: "none" }}
        onLoad={() =>
          renderCamWithBBox(canvasRef.current, imgRef.current, gradcam)
        }
      />

      <canvas
        ref={canvasRef}
        style={{
          width: "100%",
          borderRadius: 10,
          boxShadow: "0 0 0 1px rgba(255,255,255,0.05)"
        }}
      />

      <div className="gradcam-caption">
        Рамка відповідає області ROI, в межах якої модель аналізувала ознаки.
      </div>
    </div>
  );
}
