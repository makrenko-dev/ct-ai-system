import React, { useEffect, useRef } from "react";

function jet(v) {
  const r = Math.min(255, Math.max(0, 255 * Math.min(4*v - 1.5, -4*v + 4.5)));
  const g = Math.min(255, Math.max(0, 255 * Math.min(4*v - 0.5, -4*v + 3.5)));
  const b = Math.min(255, Math.max(0, 255 * Math.min(4*v + 0.5, -4*v + 2.5)));
  return [r, g, b];
}

function drawCamOnROI(canvas, image, cam, bbox, alpha = 0.6) {
  const ctx = canvas.getContext("2d");
  const [x1, y1, x2, y2] = bbox;
  const w = x2 - x1;
  const h = y2 - y1;

  canvas.width = w;
  canvas.height = h;

  // base ROI
  ctx.drawImage(image, x1, y1, w, h, 0, 0, w, h);

  // CAM
  const camCanvas = document.createElement("canvas");
  camCanvas.width = cam.width;
  camCanvas.height = cam.height;
  const camCtx = camCanvas.getContext("2d");

  const imgData = camCtx.createImageData(cam.width, cam.height);
  for (let i = 0; i < cam.values.length; i++) {
    const [r, g, b] = jet(cam.values[i]);
    imgData.data[i*4+0] = r;
    imgData.data[i*4+1] = g;
    imgData.data[i*4+2] = b;
    imgData.data[i*4+3] = 255;
  }
  camCtx.putImageData(imgData, 0, 0);

  ctx.globalAlpha = alpha;
  ctx.drawImage(camCanvas, 0, 0, w, h);
  ctx.globalAlpha = 1.0;

  // ROI frame
  ctx.strokeStyle = "rgba(255,165,0,0.9)";
  ctx.lineWidth = 2;
  ctx.strokeRect(0, 0, w, h);
}

export default function GradcamOnROI({ imageFile, gradcam, bbox }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!imageFile || !gradcam || !bbox) return;

    const img = new Image();
    img.src = URL.createObjectURL(imageFile);
    img.onload = () => {
      drawCamOnROI(canvasRef.current, img, gradcam, bbox);
    };

    return () => URL.revokeObjectURL(img.src);
  }, [imageFile, gradcam, bbox]);

  if (!gradcam || !bbox) return null;

  return (
    <div className="gradcam-block">
      <h3 className="gradcam-title">Grad-CAM (ROI)</h3>

      <div
        style={{
          maxWidth: 370,
          margin: "0 auto"
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            width: "100%",
            borderRadius: 10,
            background: "black"
          }}
        />
      </div>

      {/* Legend */}
      <div className="gradcam-legend">
        <div className="legend-row">
          <span className="legend-color red" />
          <span>Високий вплив на рішення</span>
        </div>
        <div className="legend-row">
          <span className="legend-color yellow" />
          <span>Середній вплив</span>
        </div>
        <div className="legend-row">
          <span className="legend-color blue" />
          <span>Низький / фоновий вплив</span>
        </div>
      </div>

      <div className="gradcam-caption">
        Відображено лише область ROI, у межах якої модель аналізувала ознаки.
      </div>
    </div>
  );
}
