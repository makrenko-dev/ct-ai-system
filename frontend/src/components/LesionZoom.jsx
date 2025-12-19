// src/components/LesionZoom.jsx
import { useEffect, useRef } from "react";

export default function LesionZoom({ image, lesion }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!image || !lesion?.bbox) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    const url = URL.createObjectURL(image);

    img.onload = () => {
      const [x1, y1, x2, y2] = lesion.bbox.map((v) => Math.max(0, v));
      const w = x2 - x1;
      const h = y2 - y1;

      const pad = 10;
      const sx = Math.max(0, x1 - pad);
      const sy = Math.max(0, y1 - pad);
      const sw = Math.min(img.width - sx, w + pad * 2);
      const sh = Math.min(img.height - sy, h + pad * 2);

      const targetSize = 260;
      canvas.width = targetSize;
      canvas.height = targetSize;

      ctx.clearRect(0, 0, targetSize, targetSize);
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, targetSize, targetSize);

      ctx.strokeStyle = "#f97316";
      ctx.lineWidth = 3;
      ctx.strokeRect(4, 4, targetSize - 8, targetSize - 8);

      URL.revokeObjectURL(url);
    };

    img.src = url;
  }, [image, lesion]);

  return (
    <div className="lesion-zoom glass">
      <div className="label">Збільшення вибраного утворення</div>
      <canvas ref={canvasRef} className="zoom-canvas" />
    </div>
  );
}
