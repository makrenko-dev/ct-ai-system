// src/components/PipelineViewer.jsx
import { useEffect, useRef } from "react";

export default function PipelineViewer({
  image,
  stage1 = [],
  stage2 = [],
  heatmap,
  showStage1,
  showStage2,
  selectedLesion,
  heatOpacity = 0.6,
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!image) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const img = new Image();
    const url = URL.createObjectURL(image);

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      // 1) Базовое изображение
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      // 2) Heatmap (если включена)
      if (heatmap && heatOpacity > 0 && showStage2) {
        drawHeatmap(ctx, canvas, heatmap, heatOpacity);
      }

      // 3) Stage1 – зеленые рамки
      if (showStage1) {
        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = 3;
        stage1.forEach((b) => drawBox(ctx, b.bbox));
      }

      // 4) Stage2 – синеватые рамки
      if (showStage2) {
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        stage2.forEach((b) => drawBox(ctx, b.bbox));
      }

      // 5) Selected lesion – жирная рамка
      if (selectedLesion?.bbox && showStage2) {
        ctx.strokeStyle = "#f97316"; // оранжевая
        ctx.lineWidth = 4;
        drawBox(ctx, selectedLesion.bbox);
      }

      URL.revokeObjectURL(url);
    };

    img.src = url;
  }, [image, stage1, stage2, heatmap, showStage1, showStage2, selectedLesion, heatOpacity]);

  return (
    <div className="canvas-wrapper">
      <canvas ref={canvasRef} className="canvas-main" />
    </div>
  );
}

// ---------- helpers ----------

function drawBox(ctx, [x1, y1, x2, y2]) {
  ctx.beginPath();
  ctx.rect(x1, y1, x2 - x1, y2 - y1);
  ctx.stroke();
}

function drawHeatmap(ctx, canvas, heatmap, opacity) {
  const { width, height, values } = heatmap;

  const tmp = document.createElement("canvas");
  tmp.width = width;
  tmp.height = height;
  const tctx = tmp.getContext("2d");

  values.forEach((v, i) => {
    const x = i % width;
    const y = Math.floor(i / width);

    if (v < 10) return; // отсекаем шум

    const alpha = Math.min((v / 255) * opacity, opacity);
    const r = 248;
    const g = 113;
    const b = 113; // мягкий красный

    tctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
    tctx.fillRect(x, y, 1, 1);
  });

  ctx.globalCompositeOperation = "screen";
  ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = "source-over";
}
