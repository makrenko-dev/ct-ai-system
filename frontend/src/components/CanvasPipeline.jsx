import { useRef, useEffect } from "react";

export default function CanvasPipeline({
  imageSrc,
  stage1 = [],
  stage2 = [],
  showStage1 = true,
  showStage2 = true,
}) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  useEffect(() => {
    if (!imageSrc) return;

    const img = new Image();
    img.src = imageSrc;
    img.onload = () => {
      imgRef.current = img;
      draw();
    };
  }, [imageSrc, stage1, stage2]);

  const drawBoxes = (ctx, boxes, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.font = "18px Arial";
    ctx.fillStyle = color;

    boxes.forEach((b) => {
      const [x1, y1, x2, y2] = b.bbox;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillText(b.conf.toFixed(2), x1 + 6, y1 + 20);
    });
  };

  const draw = () => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    // ✅ 1:1 размеры
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // image
    ctx.drawImage(img, 0, 0);

    if (showStage1) drawBoxes(ctx, stage1, "red");
    if (showStage2) drawBoxes(ctx, stage2, "lime");
  };

  return (
    <div style={{ maxWidth: "800px" }}>
      <canvas
        ref={canvasRef}
        style={{
          width: "100%",       // ✅ масштаб ТОЛЬКО визуальный
          height: "auto",
          border: "1px solid #444",
        }}
      />
    </div>
  );
}
