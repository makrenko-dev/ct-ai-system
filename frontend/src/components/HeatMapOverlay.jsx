export default function HeatmapOverlay({ heatmap }) {
    if (!heatmap) return null;
  
    return (
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: `radial-gradient(
            circle at center,
            rgba(255,80,80,0.65),
            rgba(255,80,80,0.15),
            transparent 70%
          )`,
          filter: "blur(18px)",
          mixBlendMode: "screen",
        }}
      />
    );
  }
  