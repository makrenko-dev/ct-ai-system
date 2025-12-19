export default function GradCamOverlay({ image, visible }) {
    if (!visible) return null;
  
    return (
      <img
        src={image}
        className="gradcam"
        style={{
          position: "absolute",
          inset: 0,
          opacity: 0.45,
          mixBlendMode: "screen",
          pointerEvents: "none",
        }}
      />
    );
  }
  