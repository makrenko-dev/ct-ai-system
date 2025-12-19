// src/components/Timeline.jsx

function Step({ title, done, active, onClick }) {
  return (
    <button
      type="button"
      className={`step ${done ? "done" : ""} ${active ? "active" : ""}`}
      onClick={onClick}
    >
      <span className="step-icon">{done ? "●" : "○"}</span>
      <span className="step-title">{title}</span>
    </button>
  );
}

export default function Timeline({
  stage1,
  stage2,
  showStage1,
  showStage2,
  onToggleStage1,
  onToggleStage2,
  onClear,            // ← добавили
}) {
  return (
    <div className="timeline glass">
      <div className="timeline-steps">
        <Step
          title="1. Breast detection"
          done={stage1.length > 0}
          active={showStage1}
          onClick={onToggleStage1}
        />
        <Step
          title="2. Tumor detection"
          done={stage2.length > 0}
          active={showStage2}
          onClick={onToggleStage2}
        />
        <div className="step done static">
          <span className="step-icon">●</span>
          <span className="step-title">3. AI assessment</span>
        </div>
      </div>

      <button className="timeline-clear-btn" onClick={onClear}>
        ✕ Очистити
      </button>
    </div>
  );
}
