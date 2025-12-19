// src/components/Stage3Panel.jsx
import WhyTooltip from "./WhyTooltip";
import GradcamViewer from "./GradCamViewer";

export default function Stage3Panel({ stage3 = [], selected, onSelect }) {
  return (
    <div className="panel glass">
      <h3 className="panel-title">Класифікація патологічних утворень</h3>

      {stage3.length === 0 && (
        <p className="muted">Патологічні утворення не виявлені або не класифіковані.</p>
      )}

      {stage3.map((l, i) => {
        const isSelected = selected && selected === l;
        return (
          <div
            key={i}
            className={
              "lesion-card " +
              `tier-${l.confidence_tier} ` +
              (isSelected ? "lesion-selected" : "")
            }
            onClick={() => onSelect?.(l)}
          >
            <div className="lesion-header">
              <span className="lesion-type">{labelUA(l.lesion_type)}</span>
              <span className="lesion-conf">
                {Math.round(l.confidence * 100)}%
              </span>
            </div>

            <div className="lesion-sub">
              Рівень впевненості: <b>{tierUA(l.confidence_tier)}</b>
            </div>

            <WhyTooltip
              type={l.lesion_type}
              confidence={l.confidence}
            />
          </div>
        );
      })}
    </div>
  );
}

function labelUA(t) {
  return t === "mass" ? "Масове утворення" : "Кальцифікації";
}

function tierUA(t) {
  return t === "high"
    ? "високий"
    : t === "medium"
    ? "середній"
    : "низький";
}
