// src/components/ClinicalSummary.jsx
import { FaExclamationTriangle } from "react-icons/fa";

export default function ClinicalSummary({ assessment, stage3 }) {
  if (!assessment?.birads || !Array.isArray(stage3) || stage3.length === 0) {
    return null;
  }

  const main = [...stage3].sort(
    (a, b) => b.confidence - a.confidence
  )[0];

  const confidencePct = Math.round(main.confidence * 100);

  return (
    <div className="panel glass clinical">
      <h3 className="panel-title">Клінічний AI-звіт</h3>

      {/* Тип ураження + впевненість */}
      <div className="summary-block">
        <div className="label">Провідний тип ураження: </div>
        <div className="summary-main">
          <span className="summary-lesion">
            {main.lesion_type === "mass"
              ? "Масове утворення "
              : "Кальцифікації "}
          </span>
          <span className="summary-conf">
            ({confidencePct}% впевненості моделі)
          </span>
        </div>

        <div className="conf-bar">
          <div
            className="conf-bar-fill"
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>

      {/* BI-RADS + risk */}
      <div className="summary-block alert">
        <FaExclamationTriangle className="alert-icon" />
        <div>
          <div className="summary-birads">
            BI-RADS {assessment.birads}
          </div>
          <div className="muted">
            Поєднання карти ризику та локальних вогнищ відповідає цій
            категорії BI-RADS.
          </div>
        </div>
      </div>

      {/* AI reasoning */}
      <div className="summary-block">
        <div className="label">Як система прийняла рішення</div>
        <p className="summary-text">
          Модель поєднує інтенсивність локальної heatmap, морфологічні ознаки
          ураження та розподіл підозрілих пікселів. Для поточного випадку
          більшість «гарячих» пікселів локалізована навколо
          {main.lesion_type === "mass"
            ? " обʼємного утворення зі щільною текстурою."
            : " кластера дрібних яскравих структур, характерних для кальцифікацій."}
        </p>
        <p className="summary-text">
          BI-RADS {assessment.birads} інтерпретується з урахуванням того, що
          {assessment.frac_high > 0.05
            ? " суттєва частина heatmap припадає на зону високого ризику."
            : " лише невелика частина heatmap демонструє високі значення, що узгоджується з низькою ймовірністю клінічно значущих змін."}
        </p>
      </div>

      <p className="disclaimer">
        Цей звіт створено автоматично з навченої моделі глибокого навчання.
        Він не є медичним висновком і повинен бути інтерпретований лише
        кваліфікованим лікарем-рентгенологом у контексті повної клінічної
        інформації.
      </p>
    </div>
  );
}
