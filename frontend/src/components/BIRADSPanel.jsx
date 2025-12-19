// src/components/BIRADSPanel.jsx

function biradsInfo(score) {
  switch (score) {
    case 2:
      return {
        label: "Ймовірно доброякісне",
        text:
          "Патерн активності теплової карти відповідає низькому онкологічному ризику. " +
          "У клінічній шкалі BI-RADS категорія 2 зазвичай означає доброякісні зміни.",
      };
    case 3:
      return {
        label: "Переважно доброякісне",
        text:
          "AI виявляє помірну активність підозрілих зон. У реальній практиці BI-RADS 3 " +
          "часто означає високу ймовірність доброякісності з рекомендацією динамічного спостереження.",
      };
    case 4:
      return {
        label: "Підозра на злоякісність",
        text:
          "Розподіл активності теплової карти нагадує випадки з підвищеним ризиком. " +
          "У клінічній шкалі BI-RADS 4 це категорія, для якої зазвичай розглядають додаткову діагностику, " +
          "зокрема біопсію.",
      };
    case 5:
      return {
        label: "Високий онкологічний ризик",
        text:
          "AI фіксує виражені, локалізовані зони високої активності. " +
          "У реальній шкалі BI-RADS категорія 5 означає високу ймовірність злоякісного процесу " +
          "(>95%) і майже завжди супроводжується рекомендацією морфологічного підтвердження (біопсія).",
      };
    default:
      return {
        label: "Орієнтовна оцінка",
        text:
          "AI-оцінка BI-RADS розрахована за спрощеною евристикою на основі теплової карти " +
          "і використовується лише для демонстраційних цілей.",
      };
  }
}

export default function BIRADSPanel({ assessment }) {
  if (!assessment) return null;

  const info = biradsInfo(assessment.birads);

  return (
    <div className="panel glass">
      <h3 className="panel-title">Орієнтовна AI-оцінка BI-RADS</h3>

      <div className="birads-header-row">
        <div className="birads-badge">BI-RADS {assessment.birads}</div>
        <div className="birads-label">{info.label}</div>
      </div>

      <p className="birads-text muted">{info.text}</p>

      <div className="birads-metrics">
        <div>
          <div className="label">Максимальна інтенсивність</div>
          <div className="value">{assessment.max_val.toFixed(2)}</div>
        </div>
        <div>
          <div className="label">Частка середнього ризику</div>
          <div className="value">
            {(assessment.frac_mid * 100).toFixed(1)}%
          </div>
        </div>
        <div>
          <div className="label">Частка високого ризику</div>
          <div className="value">
            {(assessment.frac_high * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      <p className="disclaimer">
        Це <b>демонстраційна AI-оцінка</b>, побудована лише на статистиці теплової карти.
        Вона не замінює офіційну класифікацію BI-RADS та не може використовуватися
        для прийняття клінічних рішень. Остаточний висновок завжди належить
        лікарю-рентгенологу.
      </p>
    </div>
  );
}
