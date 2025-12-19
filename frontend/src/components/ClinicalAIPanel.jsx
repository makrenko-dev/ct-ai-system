// src/components/ClinicalAIPanel.jsx
import { useState } from "react";
import { runClinical } from "../api";

const initialForm = {
  age: "",
  menopause_status: "unknown",

  lesion_type_enc: "0",   // 0 = unknown, 1 = calc, 2 = mass
  assessment: "3",        // BI-RADS-like clinical assessment
  subtlety: "3",          // visibility of findings

  palpable_lump: false,
  pain: false,
  nipple_discharge: false,
  family_history: false,
  hormone_therapy: false,
  prior_biopsies: false,

  bmi: "",
  density: "3",
};

export default function ClinicalAIPanel() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isResultOpen, setIsResultOpen] = useState(true);

  const onChange = (field) => (e) => {
    const value =
      e.target.type === "checkbox" ? e.target.checked : e.target.value;
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  async function onSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const menopauseMap = {
        pre: 0,
        post: 1,
        unknown: 0.5,
      };

      const payload = {
        age: Number(form.age) || 0,
        density: Number(form.density),

        lesion_type_enc: Number(form.lesion_type_enc),
        assessment: Number(form.assessment),
        subtlety: Number(form.subtlety),

        bmi: Number(form.bmi) || 0,
        menopause_status: menopauseMap[form.menopause_status],

        palpable_lump: form.palpable_lump ? 1 : 0,
        pain: form.pain ? 1 : 0,
        nipple_discharge: form.nipple_discharge ? 1 : 0,
        family_history: form.family_history ? 1 : 0,
        hormone_therapy: form.hormone_therapy ? 1 : 0,
        prior_biopsies: form.prior_biopsies ? 1 : 0,
      };

      const res = await runClinical(payload);
      setResult(res);
      setIsResultOpen(true);

    } catch (err) {
      console.error(err);
      alert("Помилка при виклику Clinical AI");
    } finally {
      setLoading(false);
    }
  }

  const mal = result?.malignant;
  const malignantPercent =
    mal?.prob != null ? Math.round(mal.prob * 100) : null;
    const rawScore =
    result?.model_output?.model_score != null
    ? Math.round(result.model_output.model_score * 100)
    : null;


  const lesionText = {
    0: "Невизначено (клінічних ознак недостатньо)",
    1: "Переважають ознаки кальцифікацій",
    2: "Переважають ознаки обʼємного утворення (mass)",
  };

  return (
    <div className="glass panel">
      <div className="panel-title">Clinical AI (анамнез)</div>

      <p className="muted">
        Оцінка ризику злоякісності на основі клінічних симптомів
        та анамнезу (без використання зображень).
      </p>

      <form className="clinical-form" onSubmit={onSubmit}>

        {/* === BASIC DATA === */}
        <div className="clinical-grid">
          <label>
            Вік
            <input
              type="number"
              min="18"
              max="100"
              value={form.age}
              onChange={onChange("age")}
            />
          </label>

          <label>
            Менопаузальний статус
            <select
              value={form.menopause_status}
              onChange={onChange("menopause_status")}
            >
              <option value="pre">Пре- / перименопауза</option>
              <option value="post">Постменопауза</option>
              <option value="unknown">Невідомо</option>
            </select>
          </label>

          <label>
            BMI
            <input
              type="number"
              step="0.1"
              value={form.bmi}
              onChange={onChange("bmi")}
            />
          </label>

          <label>
            Щільність тканини (BI-RADS)
            <select value={form.density} onChange={onChange("density")}>
              <option value="1">1 – майже повністю жирова</option>
              <option value="2">2 – розріджена фіброгландулярна</option>
              <option value="3">3 – неоднорідно щільна</option>
              <option value="4">4 – надзвичайно щільна</option>
            </select>
          </label>
        </div>

        {/* === CLINICAL CONTEXT === */}
        <div className="clinical-grid">
          <label>
            Ймовірний тип ураження (за симптомами)
            <select
              value={form.lesion_type_enc}
              onChange={onChange("lesion_type_enc")}
            >
              <option value="0">Невизначено</option>
              <option value="1">Кальцифікації</option>
              <option value="2">Обʼємне утворення (mass)</option>
            </select>
          </label>

          <label>
            Клінічна оцінка (1–5)
            <select value={form.assessment} onChange={onChange("assessment")}>
              <option value="1">1 – норма</option>
              <option value="2">2 – доброякісні зміни</option>
              <option value="3">3 – ймовірно доброякісні</option>
              <option value="4">4 – підозрілі</option>
              <option value="5">5 – висока підозра</option>
            </select>
          </label>

          <label>
            Вираженість змін (subtlety)
            <select value={form.subtlety} onChange={onChange("subtlety")}>
              <option value="1">1 – ледь помітні</option>
              <option value="2">2 – слабкі</option>
              <option value="3">3 – помірні</option>
              <option value="4">4 – чіткі</option>
              <option value="5">5 – дуже виражені</option>
            </select>
          </label>
        </div>

        {/* === SYMPTOMS === */}
        <div className="clinical-checkboxes">
          <label><input type="checkbox" checked={form.palpable_lump} onChange={onChange("palpable_lump")} /> Пальпований вузол</label>
          <label><input type="checkbox" checked={form.pain} onChange={onChange("pain")} /> Біль</label>
          <label><input type="checkbox" checked={form.nipple_discharge} onChange={onChange("nipple_discharge")} /> Виділення з соска</label>
          <label><input type="checkbox" checked={form.family_history} onChange={onChange("family_history")} /> Сімейний анамнез</label>
          <label><input type="checkbox" checked={form.hormone_therapy} onChange={onChange("hormone_therapy")} /> Гормонотерапія</label>
          <label><input type="checkbox" checked={form.prior_biopsies} onChange={onChange("prior_biopsies")} /> Попередні біопсії</label>
        </div>

        <button className="run-button" type="submit" disabled={loading}>
          {loading ? "Аналізуємо..." : "Оцінити клінічний ризик"}
        </button>
      </form>

      {result && (
        <button
            type="button"
            className="toggle-result-button"
            onClick={() => setIsResultOpen((v) => !v)}
        >
            {isResultOpen ? "Згорнути результат" : "Розгорнути результат"}
        </button>
        )}


      {result  && isResultOpen &&(
        <div className="clinical-result">

            {/* === INSUFFICIENT DATA === */}
            {result.status === "insufficient_data" && (
            <div className="result-card error">
                <div className="result-header">
                <span className="emoji">⚠️</span>
                <h4>Недостатньо клінічних даних</h4>
                </div>

                <p className="muted">{result.message}</p>
                <p className="muted small">{result.recommendation}</p>
            </div>
            )}

            {/* === PARTIAL DATA === */}
            {result.status === "partial" && (
                <div className="result-card warning">
                    <div className="result-header">
                    <span className="emoji">🟡</span>
                    <h4>Орієнтовна клінічна оцінка</h4>
                    </div>

                    <div className="birads-badge">
                    BI-RADS {result.malignant.birads_from_symptoms}
                    </div>

                    <p>
                    Ймовірність злоякісності:{" "}
                    <strong>{Math.round(result.malignant.prob * 100)}%</strong>
                    </p>

                    <p className="muted">
                    Дані заповнені частково, інтерпретація обмежена.
                    </p>

                    <p className="muted tiny">
                    Оцінка базується на клінічній логіці. AI score використовується як допоміжний фактор.
                    </p>
                </div>
                )}


            {/* === FULL DATA === */}
            {result.status === "full" && (
            <>
                <div className="result-card">
                <div className="result-header">
                    <span className="emoji">🧠</span>
                    <h4>Клінічний висновок</h4>
                </div>

                <div className="birads-badge birads-main">
                    BI-RADS {result.malignant.birads_from_symptoms}
                </div>

                    <p>
                    Клінічний ризик злоякісності:{" "}
                    <strong>{Math.round(result.malignant.prob * 100)}%</strong>{" "}
                    <span className="muted">
                        ({result.malignant.label_name})
                    </span>
                    </p>

                    {rawScore !== null && (
                        <p className="muted tiny">
                            AI model score (без клінічної інтерпретації): {rawScore}%
                        </p>
                    )}

                </div>

                <div className="result-card">
                <div className="result-header">
                    <span className="emoji">🧩</span>
                    <h4>Пояснення рішення AI</h4>
                </div>

                <p className="muted">{result.explanation.summary}</p>

                <ul className="factor-list">
                    {result.explanation.key_factors.map((f, i) => (
                    <li key={i}>{f}</li>
                    ))}
                </ul>

                <p className="muted tiny">{result.explanation.note}</p>
                </div>
            </>
            )}
        </div>
        )}

    </div>
  );
}
