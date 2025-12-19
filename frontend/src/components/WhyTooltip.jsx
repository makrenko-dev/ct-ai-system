// src/components/WhyTooltip.jsx
import { useState } from "react";
import { createPortal } from "react-dom";

export default function WhyTooltip({ type, confidence }) {
  const [open, setOpen] = useState(false);

  const tier =
    confidence >= 0.9
      ? "високий"
      : confidence >= 0.75
      ? "середній"
      : "низький";

  return (
    <div
      className="why-wrap"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <span className="why-link">Чому саме така класифікація?</span>

      {open &&
        createPortal(
        <div className="why-tooltip glass">
          {type === "mass" ? (
            <p>
              Виявлена зона має{" "}
              <b>локальне ущільнення</b> з{" "}
              <b>відносно однорідною текстурою</b> та{" "}
              <b>чітко відмежованими границями</b>. Така морфологія типово
              відповідає <b>масовому утворенню (mass)</b> на мамографії.
            </p>
          ) : (
            <p>
              У ділянці спостерігаються{" "}
              <b>множинні дрібні яскраві фокуси</b> з{" "}
              <b>зернистою структурою</b>, розташовані у
              невеликому кластері. Це відповідає типовому патерну{" "}
              <b>кальцифікацій</b> на мамографічних зображеннях.
            </p>
          )}

          {confidence !== undefined && (
            <div className="why-footer">
              Рівень впевненості моделі: <b>{tier}</b> (
              {Math.round(confidence * 100)}%)
            </div>
          )}
        </div>,
        document.body
      )}
    </div>
  );
}
