// src/App.jsx
import { useState } from "react";
import { runPipeline } from "./api";

import PipelineViewer from "./components/PipelineViewer";
import Timeline from "./components/Timeline";
import Toggles from "./components/Toggles";
import BIRADSPanel from "./components/BIRADSPanel";
import Stage3Panel from "./components/Stage3Panel";
import ClinicalSummary from "./components/ClinicalSummary";
import LesionZoom from "./components/LesionZoom.jsx";
import HeatmapDistribution from "./components/HeatmapDistribution";
import AIReasoning from "./components/AiReasoning";
import GradcamViewer from "./components/GradCamViewer";
import ClinicalAIPanel from "./components/ClinicalAIPanel";
import GradcamOnImage from "./components/GradCamOnImage";

export default function App() {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);

  const [showStage1, setShowStage1] = useState(true);
  const [showStage2, setShowStage2] = useState(true);
  const [heatOpacity, setHeatOpacity] = useState(0.6);

  const [selectedLesion, setSelectedLesion] = useState(null);

  async function onRun() {
    if (!file) return;
    const result = await runPipeline(file);
    setData(result);

    // выбрать самый уверенный очаг по умолчанию
    if (result?.boxes?.stage3?.length) {
      const best = [...result.boxes.stage3].sort(
        (a, b) => b.confidence - a.confidence
      )[0];
      setSelectedLesion(best);
    }
  }

  function onClear() {
    setFile(null);
    setData(null);
  }
  

  return (
    <div className="app-root">
       <div className="main-header-layout">
        <div className="left-header">
          <header className="app-header">
            <h1>AI-аналіз мамографічних зображень</h1>
            <p className="app-subtitle">
              Дослідницький прототип. Результати не є клінічним діагнозом.
            </p>
          </header>

        {/* Upload card */}
        <div className="upload-card glass">
          <input
            id="file"
            type="file"
            hidden
            accept="image/png,image/jpeg"
            onChange={(e) => {
              const f = e.target.files?.[0];
              setFile(f || null);
              setData(null);
              setSelectedLesion(null);
            }}
          />

          <div htmlFor="file" className="upload-drop" onClick={() => {
            document.getElementById("file").click();
          }}>
            <div className="upload-title">
              Завантажте мамографічне зображення
            </div>
            <div className="upload-sub">
              Підтримувані формати: PNG / JPG • Один знімок
            </div>

            {file && (
              <div className="upload-file">
                <span>📄 {file.name}</span>
              </div>
            )}
          </div>

          <button
            className="run-button"
            onClick={onRun}
            disabled={!file}
          >
            Запустити AI-аналіз
          </button>
        </div>
        </div>
        <div className="right-header">
          <header className="app-header">
            <h1>AI-аналіз анамнезу пацієнта</h1>
            <p className="app-subtitle">
              Дослідницький прототип. Результати не є клінічним діагнозом.
            </p>
          </header>

          <ClinicalAIPanel />
          </div>
        </div>

      {data && (
        <>
          {/* Timeline, который тоже умеет включать/выключать слои */}
          <Timeline
            stage1={data.boxes.stage1}
            stage2={data.boxes.stage2}
            showStage1={showStage1}
            showStage2={showStage2}
            onToggleStage1={() => setShowStage1(s => !s)}
            onToggleStage2={() => setShowStage2(s => !s)}
            onClear={onClear}
          />


          <div className="main-layout">
            {/* Левая часть – изображение + heatmap + слайдер */}
            <div className="main-left glass">
              <div className="canvas-header">
                <div>
                  <div className="label">Візуалізація</div>
                  <div className="value">Зони підвищеного ризику</div>
                </div>

                <div className="heatmap-controls">
                  <span className="label">Прозорість heatmap</span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={heatOpacity}
                    onChange={(e) =>
                      setHeatOpacity(parseFloat(e.target.value))
                    }
                  />
                </div>
              </div>

              <PipelineViewer
                image={file}
                heatmap={data.heatmap}
                stage1={data.boxes.stage1}
                stage2={data.boxes.stage2}
                showStage1={showStage1}
                showStage2={showStage2}
                selectedLesion={selectedLesion}
                heatOpacity={heatOpacity}
              />

              
            </div>

            {/* Правая часть – панелі з інформацією */}
            <div className="main-right">

              <Stage3Panel
                stage3={data.boxes.stage3}
                selected={selectedLesion}
                onSelect={(lesion) => setSelectedLesion(lesion)}
              />
              {/* {selectedLesion && (
                <LesionZoom image={file} lesion={selectedLesion} />
              )} */}
            {selectedLesion?.gradcam && (
              <GradcamOnImage
                imageFile={file}
                gradcam={selectedLesion.gradcam}
                bbox={selectedLesion.bbox}
              />
            )}




            </div>
          </div>
          <div className="secondary-layout">
              <div className="left-group">
                <BIRADSPanel
                  assessment={data.assessment}
                  heatmap={data.heatmap}
                />
                <AIReasoning text={data.ai_reasoning} />
              </div>
            
            <ClinicalSummary
              stage3={data.boxes.stage3}
              assessment={data.assessment}
            />

          </div>
        </>
      )}
    </div>
  );
}
