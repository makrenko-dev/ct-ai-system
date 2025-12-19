export default function HeatmapDistribution({ heatmap }) {
    if (!heatmap?.values?.length) return null;
  
    // buckets 0–255 → 8 зон
    const buckets = new Array(8).fill(0);
  
    heatmap.values.forEach(v => {
      const idx = Math.min(7, Math.floor(v / 32));
      buckets[idx]++;
    });
  
    const max = Math.max(...buckets);
  
    return (
      <div className="panel glass">
        <h3 className="panel-title">Розподіл активності теплової карти</h3>
  
        <div className="histogram">
          {buckets.map((v, i) => (
            <div key={i} className="hist-bar">
              <div
                className="hist-fill"
                style={{
                  height: `${(v / max) * 100}%`,
                  background: barColor(i)
                }}
              />
              <div className="hist-label">{i * 12.5}%</div>
            </div>
          ))}
        </div>
  
        <div className="muted">
          max activity: <b>{heatmap.stats?.max}</b> ·
          mean: <b>{heatmap.stats?.mean}</b>
        </div>
      </div>
    );
  }
  
  function barColor(i) {
    if (i < 3) return "#22c55e";
    if (i < 5) return "#f59e0b";
    return "#ef4444";
  }
  