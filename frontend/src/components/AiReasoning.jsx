export default function AIReasoning({ text }) {
    if (!text) return null;
  
    return (
      <div className="glass panel">
        <h3 className="panel-title">AI reasoning</h3>
        <p className="reasoning-text">{text}</p>
  
        <p className="muted" style={{ fontSize: "12px", marginTop: "8px" }}>
          *Це автоматично згенероване пояснення — не є клінічним висновком.
        </p>
      </div>
    );
  }
  