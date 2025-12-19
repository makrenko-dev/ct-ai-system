export default function Toggles({
  show1, show2, show3,
  setShow1, setShow2, setShow3
}) {
  return (
    <div className="panel">
      <h3>Layers</h3>

      <label>
        <input
          type="checkbox"
          checked={show1}
          onChange={() => setShow1(!show1)}
        />
        Stage 1 – breast
      </label>

      <label>
        <input
          type="checkbox"
          checked={show2}
          onChange={() => setShow2(!show2)}
        />
        Stage 2 – lesions
      </label>

      <label>
        <input
          type="checkbox"
          checked={show3}
          onChange={() => setShow3(!show3)}
        />
        Stage 3 – lesion type
      </label>
    </div>
  );
}
