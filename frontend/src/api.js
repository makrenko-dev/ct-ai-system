export async function runPipeline(file) {
    const form = new FormData();
    form.append("file", file);
  
    const res = await fetch("http://127.0.0.1:8000/pipeline_visual", {
      method: "POST",
      body: form,
    });
  
    if (!res.ok) {
      throw new Error("Pipeline failed");
    }
  
    return await res.json();
  }
  
export async function runClinical(input) {
const res = await fetch("http://localhost:8000/clinical_predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
});

if (!res.ok) {
    throw new Error("Clinical AI request failed");
}
return await res.json();
}