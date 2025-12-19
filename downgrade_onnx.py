import onnx

src = "backend/models/small_tumor_best.onnx"
dst = "backend/models/small_tumor_best_ir9.onnx"

model = onnx.load(src)

print("Before IR:", model.ir_version)

# 🔽 ЖЁСТКО понизим IR
model.ir_version = 9

onnx.save(model, dst)

print("✅ Saved:", dst)
print("After IR:", model.ir_version)
