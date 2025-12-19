import pydicom
from pathlib import Path

path = Path("/Users/mariamakrenko/Documents/uni/ct-ai-system/datasets/CMMD/dicom/manifest-1616439774456/CMMD/D1-0304")

for dicom in path.rglob("*.dcm"):
    print("\n===== FILE:", dicom, "=====")
    ds = pydicom.dcmread(dicom)
    print(ds)
    break
