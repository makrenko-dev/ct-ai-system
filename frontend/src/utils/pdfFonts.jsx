import jsPDF from "jspdf";
import Roboto from "../assets/fonts/Roboto-Regular.ttf";

export function registerFonts(pdf) {
  pdf.addFileToVFS("Roboto-Regular.ttf", Roboto);
  pdf.addFont("Roboto-Regular.ttf", "Roboto", "normal");
  pdf.setFont("Roboto");
}
