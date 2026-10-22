// Builds UPAR_Track1_presentation.pptx (needs: npm install pptxgenjs@3.12.0).
// Usage: node build_deck.js [path/to/apply_theme.js]
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const OUT = path.join(HERE, "UPAR_Track1_presentation.pptx");
const SHOT = path.join(HERE, "..", "proof_of_submission", "sub03_screenshot.png");

const THEME = {
  name: "UPAR Surveillance",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1E2328", lt1: "FFFFFF", dk2: "2B3A42", lt2: "EEF1F3",
    accent1: "E8A33D", accent2: "2A9D8F", accent3: "6C7A89", accent4: "C8553D",
    accent5: "8AB0AB", accent6: "3D5A80", hlink: "2A9D8F", folHlink: "6C7A89",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "UPAR 2027 Track 1: Pedestrian Attribute Recognition";
pres.author = "Muhammad Ali, Fasih Hasan Khan";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.55, w: 8.8, h: 1.5, fontFace: THEME.headFontFace,
      fontSize: 38, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.2, w: 8.8, h: 1.4, fontSize: 18,
      color: C.accent1, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.5, 0.5, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontFace: THEME.headFontFace,
      fontSize: 26, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.1, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent3, align: "right" },
});

const card = (slide, x, y, w, h, fill, name) =>
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill },
    line: { color: fill }, objectName: name });

const text = (slide, t, o) => slide.addText(t, { isTextBox: true, margin: 0, ...o });

// ---------- 1. title ----------
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
s.addText("Pedestrian Attribute Recognition with a Language-Anchored CLIP", { placeholder: "title" });
s.addText([
  { text: "UPAR Challenge 2027, Track 1 (WACV 2027 Real-World Surveillance Workshop)", options: { breakLine: true } },
  { text: "Muhammad Ali  ·  Fasih Hasan Khan", options: { color: C.background2, fontSize: 16 } },
], { placeholder: "body" });
s.addNotes("We entered Track 1 of the UPAR Challenge: predicting 40 attributes of a pedestrian from a surveillance crop. " +
  "This talk covers the task, our approach, results on val and on the hidden test set, and what went wrong at the end.");

// ---------- 2. task ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("The task: 40 attributes per pedestrian", { placeholder: "title" });
const stats = [
  ["40", "binary attributes: age, gender, hair, clothing type and colour, accessories"],
  ["3", "training datasets with harmonized labels: Market1501, PA100k, PETA"],
  ["29,248", "hidden test images, scored with a fixed 0.5 threshold"],
  ["5 min", "for the whole run: our code in Docker on an A10 GPU, no network"],
];
stats.forEach(([big, small], i) => {
  const x = 0.5 + i * 2.3;
  card(s, x, 1.35, 2.1, 2.45, C.background2, `stat-card-${i}`);
  text(s, big, { x: x + 0.15, y: 1.5, w: 1.8, h: 0.9, fontFace: THEME.headFontFace, fontSize: 36, bold: true,
    color: i === 3 ? C.accent4 : C.accent2, objectName: `stat-${i}` });
  text(s, small, { x: x + 0.15, y: 2.45, w: 1.8, h: 1.25, fontSize: 13, color: C.text2, valign: "top",
    objectName: `stat-label-${i}` });
});
text(s, [
  { text: "Metric: ", options: { bold: true } },
  { text: "Challenge_Avg = harmonic mean of mA (mean of TPR and TNR per attribute) and instance F1, computed per dataset and averaged." },
], { x: 0.5, y: 4.15, w: 9.0, h: 0.75, fontSize: 14, color: C.text1, objectName: "metric" });
s.addNotes("Each image has 40 binary labels. The metric is the harmonic mean of label-based mA and instance F1, " +
  "computed per dataset and averaged. The threshold is fixed at 0.5, so calibration matters. " +
  "Submissions are code, run on the organizers' GPU with a 5-minute limit.");

// ---------- 3. approach ----------
pres.addSection({ title: "Approach" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Approach" });
s.addText("Approach: a CLIP backbone plus five components", { placeholder: "title" });
const flow = [
  ["Image", "uint8, 256×128"],
  ["CLIP ViT", "B/16 or L/14 image encoder"],
  ["N1 head", "text-initialised: s·cos(f, w_k) + b_k"],
  ["N4 offsets", "one global logit shift per attribute"],
  ["N5 decoding", "exclusive groups get an arg-max"],
];
flow.forEach(([h, d], i) => {
  const x = 0.5 + i * 1.84;
  card(s, x, 1.35, 1.6, 1.35, i === 0 ? C.text2 : C.background2, `flow-${i}`);
  text(s, h, { x: x + 0.1, y: 1.45, w: 1.4, h: 0.4, fontSize: 15, bold: true, color: i === 0 ? C.background1 : C.text1,
    objectName: `flow-h-${i}` });
  text(s, d, { x: x + 0.1, y: 1.88, w: 1.4, h: 0.75, fontSize: 12, color: i === 0 ? C.background2 : C.text2, valign: "top",
    objectName: `flow-d-${i}` });
  if (i < flow.length - 1)
    s.addShape(pres.shapes.RIGHT_TRIANGLE, { x: x + 1.64, y: 1.92, w: 0.16, h: 0.22, rotate: 0, flipH: false,
      fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `arrow-${i}` });
});
const trainBits = [
  ["N2  Domain-balanced sampling", "each dataset is a third of every epoch, matching the metric's equal weighting"],
  ["N3  Surveillance degradation", "low resolution, JPEG artefacts, blur and lighting changes during training"],
  ["Export", "one fp16 TorchScript module with offsets and decoding built in; run.py overlaps JPEG decoding with the GPU"],
];
trainBits.forEach(([h, d], i) => {
  const y = 3.05 + i * 0.68;
  s.addShape(pres.shapes.OVAL, { x: 0.5, y: y + 0.07, w: 0.22, h: 0.22, fill: { color: C.accent1 }, line: { color: C.accent1 },
    objectName: `dot-${i}` });
  text(s, [{ text: h + ":  ", options: { bold: true, color: C.text1 } }, { text: d, options: { color: C.text2 } }],
    { x: 0.9, y, w: 8.6, h: 0.4, fontSize: 14, objectName: `train-${i}` });
});
s.addNotes("At inference the image goes through the CLIP image encoder and a head whose weights start as CLIP text embeddings " +
  "of attribute prompts. Then calibrated offsets and group decoding. During training we balance the three datasets and simulate " +
  "surveillance-quality images. Everything is baked into one TorchScript file.");

// ---------- 4. val results ----------
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Validation: the ViT-B + ViT-L ensemble scores best", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{
  name: "held-out Challenge_Avg",
  labels: ["ConvNeXt-T", "CLIP ViT-B/16", "CLIP ViT-L/14", "ViT-B + ViT-L"],
  values: [0.8569, 0.8742, 0.8730, 0.8803],
}], {
  x: 0.5, y: 1.2, w: 5.6, h: 3.9, barDir: "col",
  chartColors: [HEX.accent3, HEX.accent2, HEX.accent6, HEX.accent1],
  valAxisMinVal: 0.84, valAxisMaxVal: 0.89, valAxisMajorUnit: 0.01, valAxisLabelFormatCode: "0.00",
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0000", dataLabelFontSize: 12,
  dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.accent3, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 10,
  valGridLine: { color: "D9DEE2", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
  showTitle: true, title: "Val Challenge_Avg, held-out after N4 + N5", titleFontSize: 13, titleColor: HEX.dk2,
  titleFontFace: "+mn-lt", objectName: "val-chart",
});
const notes4 = [
  ["0.530", "organizers' prior-only baseline"],
  ["0.638", "CLIP zero-shot with our text head, no training"],
  ["+0.017", "from calibration (N4) on Model B"],
];
notes4.forEach(([big, small], i) => {
  const y = 1.3 + i * 1.25;
  text(s, big, { x: 6.5, y, w: 3.0, h: 0.6, fontFace: THEME.headFontFace, fontSize: 30, bold: true, color: C.accent2,
    objectName: `val-stat-${i}` });
  text(s, small, { x: 6.5, y: y + 0.6, w: 3.0, h: 0.45, fontSize: 13, color: C.text2, objectName: `val-stat-label-${i}` });
});
s.addNotes("Held-out means calibration is fitted on one half of val and scored on the other. CLIP ViT-B/16 beats ConvNeXt by " +
  "1.7 points, ViT-L alone is level with ViT-B, and the ensemble of both is best at 0.8803.");

// ---------- 5. ablations ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Ablations: each component's gain", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{
  name: "gain (points)",
  labels: ["N5 decoding", "N1 text head", "N2 balancing", "N3 degradation*", "N4 calibration"],
  values: [0.07, 0.34, 0.44, 0.60, 1.67],
}], {
  x: 0.5, y: 1.2, w: 5.9, h: 3.9, barDir: "bar",
  chartColors: [HEX.accent2], valAxisMinVal: 0, valAxisMaxVal: 2, valAxisMajorUnit: 0.5,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.00", dataLabelFontSize: 12,
  dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.accent3, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 10,
  valGridLine: { color: "D9DEE2", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false,
  showTitle: true, title: "Challenge_Avg gain, in points (2-epoch CLIP ViT-B/16)", titleFontSize: 13, titleColor: HEX.dk2,
  titleFontFace: "+mn-lt", objectName: "ablation-chart",
});
card(s, 6.75, 1.3, 2.75, 3.6, C.background2, "ablation-takeaways");
text(s, [
  { text: "Calibration gives the most, because the 0.5 threshold is fixed.", options: { bullet: true, breakLine: true } },
  { text: "*N3 adds nothing on clean val, but 0.6 points on degraded val.", options: { bullet: true, breakLine: true } },
  { text: "Square-root sampling scored 0.15 points below equal sampling.", options: { bullet: true } },
], { x: 6.95, y: 1.5, w: 2.4, h: 3.2, fontSize: 14, color: C.text1, valign: "top", paraSpaceAfter: 8,
  objectName: "ablation-notes" });
s.addNotes("Each bar is the full recipe minus one component. Calibration is the largest gain. Degradation augmentation is " +
  "invisible on clean val, but worth 0.6 points when val images are downscaled and recompressed, which is closer to real CCTV.");

// ---------- 6. test result ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Hidden test set: accepted and scored", { placeholder: "title" });
const testStats = [["0.7699", "our test Challenge_Avg (sub03, CLIP ViT-B/16)", C.accent1],
  ["0.8021", "first place on the leaderboard", C.accent3],
  ["34.8 s", "to predict all 29,248 test images (300 s allowed)", C.accent2]];
testStats.forEach(([big, small, col], i) => {
  const y = 1.25 + i * 1.22;
  text(s, big, { x: 0.5, y, w: 3.4, h: 0.65, fontFace: THEME.headFontFace, fontSize: 34, bold: true, color: col,
    objectName: `test-stat-${i}` });
  text(s, small, { x: 0.5, y: y + 0.65, w: 3.4, h: 0.45, fontSize: 13, color: C.text2, objectName: `test-label-${i}` });
});
const hasShot = fs.existsSync(SHOT);
if (hasShot) {
  // leaderboard strip (1332 x 148 px): rows 47 (ali123) and 48 (fasihhk)
  s.addImage({ path: SHOT, x: 4.2, y: 1.3, w: 5.3, h: 5.3 * 148 / 1332,
    objectName: "sub03-screenshot", altText: "Codabench leaderboard rows 47 (ali123) and 48 (fasihhk), both 0.7699" });
}
const recRows = [
  ["Codabench record", ""],
  ["Competition", "18180, Evaluation Phase"],
  ["Submissions", "1004948 (ali123), 1005565 (fasihhk)"],
  ["Package", "sub03_clip_b16_calib.zip"],
  ["Status", "Finished"],
  ["mA / instance F1", "0.7725 / 0.7672"],
  ["Rank", "47 of 50 listed entries"],
].map((r, i) => r.map((c, j) => ({ text: c, options: {
  bold: i === 0 || j === 0, color: i === 0 ? HEX.lt1 : HEX.dk1, fill: { color: i === 0 ? HEX.dk2 : (i % 2 ? HEX.lt2 : HEX.lt1) },
} })));
s.addTable(recRows, { x: 4.2, y: hasShot ? 2.15 : 1.3, w: 5.3, colW: [1.7, 3.6], fontSize: 13, rowH: 0.38, margin: 0.06,
  border: { type: "solid", pt: 0.5, color: "D9DEE2" }, objectName: "sub03-record" });
s.addNotes("Our CLIP ViT-B/16 submission was scored at 0.7699 on the hidden test set, against 0.874 on val. " +
  "The gap shows the test images come from a different domain. Inference took 35 seconds of the 5-minute budget.");

// ---------- 7. demo ----------
pres.addSection({ title: "Demo" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demo" });
s.addText("Demo: from training to a valid submission", { placeholder: "title" });
card(s, 0.5, 1.25, 5.6, 3.7, C.text1, "terminal");
text(s, [
  { text: "# 1  train (tqdm bar shows time left)", options: { color: HEX.accent1, breakLine: true } },
  { text: "python src/train.py --config configs/clip_b16.yaml", options: { breakLine: true } },
  { text: "# 2  fit N4 offsets + N5 groups", options: { color: HEX.accent1, breakLine: true } },
  { text: "python src/calibrate.py --run results/clip_b16", options: { breakLine: true } },
  { text: "# 3  export one TorchScript file", options: { color: HEX.accent1, breakLine: true } },
  { text: "python src/export.py --run results/clip_b16 --no-flip", options: { breakLine: true } },
  { text: "# 4  replay Codabench locally", options: { color: HEX.accent1, breakLine: true } },
  { text: "python tools/simulate_ingestion.py --package submission", options: { breakLine: true } },
  { text: "# 5  zip for upload", options: { color: HEX.accent1, breakLine: true } },
  { text: "python tools/package.py --out upload/sub03.zip" },
], { x: 0.7, y: 1.4, w: 5.25, h: 3.4, fontFace: "Courier New", fontSize: 11, color: HEX.lt1, valign: "top",
  paraSpaceAfter: 4, objectName: "commands" });
const demo = [["Simulator", "checks the output format, the score on val, and projects the run time for 30k images"],
  ["Per-run folder", "results/<run>/ holds the config, train.log, metrics, calib.json and checkpoints"],
  ["Upload", "every zip in upload/ is ready for Codabench"]];
demo.forEach(([h, d], i) => {
  const y = 1.3 + i * 1.22;
  text(s, h, { x: 6.45, y, w: 3.05, h: 0.4, fontSize: 16, bold: true, color: C.accent2, objectName: `demo-h-${i}` });
  text(s, d, { x: 6.45, y: y + 0.4, w: 3.05, h: 0.75, fontSize: 13, color: C.text2, valign: "top",
    objectName: `demo-d-${i}` });
});
s.addNotes("Live demo: run the simulator on the exported package. It loads run.py exactly as Codabench does, predicts val images, " +
  "scores them with the challenge metric and projects the run time. Then show results/clip_b16 and the upload folder.");

// ---------- 8. outage timeline ----------
pres.addSection({ title: "Wrap-up" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wrap-up" });
s.addText("More models were ready, then Codabench went down", { placeholder: "title" });
const events = [
  ["20 Oct 09:31", "sub03 scored: 0.7699", C.accent2],
  ["20 Oct 18:20", "sub05 (train+val) predicted in 46 s, never scored", C.accent3],
  ["21 Oct 13:40", "sub07 (ViT-B + ViT-L, val 0.8803) queued", C.accent1],
  ["21 Oct 13:45", "codabench.org stops responding", C.accent4],
  ["22 Oct 12:00", "phase closes; site still down", C.accent4],
];
s.addShape(pres.shapes.LINE, { x: 1.3, y: 2.25, w: 7.4, h: 0, line: { color: C.accent3, width: 2 }, objectName: "timeline" });
events.forEach(([when, what, col], i) => {
  const x = 1.3 + i * 1.85;
  s.addShape(pres.shapes.OVAL, { x: x - 0.14, y: 2.11, w: 0.28, h: 0.28, fill: { color: col }, line: { color: C.background1, width: 2 },
    objectName: `event-dot-${i}` });
  text(s, when, { x: x - 0.8, y: 1.5, w: 1.6, h: 0.4, fontSize: 13, bold: true, color: C.text1, align: "center",
    objectName: `event-when-${i}` });
  text(s, what, { x: x - 0.85, y: 2.6, w: 1.7, h: 1.0, fontSize: 12, color: C.text2, align: "center", valign: "top",
    objectName: `event-what-${i}` });
});
card(s, 0.5, 3.85, 9.0, 1.05, C.background2, "unscored");
text(s, [
  { text: "Built and checked but never scored:  ", options: { bold: true, color: C.text1 } },
  { text: "sub05 ViT-B on train+val  ·  sub06 ViT-L/14  ·  sub07 ViT-B + ViT-L ensemble  ·  sub08 ensemble on train+val",
    options: { color: C.text2 } },
], { x: 0.7, y: 3.95, w: 8.6, h: 0.85, fontSize: 14, valign: "middle", objectName: "unscored-text" });
s.addNotes("After sub03 we trained ViT-L/14 and an ensemble that scored 0.8803 on val. sub05's predictions finished on the platform " +
  "but were never scored, and from 21 October 13:45 UTC the site stopped responding until after the deadline. So sub03 is our only test score.");

// ---------- 9. challenges & improvements ----------
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Wrap-up" });
s.addText("Challenges and next steps", { placeholder: "title" });
const cols = [
  ["Challenges", C.accent4, [
    "Val 0.874 vs test 0.770: the test images are a different domain",
    "Market1501 overfits after epoch 1 with equal sampling",
    "One 8 GB GPU: ViT-L only with frozen lower blocks",
    "Queue stalls, then the site went down",
  ]],
  ["Next steps", C.accent2, [
    "Fit offsets leaving one dataset out; shrink them (λ < 1)",
    "Stronger colour and style augmentation for new domains",
    "Fully fine-tuned ViT-L and larger ensembles",
    "Score the train+val models (sub05, sub08)",
  ]],
];
cols.forEach(([head, col, items], i) => {
  const x = 0.5 + i * 4.6;
  card(s, x, 1.25, 4.4, 3.65, C.background2, `col-card-${i}`);
  text(s, head, { x: x + 0.25, y: 1.4, w: 3.9, h: 0.5, fontSize: 20, bold: true, color: col, objectName: `col-head-${i}` });
  text(s, items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1 } })),
    { x: x + 0.25, y: 2.0, w: 3.9, h: 2.75, fontSize: 14, color: C.text1, valign: "top", paraSpaceAfter: 8,
      objectName: `col-items-${i}` });
});
s.addNotes("The biggest lesson is that val did not predict test: offsets fitted on val push rare colours hard, which may not " +
  "transfer. Next we would fit them leaving one dataset out and train with stronger augmentation for new domains.");

// ---------- 10. close ----------
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Wrap-up" });
s.addText("Thank you", { placeholder: "title" });
s.addText([
  { text: "Valid submission scored 0.7699 on the hidden test set; best ensemble 0.8803 on val", options: { breakLine: true } },
  { text: "Code, results, report and proof of submission are in the repository", options: { color: C.background2, fontSize: 16 } },
], { placeholder: "body" });
s.addNotes("Questions.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  const applyThemePath = process.argv[2];
  if (applyThemePath) {
    const { applyTheme } = require(applyThemePath);
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT);
})();
