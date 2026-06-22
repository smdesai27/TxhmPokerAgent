// Concise assertion-evidence deck for the AlphaHoldem-inspired RL project.
// One terse claim + one visual per slide. Reuses docs/figures/deck/*.png.
// Run from presentation/:  node build_deck.js
const pptxgen = require("pptxgenjs");
const fs = require("fs");

const INK = "15233B", INK2 = "0F1B30", MUTED = "5B6B82", LINE = "E6EAF0";
const SOFT = "F5F8FC", WHITE = "FFFFFF", ICE = "9DB8E6", SUBTXT = "C9D6EC";
const BLUE = "2563EB", TEAL = "0D9488", RED = "DC2626", AMBER = "B7791F", GREEN = "15803D";
const HEAD = "Georgia", BODY = "Calibri";

const W = 13.33, H = 7.5, LX = 0.62, CW = W - 2 * LX;
const cardSh = () => ({ type: "outer", color: "9AABC4", blur: 7, offset: 2, angle: 90, opacity: 0.18 });

const pres = new pptxgen();
pres.defineLayout({ name: "W", width: W, height: H });
pres.layout = "W";
pres.author = "Sanil Desai";
pres.title = "AlphaHoldem-inspired self-play RL for Heads-Up No-Limit Hold'em";

function pngSize(p) { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }

function footer(slide, n, dark) {
  const c = dark ? "7E94BC" : MUTED;
  if (!dark) slide.addShape(pres.shapes.LINE, { x: LX, y: 6.96, w: CW, h: 0, line: { color: LINE, width: 1 } });
  slide.addText("Sanil Desai  ·  AlphaHoldem-inspired self-play RL for Heads-Up No-Limit Hold'em",
    { x: LX, y: 7.04, w: 10, h: 0.3, fontFace: BODY, fontSize: 9.5, color: c, align: "left", margin: 0 });
  slide.addText(String(n), { x: W - 1.3, y: 7.04, w: 0.7, h: 0.3, fontFace: BODY, fontSize: 9.5, color: c, align: "right", margin: 0 });
}

function content(kickerText, kickerColor, claim, n, subline) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  s.addShape(pres.shapes.OVAL, { x: LX, y: 0.585, w: 0.12, h: 0.12, fill: { color: kickerColor } });
  s.addText(kickerText.toUpperCase(), { x: LX + 0.22, y: 0.5, w: CW - 0.22, h: 0.3, fontFace: BODY, bold: true, fontSize: 12, color: kickerColor, charSpacing: 2, margin: 0 });
  s.addText(claim, { x: LX, y: 0.92, w: CW, h: 1.1, fontFace: HEAD, bold: true, fontSize: 24, color: INK, align: "left", valign: "top", lineSpacingMultiple: 1.02, margin: 0 });
  if (subline) s.addText(subline, { x: LX, y: 2.02, w: CW, h: 0.42, fontFace: BODY, fontSize: 13, color: MUTED, align: "left", valign: "top", margin: 0 });
  footer(s, n, false);
  return s;
}

function figure(slide, file, top, bottom) {
  const { w, h } = pngSize(file); const ar = w / h;
  const maxH = bottom - top; let dw = CW, dh = dw / ar;
  if (dh > maxH) { dh = maxH; dw = dh * ar; }
  const x = (W - dw) / 2, y = top + (maxH - dh) / 2;
  slide.addImage({ path: file, x, y, w: dw, h: dh });
}

const FIG = (f) => `../docs/figures/deck/${f}`;

// 1 — TITLE
{
  const s = pres.addSlide();
  s.background = { path: "assets/hero_bg.png" };
  s.addText("SELF-PLAY REINFORCEMENT LEARNING  ·  IMPERFECT-INFORMATION GAMES",
    { x: 1.0, y: 1.7, w: 11.3, h: 0.4, fontFace: BODY, bold: true, fontSize: 13, color: ICE, charSpacing: 2, align: "center", margin: 0 });
  s.addText("AlphaHoldem-inspired self-play RL\nfor Heads-Up No-Limit Hold'em",
    { x: 0.8, y: 2.3, w: 11.7, h: 1.9, fontFace: HEAD, bold: true, fontSize: 42, color: WHITE, align: "center", lineSpacingMultiple: 1.04, margin: 0 });
  s.addText("From-scratch PyTorch self-play agent  ·  evaluation harness with behavior gates  ·  exact-exploitability study on Leduc",
    { x: 1.2, y: 4.45, w: 10.9, h: 0.7, fontFace: BODY, fontSize: 16, color: SUBTXT, align: "center", lineSpacingMultiple: 1.1, margin: 0 });
  s.addText("Sanil Desai     ·     PyTorch · OpenSpiel     ·     single GPU",
    { x: 1.0, y: 5.85, w: 11.3, h: 0.4, fontFace: BODY, fontSize: 13.5, color: ICE, align: "center", margin: 0 });
}

// 2 — ARCHITECTURE
{
  const s = content("Architecture", BLUE, "Pseudo-siamese two-tower actor–critic, ~1.5M parameters.", 2,
    "PPO + GAE  ·  K-best self-play league  ·  FCPA (4) / FCHPA (5) abstraction  ·  OpenSpiel  ·  single GPU");
  figure(s, FIG("01_architecture.png"), 2.6, 6.78);
}

// 3 — SCOPE (table)
{
  const s = content("Scope", AMBER, "Inspired, not reproduced.", 3);
  const hopt = { fill: { color: INK }, color: WHITE, bold: true, fontFace: BODY, fontSize: 13, valign: "middle" };
  const dev = { color: RED }, ok = { color: GREEN, bold: true };
  const rows = [
    [{ text: "Axis", options: hopt }, { text: "AlphaHoldem (paper)", options: hopt }, { text: "This repo", options: hopt }],
    ["Card / action input", "multi-channel tensors → ConvNets", { text: "53-token embedding → MLP / LSTM", options: dev }],
    ["Loss", "Trinal-Clip PPO", { text: "clipped PPO + Huber value", options: dev }],
    ["Self-play", "top-K by Elo", { text: "top-K rolling + rating-temperature sampling", options: dev }],
    ["Game scope", "abstraction-free full game", { text: "FCPA (4) / FCHPA (5)", options: dev }],
    ["Inference", "search-free, single forward pass", { text: "search-free, single forward pass ✓", options: ok }],
    ["Parameters", "~8.6M", { text: "~1.5M", options: dev }],
  ];
  s.addTable(rows, {
    x: LX, y: 2.45, w: CW, colW: [2.7, 4.6, CW - 2.7 - 4.6],
    rowH: 0.52, fontFace: BODY, fontSize: 13, color: INK, valign: "middle",
    border: { type: "solid", pt: 1, color: LINE }, align: "left", margin: [2, 6, 2, 6], fill: { color: WHITE },
  });
}

// 4 — EVALUATION (fig 3)
{
  const s = content("Evaluation", TEAL, "Behavior gates gate promotion — and caught a policy collapse.", 4,
    "Envelopes on fold frequency, aggression, entropy, and bet-size mix. A later 49k run scored well on win-rate but collapsed to fold-heavy play; the gates blocked it.");
  figure(s, FIG("03_behavior_gates.png"), 2.6, 6.78);
}

// 5 — BASELINE (retired number)
{
  const s = content("Baseline", RED, "“+897 bb/100 vs MCCFR-ES” is vs a near-random control.", 5);
  const y = 2.85, gap = 0.34, w = (CW - 2 * gap) / 3;
  const cells = [["+970", "vs 0-iteration uniform (random)", RED], ["+898", "vs 5k-iteration “solver”", MUTED], ["+934", "vs 10k-iteration “solver”", MUTED]];
  cells.forEach((c, i) => {
    const x = LX + i * (w + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.7, rectRadius: 0.08, fill: { color: SOFT }, line: { color: LINE, width: 1 } });
    s.addText(c[0], { x, y: y + 0.2, w, h: 0.8, fontFace: HEAD, bold: true, fontSize: 38, color: c[2], align: "center", margin: 0 });
    s.addText(c[1] + "  ·  bb/100", { x: x + 0.2, y: y + 1.04, w: w - 0.4, h: 0.5, fontFace: BODY, fontSize: 12.5, color: MUTED, align: "center", valign: "top", margin: 0 });
  });
  s.addText("The uniform (0-iteration) control scores highest — adding “solver” iterations makes the opponent easier, not harder. The MCCFR-ES baseline is near-random; the figure is a sanity check, not strength, and is never placed next to AlphaHoldem’s +11 mbb/h.",
    { x: LX, y: 5.0, w: CW, h: 1.4, fontFace: BODY, fontSize: 15, color: INK, align: "left", valign: "top", lineSpacingMultiple: 1.08, margin: 0 });
}

// 6 — EXPLOITABILITY (fig 4)
{
  const s = content("Exploitability", BLUE, "Leduc (exact NashConv): the iterate cycles; strategy-averaging plateaus ≈ 0.56.", 6,
    "HUNL exploitability is intractable (~10¹⁶¹ info sets), so the same recipe is probed on Leduc (936 states, exact).");
  figure(s, FIG("04_leduc_convergence.png"), 2.6, 6.78);
}

// 7 — CALIBRATION (fig 5)
{
  const s = content("Calibration", BLUE, "≈8× below random, ≈9× above tuned NFSP — methodology, not magnitude.", 7);
  figure(s, FIG("05_calibration_ladder.png"), 2.7, 6.4);
}

// 8 — ALPHAHOLDEM
{
  const s = content("AlphaHoldem", AMBER, "“Convergence” is not low exploitability.", 8);
  s.addText([
    { text: "AlphaHoldem reports loss flattening, Elo plateau, and win-rate (+111.6 mbb/h vs Slumbot). It never computes exploitability.", options: { bullet: true, breakLine: true } },
    { text: "Win-rate ≠ exploitability — Lisý & Bowling: two bots tied head-to-head (~20 mbb/g) yet ~1300 mbb/g apart in exploitability.", options: { bullet: true, breakLine: true } },
    { text: "The Leduc study is the test HUNL cannot run — not a failed reproduction.", options: { bullet: true } },
  ], { x: LX, y: 2.7, w: CW - 1.0, h: 3.4, fontFace: BODY, fontSize: 18, color: INK, align: "left", valign: "top", paraSpaceAfter: 14, lineSpacingMultiple: 1.1, margin: 0 });
}

// 9 — SCALED RUN (fig 2)
{
  const s = content("Scaled run", RED, "Regressed −511 bb/100 against the champion it started from.", 9,
    "90k GPU iterations (~9×10⁷ hands)  ·  vectorized collector (30–100×)  ·  PFSP fix  ·  Trinal-Clip  ·  warm-started from the 21k champion");
  figure(s, FIG("02_metric_mirage.png"), 2.65, 6.78);
}

// 10 — DIAGNOSIS (fig 6)
{
  const s = content("Diagnosis", RED, "Over-aggression drift, not collapse.", 10,
    "No calling opponents in the training mix; the co-evolving league learned to fold to aggression.");
  figure(s, FIG("06_over_aggression.png"), 2.6, 6.78);
}

// 11 — LIMITATIONS (closing, dark)
{
  const s = pres.addSlide();
  s.background = { path: "assets/hero_bg.png" };
  s.addShape(pres.shapes.OVAL, { x: LX, y: 1.0, w: 0.12, h: 0.12, fill: { color: ICE } });
  s.addText("LIMITATIONS", { x: LX + 0.22, y: 0.91, w: 11, h: 0.3, fontFace: BODY, bold: true, fontSize: 12, color: ICE, charSpacing: 2, margin: 0 });
  s.addText("Strong-but-exploitable, measured carefully.",
    { x: LX, y: 1.5, w: CW, h: 0.9, fontFace: HEAD, bold: true, fontSize: 30, color: WHITE, align: "left", margin: 0 });
  s.addText([
    { text: "Not near-Nash, not a record win-rate.", options: { bullet: true, breakLine: true } },
    { text: "Throughput gap vs AlphaHoldem (~2.7B hands, 8 GPUs, 3 days) is not closeable solo.", options: { bullet: true, breakLine: true } },
    { text: "The champion remains the strongest model; the scaled run was a negative result.", options: { bullet: true } },
  ], { x: LX, y: 2.9, w: CW - 1.0, h: 2.4, fontFace: BODY, fontSize: 17, color: SUBTXT, align: "left", valign: "top", paraSpaceAfter: 12, lineSpacingMultiple: 1.1, margin: 0 });
  s.addText("txhm-poker-api.onrender.com     ·     github.com/smdesai27/TxhmPokerAgent     ·     Sanil Desai",
    { x: LX, y: 6.7, w: CW, h: 0.4, fontFace: BODY, bold: true, fontSize: 13, color: ICE, align: "center", margin: 0 });
}

pres.writeFile({ fileName: "AlphaHoldem_RL_deck.pptx" }).then(f => console.log("wrote", f));
