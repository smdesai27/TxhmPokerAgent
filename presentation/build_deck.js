// Assertion-evidence portfolio deck for the AlphaHoldem-inspired RL project.
// Each slide = one full-sentence claim + one visual. Reuses docs/figures/*.png.
// Run from presentation/:  node build_deck.js
const pptxgen = require("pptxgenjs");
const fs = require("fs");

// ---- palette / type (matches the figures + the GitHub Pages site) ----
const INK = "15233B", INK2 = "0F1B30", MUTED = "5B6B82", LINE = "E6EAF0";
const SOFT = "F5F8FC", WHITE = "FFFFFF", ICE = "9DB8E6", SUBTXT = "C9D6EC";
const BLUE = "2563EB", TEAL = "0D9488", RED = "DC2626", AMBER = "B7791F", GREEN = "15803D";
const HEAD = "Georgia", BODY = "Calibri";

const W = 13.33, H = 7.5, LX = 0.62, CW = W - 2 * LX;
const sh = () => ({ type: "outer", color: "8AA0BE", blur: 9, offset: 3, angle: 90, opacity: 0.22 });
const cardSh = () => ({ type: "outer", color: "9AABC4", blur: 7, offset: 2, angle: 90, opacity: 0.20 });

const pres = new pptxgen();
pres.defineLayout({ name: "W", width: W, height: H });
pres.layout = "W";
pres.author = "Sanil Desai";
pres.title = "Knowing whether it was good was the hard half";

function pngSize(p) { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }

function footer(slide, n, dark) {
  const c = dark ? "7E94BC" : MUTED;
  if (!dark) slide.addShape(pres.shapes.LINE, { x: LX, y: 6.96, w: CW, h: 0, line: { color: LINE, width: 1 } });
  slide.addText("Sanil Desai  ·  AlphaHoldem-inspired self-play RL for Heads-Up No-Limit Hold'em",
    { x: LX, y: 7.04, w: 10, h: 0.3, fontFace: BODY, fontSize: 9.5, color: c, align: "left", margin: 0 });
  slide.addText(String(n), { x: W - 1.3, y: 7.04, w: 0.7, h: 0.3, fontFace: BODY, fontSize: 9.5, color: c, align: "right", margin: 0 });
}

function kicker(slide, text, color) {
  slide.addShape(pres.shapes.OVAL, { x: LX, y: 0.585, w: 0.12, h: 0.12, fill: { color } });
  slide.addText(text.toUpperCase(), { x: LX + 0.22, y: 0.5, w: CW - 0.22, h: 0.3, fontFace: BODY, bold: true, fontSize: 12, color, charSpacing: 2, align: "left", margin: 0 });
}

function assertion(slide, text, color) {
  slide.addText(text, { x: LX, y: 0.92, w: CW, h: 1.25, fontFace: HEAD, bold: true, fontSize: 25, color: color || INK, align: "left", valign: "top", lineSpacingMultiple: 1.02, margin: 0 });
}

// content slide scaffold -> returns body-top y
function content(kickerText, kickerColor, claim, n) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  kicker(s, kickerText, kickerColor);
  assertion(s, claim);
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

function card(slide, x, y, w, h, accent, tag, body) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: WHITE }, line: { color: LINE, width: 1 }, shadow: cardSh() });
  slide.addShape(pres.shapes.OVAL, { x: x + 0.28, y: y + 0.3, w: 0.16, h: 0.16, fill: { color: accent } });
  slide.addText(tag.toUpperCase(), { x: x + 0.54, y: y + 0.22, w: w - 0.7, h: 0.32, fontFace: BODY, bold: true, fontSize: 12.5, color: accent, charSpacing: 1, margin: 0 });
  slide.addText(body, { x: x + 0.28, y: y + 0.66, w: w - 0.56, h: h - 0.86, fontFace: BODY, fontSize: 13.5, color: INK, align: "left", valign: "top", lineSpacingMultiple: 1.05, margin: 0 });
}

function stat(slide, x, y, w, big, label, color) {
  slide.addText(big, { x, y, w, h: 0.95, fontFace: HEAD, bold: true, fontSize: 40, color, align: "left", margin: 0 });
  slide.addText(label, { x, y: y + 0.92, w, h: 0.7, fontFace: BODY, fontSize: 13, color: MUTED, align: "left", valign: "top", margin: 0 });
}

const FIG = (f) => `../docs/figures/deck/${f}`;

// ===================================================================== //
// 1 — TITLE
// ===================================================================== //
{
  const s = pres.addSlide();
  s.background = { path: "assets/hero_bg.png" };
  s.addText("SELF-PLAY REINFORCEMENT LEARNING  ·  IMPERFECT-INFORMATION GAMES",
    { x: 1.0, y: 1.45, w: 11.3, h: 0.4, fontFace: BODY, bold: true, fontSize: 13, color: ICE, charSpacing: 2, align: "center", margin: 0 });
  s.addText("Knowing whether it was good\nwas the hard half",
    { x: 0.8, y: 2.0, w: 11.7, h: 2.0, fontFace: HEAD, bold: true, fontSize: 46, color: WHITE, align: "center", lineSpacingMultiple: 1.04, margin: 0 });
  s.addText("An AlphaHoldem-inspired self-play RL agent for Heads-Up No-Limit Hold'em —\nand the evaluation discipline that kept catching my “better” models being worse.",
    { x: 1.6, y: 4.25, w: 10.1, h: 1.0, fontFace: BODY, italic: true, fontSize: 17.5, color: SUBTXT, align: "center", lineSpacingMultiple: 1.1, margin: 0 });
  s.addText("Sanil Desai     ·     PyTorch · OpenSpiel     ·     trained on a university GPU cluster",
    { x: 1.0, y: 5.95, w: 11.3, h: 0.4, fontFace: BODY, fontSize: 13.5, color: ICE, align: "center", margin: 0 });
}

// ===================================================================== //
// 2 — TL;DR
// ===================================================================== //
{
  const s = content("The short version", BLUE, "Building the agent was the easy half. Knowing whether it was good\nwas the hard half.", 2);
  const y = 2.75, h = 2.85, gap = 0.34, w = (CW - 2 * gap) / 3;
  card(s, LX, y, w, h, BLUE, "What I built",
    "A complete, from-scratch deep-RL system for HUNL — PPO + GAE, a K-best self-play league, an FCPA→FCHPA curriculum, an automated eval harness, and a deployed play-against-it demo. All on a single GPU.");
  card(s, LX + w + gap, y, w, h, TEAL, "The honest result",
    "A balanced, gate-passing champion (entropy 1.44 bits). And an evaluation harness that repeatedly caught my “improved” models quietly regressing — when the training curves said they were winning.");
  card(s, LX + 2 * (w + gap), y, w, h, RED, "The honest gap",
    "Not a Slumbot-beater, not near-Nash. Strong-but-exploitable, measured carefully. The deliverable is the measurement discipline — the value is the method, not a magnitude.");
}

// ===================================================================== //
// 3 — WHAT I BUILT
// ===================================================================== //
{
  const s = content("The system", BLUE, "A complete, end-to-end self-play RL system — built from scratch in PyTorch.", 3);
  const items = [
    [BLUE, "Pseudo-siamese net", "Two-tower actor–critic, ~1.5M params, search-free single forward pass."],
    [TEAL, "PPO + GAE", "Clipped policy loss, Huber value, entropy regularization, KL-adaptive LR."],
    [BLUE, "K-best self-play league", "Rolling snapshot pool, Elo-rated, prioritized opponent sampling."],
    [AMBER, "FCPA → FCHPA curriculum", "4- then 5-action betting abstractions over OpenSpiel universal_poker."],
    [TEAL, "Evaluation harness", "Multi-seed Student-t CIs, behavior gates, duplicate mirrored-hand eval."],
    [RED, "Served & deployed", "FastAPI inference backend + a browser demo you can play against."],
  ];
  const cols = 3, gap = 0.32, w = (CW - (cols - 1) * gap) / cols, h = 1.78, y0 = 2.5;
  items.forEach((it, i) => {
    const r = Math.floor(i / cols), c = i % cols;
    card(s, LX + c * (w + gap), y0 + r * (h + 0.3), w, h, it[0], it[1], it[2]);
  });
}

// ===================================================================== //
// 4 — ARCHITECTURE (figure)
// ===================================================================== //
{
  const s = content("The network", BLUE, "A pseudo-siamese two-tower actor–critic — search-free, one masked forward pass per decision.", 4);
  figure(s, FIG("01_architecture.png"), 2.25, 6.75);
}

// ===================================================================== //
// 5 — INSPIRED, NOT REPRODUCED (table)
// ===================================================================== //
{
  const s = content("Honest scope", AMBER, "This is AlphaHoldem-inspired — not a reproduction. I name every deviation.", 5);
  const hopt = { fill: { color: INK }, color: WHITE, bold: true, fontFace: BODY, fontSize: 13, valign: "middle" };
  const dev = { color: RED }, ok = { color: GREEN, bold: true };
  const rows = [
    [{ text: "Axis", options: hopt }, { text: "AlphaHoldem (paper)", options: hopt }, { text: "This repo", options: hopt }],
    ["Card / action input", "multi-channel tensors → ConvNets", { text: "53-token embedding → MLP / LSTM", options: dev }],
    ["Loss", "Trinal-Clip PPO", { text: "vanilla clipped PPO + Huber value", options: dev }],
    ["Self-play", "top-K by Elo", { text: "top-K rolling + rating-temperature sampling", options: dev }],
    ["Game scope", "abstraction-free full game", { text: "FCPA (4) / FCHPA (5) betting abstraction", options: dev }],
    ["Inference", "search-free, single forward pass", { text: "search-free, single forward pass ✓", options: ok }],
    ["Parameters", "~8.6M", { text: "~1.5M", options: dev }],
  ];
  s.addTable(rows, {
    x: LX, y: 2.5, w: CW, colW: [2.7, 4.6, CW - 2.7 - 4.6],
    rowH: 0.52, fontFace: BODY, fontSize: 13, color: INK, valign: "middle",
    border: { type: "solid", pt: 1, color: LINE }, align: "left", margin: [2, 6, 2, 6],
    fill: { color: WHITE },
  });
}

// ===================================================================== //
// 6 — HARNESS CATCHES COLLAPSE (figure)
// ===================================================================== //
{
  const s = content("Did it actually work?", TEAL, "The harness promotes on behavior gates, not win-rate — and it caught a policy collapse.", 6);
  figure(s, FIG("03_behavior_gates.png"), 2.3, 6.75);
}

// ===================================================================== //
// 7 — THE RETIRED NUMBER (stat callout)
// ===================================================================== //
{
  const s = content("A number I retired", RED, "My flashiest number measured the opponent's weakness — not my agent's strength.", 7);
  s.addText("Champion win-rate vs the in-house “MCCFR-ES solver”, by solver strength:",
    { x: LX, y: 2.45, w: CW, h: 0.4, fontFace: BODY, fontSize: 14, color: MUTED, margin: 0 });
  const y = 3.0, gap = 0.34, w = (CW - 2 * gap) / 3;
  const cells = [
    ["+970", "vs 0-iteration\nuniform (random)", RED],
    ["+898", "vs 5k-iteration\n“solver”", MUTED],
    ["+934", "vs 10k-iteration\n“solver”", MUTED],
  ];
  cells.forEach((c, i) => {
    const x = LX + i * (w + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.7, rectRadius: 0.08, fill: { color: SOFT }, line: { color: LINE, width: 1 } });
    s.addText(c[0], { x, y: y + 0.18, w, h: 0.8, fontFace: HEAD, bold: true, fontSize: 38, color: c[2], align: "center", margin: 0 });
    s.addText(c[1] + "  bb/100", { x: x + 0.2, y: y + 1.02, w: w - 0.4, h: 0.6, fontFace: BODY, fontSize: 12.5, color: MUTED, align: "center", valign: "top", margin: 0 });
  });
  s.addText([
    { text: "The 0-iteration (uniform-random) control scores the highest. ", options: { bold: true } },
    { text: "Adding “solver” iterations does not make the opponent harder — the MCCFR-ES baseline is near-random on most states. So I retired “+897 bb/100 vs a solver,” and never place it next to AlphaHoldem’s +11 mbb/h vs Slumbot.", options: {} },
  ], { x: LX, y: 5.05, w: CW, h: 1.7, fontFace: BODY, fontSize: 15, color: INK, align: "left", valign: "top", lineSpacingMultiple: 1.08, margin: 0 });
}

// ===================================================================== //
// 8 — NEED EXPLOITABILITY (stat + text)
// ===================================================================== //
{
  const s = content("The right yardstick", BLUE, "Win-rate is gameable; exploitability isn't — but on HUNL it's intractable to even measure.", 8);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: LX, y: 2.6, w: 4.3, h: 3.5, rectRadius: 0.08, fill: { color: INK }, shadow: sh() });
  s.addText("10¹⁶¹", { x: LX, y: 3.2, w: 4.3, h: 1.2, fontFace: HEAD, bold: true, fontSize: 60, color: WHITE, align: "center", margin: 0 });
  s.addText("information sets in full HUNL — exact best-response computation is prohibitive, so I make no exploitability claim about the HUNL agent.",
    { x: LX + 0.35, y: 4.5, w: 4.3 - 0.7, h: 1.4, fontFace: BODY, fontSize: 13.5, color: SUBTXT, align: "center", valign: "top", lineSpacingMultiple: 1.08, margin: 0 });
  const rx = LX + 4.3 + 0.5, rw = CW - 4.3 - 0.5;
  s.addText([
    { text: "Lisý & Bowling (LBR): ", options: { bold: true } },
    { text: "two bots tied head-to-head (within ~20 mbb/g) yet differed by ~1300 mbb/g in measured exploitability. A strong win-rate can hide a highly exploitable strategy.", options: { breakLine: true } },
    { text: "\n" },
    { text: "So I moved the question to ", options: {} },
    { text: "Leduc poker", options: { bold: true } },
    { text: " — 936 information states, where OpenSpiel computes exact NashConv — using the same PPO + self-play recipe as the HUNL configs.", options: {} },
  ], { x: rx, y: 2.7, w: rw, h: 3.4, fontFace: BODY, fontSize: 16, color: INK, align: "left", valign: "top", lineSpacingMultiple: 1.12, margin: 0 });
}

// ===================================================================== //
// 9 — LEDUC RESULT (figure)
// ===================================================================== //
{
  const s = content("The rigorous core", BLUE, "On Leduc, where exploitability is exact: the raw PPO iterate cycles; only strategy-averaging is non-divergent.", 9);
  figure(s, FIG("04_leduc_convergence.png"), 2.35, 6.75);
}

// ===================================================================== //
// 10 — METHOD NOT MAGNITUDE (figure)
// ===================================================================== //
{
  const s = content("Calibrated honestly", BLUE, "The 0.56 plateau is methodology, not a headline — ≈8× below random, but ≈9× above tuned NFSP.", 10);
  figure(s, FIG("05_calibration_ladder.png"), 2.55, 6.4);
}

// ===================================================================== //
// 11 — RECONCILING (two cards)
// ===================================================================== //
{
  const s = content("Reconciling with AlphaHoldem", AMBER, "AlphaHoldem's “convergence” and my finding measure different quantities — there is no contradiction.", 11);
  const y = 2.7, gap = 0.5, w = (CW - gap) / 2, h = 2.55;
  card(s, LX, y, w, h, BLUE, "What AlphaHoldem measured",
    "Training loss flattening · Elo plateauing in its self-play league · head-to-head win-rate (+111.6 mbb/h vs Slumbot). On full HUNL it explicitly concedes best-response is prohibitive and never computes a NashConv.");
  card(s, LX + w + gap, y, w, h, TEAL, "What exploitability measures",
    "The gain an optimal counter-strategy extracts — opponent-independent. As LBR shows, a strong head-to-head record is fully consistent with a strategy that remains highly exploitable.");
  s.addText([
    { text: "Win-rate ≠ exploitability. ", options: { bold: true, color: INK } },
    { text: "My Leduc study is the test HUNL can’t run — not a failed reproduction.", options: { color: MUTED } },
  ], { x: LX, y: 5.75, w: CW, h: 0.6, fontFace: BODY, fontSize: 15, align: "center", margin: 0 });
}

// ===================================================================== //
// 12 — SCALED ATTEMPT (stat + text)
// ===================================================================== //
{
  const s = content("Pushing for scale", BLUE, "I made self-play collection 30–100× faster, then ran 90k GPU iterations from the champion.", 12);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: LX, y: 2.6, w: 4.0, h: 3.5, rectRadius: 0.08, fill: { color: SOFT }, line: { color: LINE, width: 1 } });
  stat(s, LX + 0.4, 2.95, 3.3, "30–100×", "faster trajectory collection (batched inference vs one game at a time)", TEAL);
  stat(s, LX + 0.4, 4.55, 3.3, "90k iters", "~9×10⁷ hands · ~38 h on one GPU", BLUE);
  const rx = LX + 4.0 + 0.5, rw = CW - 4.0 - 0.5;
  s.addText([
    { text: "Rewrote the collector ", options: { bold: true } },
    { text: "to step B games in lockstep and batch all policy/value inference into a single forward pass — the GPU had been starved by Python-side game stepping.", options: { bullet: true, breakLine: true } },
    { text: "Fixed an inverted PFSP bug ", options: { bold: true } },
    { text: "(opponent sampling had favored the strongest snapshots) and switched the loss to Trinal-Clip.", options: { bullet: true, breakLine: true } },
    { text: "Warm-started from the Stage D 21k champion ", options: { bold: true } },
    { text: "and let it run — to see whether scale alone would push past a certified-good policy.", options: { bullet: true } },
  ], { x: rx, y: 2.7, w: rw, h: 3.4, fontFace: BODY, fontSize: 15.5, color: INK, align: "left", valign: "top", paraSpaceAfter: 8, margin: 0 });
}

// ===================================================================== //
// 13 — METRIC MIRAGE (figure) — money slide
// ===================================================================== //
{
  const s = content("The regression the evaluation caught", RED, "The training metric stayed positive the whole run; a fixed reference said it had regressed −511 bb/100.", 13);
  figure(s, FIG("02_metric_mirage.png"), 2.4, 6.75);
}

// ===================================================================== //
// 14 — DIAGNOSIS (figure)
// ===================================================================== //
{
  const s = content("What went wrong", RED, "Over-aggression drift, not collapse: it crushes random even harder than the champion — but a calling station punishes it.", 14);
  figure(s, FIG("06_over_aggression.png"), 2.45, 6.75);
}

// ===================================================================== //
// 15 — CLOSING (dark)
// ===================================================================== //
{
  const s = pres.addSlide();
  s.background = { path: "assets/hero_bg.png" };
  s.addShape(pres.shapes.OVAL, { x: LX, y: 0.92, w: 0.12, h: 0.12, fill: { color: ICE } });
  s.addText("WHAT IT DEMONSTRATES", { x: LX + 0.22, y: 0.83, w: 11, h: 0.3, fontFace: BODY, bold: true, fontSize: 12, color: ICE, charSpacing: 2, margin: 0 });
  s.addText("Choose ungameable metrics. Report honest negatives.\nBuild the evaluation discipline to know when you're fooling yourself.",
    { x: LX, y: 1.4, w: CW, h: 1.7, fontFace: HEAD, bold: true, fontSize: 29, color: WHITE, align: "left", lineSpacingMultiple: 1.06, margin: 0 });
  const y = 3.5, gap = 0.4, w = (CW - 2 * gap) / 3, h = 1.9;
  const items = [
    ["Ungameable metrics", "Exact exploitability on Leduc, fixed-reference ladders, and an LBR lower bound — chosen because HUNL's metric is intractable."],
    ["Honest negatives", "The iterate cycles, network-averaging fails, Trinal-Clip is inert at Leduc scale, the scaled run regressed −511 bb/100 — reported, not buried."],
    ["Evaluation discipline", "The harness caught what the training curves structurally couldn't see — and refused to promote regressions automatically."],
  ];
  items.forEach((it, i) => {
    const x = LX + i * (w + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.07, fill: { color: "16294A" }, line: { color: "31507F", width: 1 } });
    s.addText(it[0], { x: x + 0.26, y: y + 0.2, w: w - 0.5, h: 0.4, fontFace: BODY, bold: true, fontSize: 14.5, color: ICE, margin: 0 });
    s.addText(it[1], { x: x + 0.26, y: y + 0.66, w: w - 0.52, h: h - 0.85, fontFace: BODY, fontSize: 12, color: SUBTXT, align: "left", valign: "top", lineSpacingMultiple: 1.06, margin: 0 });
  });
  s.addText("The honest ceiling: strong-but-exploitable, measured rigorously — not near-Nash, not a record win-rate. What transfers is first-principles systems building, measurement rigor, and the discipline to publish the negative result.",
    { x: LX, y: 5.65, w: CW, h: 0.9, fontFace: BODY, italic: true, fontSize: 14, color: SUBTXT, align: "center", lineSpacingMultiple: 1.08, margin: 0 });
  s.addText("txhm-poker-api.onrender.com   ·   github.com/smdesai27/TxhmPokerAgent   ·   Sanil Desai",
    { x: LX, y: 6.75, w: CW, h: 0.4, fontFace: BODY, bold: true, fontSize: 13, color: ICE, align: "center", margin: 0 });
}

pres.writeFile({ fileName: "AlphaHoldem_RL_deck.pptx" }).then(f => console.log("wrote", f));
