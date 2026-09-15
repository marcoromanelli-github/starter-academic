import fs from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";

const ROOT = "/workspace/scratch/6cfe5cc6fb4a";
const OUT = path.join(ROOT, "output", "backprop_gradient_frames");
await fs.mkdir(OUT, { recursive: true });

const W = 1280;
const H = 720;
const N = 84;

const C = {
  bg: "#F5F7FA",
  navy: "#152238",
  muted: "#6B778C",
  faint: "#DDE4EE",
  line: "#AEBBD0",
  blue: "#2878FF",
  blueSoft: "#DDEBFF",
  orange: "#F28B20",
  orangeSoft: "#FFF0DF",
  green: "#11A66A",
  greenSoft: "#DFF7EC",
  purple: "#7546FF",
  white: "#FFFFFF",
};

const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const smooth = (x) => {
  x = clamp(x);
  return x * x * (3 - 2 * x);
};
const phase = (f, a, b) => smooth((f - a) / (b - a));
const esc = (s) => String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const lerp = (a, b, t) => a + (b - a) * t;
const hexMix = (a, b, t) => {
  const pa = [1, 3, 5].map(i => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map(i => parseInt(b.slice(i, i + 2), 16));
  return `#${pa.map((v, i) => Math.round(lerp(v, pb[i], t)).toString(16).padStart(2, "0")).join("")}`;
};

function text(x, y, value, size, color = C.navy, weight = 400, anchor = "start", opacity = 1, italic = false) {
  return `<text x="${x}" y="${y}" font-family="Arial, Nimbus Sans, DejaVu Sans, sans-serif" font-size="${size}" font-weight="${weight}" fill="${color}" text-anchor="${anchor}" opacity="${opacity}"${italic ? ' font-style="italic"' : ""}>${esc(value)}</text>`;
}

function pill(x, y, w, h, label, fill, stroke, color, opacity = 1) {
  return `<g opacity="${opacity}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${h / 2}" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>${text(x + w / 2, y + h / 2 + 7, label, 19, color, 700, "middle")}</g>`;
}

function circle(x, y, r, fill, stroke, sw = 3, opacity = 1) {
  return `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" opacity="${opacity}"/>`;
}

function connector(x1, y1, x2, y2, color, sw, opacity = 1) {
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${sw}" stroke-linecap="round" opacity="${opacity}"/>`;
}

function pulse(x1, y1, x2, y2, t, color, r = 8, opacity = 1) {
  const x = lerp(x1, x2, t);
  const y = lerp(y1, y2, t);
  return `<circle cx="${x}" cy="${y}" r="${r + 4}" fill="${color}" opacity="${0.12 * opacity}"/><circle cx="${x}" cy="${y}" r="${r}" fill="${color}" opacity="${opacity}"/>`;
}

function timeline(active) {
  const labels = ["Predict", "Measure error", "Compute gradient", "Update weight", "Try again"];
  const xs = [82, 300, 540, 802, 1045];
  let s = "";
  for (let i = 0; i < labels.length; i++) {
    if (i < labels.length - 1) s += connector(xs[i] + 13, 674, xs[i + 1] - 13, 674, i < active ? C.green : C.line, 2);
    const color = i < active ? C.green : i === active ? C.blue : C.line;
    s += circle(xs[i], 674, 6, color, C.bg, 2);
    s += text(xs[i] + 14, 680, labels[i], 15, i <= active ? C.navy : C.muted, i === active ? 700 : 500);
  }
  return s;
}

function frameSvg(f) {
  const stage = f < 16 ? 0 : f < 30 ? 1 : f < 52 ? 2 : f < 68 ? 3 : 4;
  const stageTitles = [
    ["1. Make a prediction", "The connection turns an input into a prediction."],
    ["2. Measure the error", "The target tells us how far the prediction missed."],
    ["3. Compute the gradient", "The chain rule assigns this connection its share of the error."],
    ["4. Update the weight", "Gradient descent changes the weight in the direction that lowers the loss."],
    ["5. See what changed", "The stronger connection raises the prediction and reduces the loss."],
  ];

  const updateP = phase(f, 53, 65);
  const reviewP = phase(f, 69, 79);
  const w = lerp(0.50, 0.577, updateP);
  const a = stage < 4 ? lerp(0.599, 0.613, stage === 3 ? updateP : 0) : lerp(0.599, 0.613, reviewP);
  const loss = stage < 4 ? lerp(0.081, 0.075, stage === 3 ? updateP : 0) : lerp(0.081, 0.075, reviewP);
  const connColor = hexMix(C.blue, C.green, stage === 3 ? updateP : stage === 4 ? 1 : 0);
  const connWidth = lerp(5, 10, stage === 3 ? updateP : stage === 4 ? 1 : 0);

  let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect width="100%" height="100%" fill="${C.bg}"/>`;
  s += text(62, 58, "ONE WEIGHT, ONE STEP OF LEARNING", 30, C.navy, 800);
  s += text(62, 105, stageTitles[stage][0], 27, C.navy, 800);
  s += text(62, 135, stageTitles[stage][1], 18, C.muted, 400);

  // Main two-neuron diagram.
  s += text(250, 194, "INPUT", 13, C.muted, 700, "middle");
  s += text(785, 194, "OUTPUT NEURON", 13, C.muted, 700, "middle");
  s += connector(310, 302, 718, 302, connColor, connWidth);
  s += circle(260, 302, 48, C.white, C.blue, 3);
  s += circle(770, 302, 52, stage >= 3 ? C.greenSoft : C.white, stage >= 3 ? C.green : C.purple, 3);
  s += text(260, 295, "x", 25, C.navy, 700, "middle", 1, true);
  s += text(260, 326, "0.80", 21, C.blue, 700, "middle");
  s += text(770, 295, "â", 25, C.navy, 700, "middle", 1, true);
  s += text(770, 326, a.toFixed(2), 21, stage >= 3 ? C.green : C.purple, 700, "middle");
  s += pill(448, 242, 132, 42, `w = ${w.toFixed(2)}`, stage >= 3 ? C.greenSoft : C.blueSoft, stage >= 3 ? C.green : C.blue, C.navy);
  s += text(514, 337, `z = wx = ${(0.8 * w).toFixed(2)}`, 18, C.muted, 600, "middle");
  s += text(770, 377, "â = σ(z)", 18, C.muted, 600, "middle");

  if (stage === 0) {
    const t = clamp((f % 16) / 12);
    s += pulse(310, 302, 718, 302, t, C.blue, 8, t < 1 ? 1 : 0);
    s += pill(935, 248, 225, 108, "", C.white, C.faint, C.navy);
    s += text(1048, 280, "FORWARD PASS", 13, C.blue, 800, "middle");
    s += text(1048, 317, "0.80 × 0.50 = 0.40", 21, C.navy, 700, "middle");
    s += text(1048, 345, "σ(0.40) = 0.60", 21, C.navy, 700, "middle");
  }

  if (stage >= 1) {
    const op = stage === 1 ? phase(f, 16, 20) : 1;
    s += pill(696, 156, 148, 44, "target y = 1.00", C.orangeSoft, C.orange, C.navy, op);
    s += connector(770, 202, 770, 244, C.orange, 3, op);
    s += `<path d="M758 231 L770 245 L782 231" fill="none" stroke="${C.orange}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" opacity="${op}"/>`;
  }

  if (stage === 1) {
    const p = phase(f, 19, 27);
    s += pill(905, 240, 292, 126, "", C.orangeSoft, C.orange, C.navy, p);
    s += text(1051, 274, "SQUARED ERROR", 13, C.orange, 800, "middle", p);
    s += text(1051, 316, "L = ½(â − y)²", 26, C.navy, 700, "middle", p);
    s += text(1051, 350, "= ½(0.60 − 1.00)² = 0.081", 18, C.navy, 600, "middle", p);
  }

  if (stage >= 2) {
    const baseOp = stage === 2 ? phase(f, 30, 34) : 1;
    s += text(62, 445, "CHAIN RULE FOR THIS WEIGHT", 14, stage >= 3 ? C.green : C.orange, 800, "start", baseOp);
    s += text(62, 489, "∂L/∂w", 30, C.navy, 800, "start", baseOp);
    s += text(176, 489, "=", 30, C.navy, 500, "start", baseOp);
    const f1 = stage > 2 ? 1 : phase(f, 33, 38);
    const f2 = stage > 2 ? 1 : phase(f, 38, 43);
    const f3 = stage > 2 ? 1 : phase(f, 43, 48);
    s += pill(220, 448, 180, 58, "(â − y) = −0.40", C.orangeSoft, C.orange, C.navy, f1);
    s += text(411, 485, "×", 25, C.navy, 600, "middle", f2);
    s += pill(430, 448, 208, 58, "σ′(z) = 0.24", C.orangeSoft, C.orange, C.navy, f2);
    s += text(649, 485, "×", 25, C.navy, 600, "middle", f3);
    s += pill(668, 448, 130, 58, "x = 0.80", C.orangeSoft, C.orange, C.navy, f3);
    const resultOp = stage > 2 ? 1 : phase(f, 47, 51);
    s += text(823, 489, "= −0.077", 31, stage >= 3 ? C.green : C.orange, 800, "start", resultOp);
    s += text(62, 539, "error signal", 15, C.muted, 500, "start", f1);
    s += text(460, 539, "neuron slope", 15, C.muted, 500, "start", f2);
    s += text(696, 539, "input carried by the connection", 15, C.muted, 500, "start", f3);
  }

  if (stage === 2) {
    const bt = clamp((f - 32) / 17);
    s += pulse(718, 302, 310, 302, bt, C.orange, 8, bt < 1 ? 1 : 0);
    if (f >= 47) {
      s += pill(925, 250, 250, 104, "", C.white, C.orange, C.navy, phase(f, 47, 50));
      s += text(1050, 286, "NEGATIVE GRADIENT", 13, C.orange, 800, "middle", phase(f, 47, 50));
      s += text(1050, 323, "Increase the weight", 23, C.navy, 800, "middle", phase(f, 47, 50));
      s += text(1050, 348, "to lower the loss", 17, C.muted, 500, "middle", phase(f, 47, 50));
    }
  }

  if (stage === 3) {
    const op = phase(f, 52, 55);
    s += `<rect x="62" y="565" width="870" height="66" rx="12" fill="${C.greenSoft}" stroke="${C.green}" stroke-width="1.5" opacity="${op}"/>`;
    s += text(86, 607, "wnew = w − η ∂L/∂w = 0.50 − 1(−0.077) = 0.577", 25, C.navy, 700, "start", op);
    const t = clamp((f - 53) / 12);
    s += pulse(310, 302, 718, 302, t, C.green, 9, t < 1 ? 1 : 0);
    s += pill(972, 268, 210, 78, "", C.greenSoft, C.green, C.navy, op);
    s += text(1077, 299, "WEIGHT GROWS", 13, C.green, 800, "middle", op);
    s += text(1077, 331, "0.50  →  0.58", 23, C.navy, 800, "middle", op);
  }

  if (stage === 4) {
    const op = phase(f, 68, 74);
    const t = clamp((f - 68) / 10);
    s += pulse(310, 302, 718, 302, t, C.green, 9, t < 1 ? 1 : 0);
    s += `<rect x="62" y="565" width="1115" height="66" rx="12" fill="${C.greenSoft}" stroke="${C.green}" stroke-width="1.5" opacity="${op}"/>`;
    s += text(86, 607, `new prediction: ${a.toFixed(2)}     new loss: ${loss.toFixed(3)}`, 26, C.navy, 700, "start", op);
    s += text(630, 607, "The prediction rises; the loss falls.", 23, C.green, 800, "start", op);
    s += pill(945, 242, 238, 120, "", C.white, C.green, C.navy, op);
    s += text(1064, 277, "AFTER ONE STEP", 13, C.green, 800, "middle", op);
    s += text(1064, 313, "â: 0.60  →  0.61", 22, C.navy, 800, "middle", op);
    s += text(1064, 344, "L: 0.081  →  0.075", 20, C.navy, 700, "middle", op);
  }

  s += timeline(stage);
  s += `</svg>`;
  return s;
}

for (let f = 0; f < N; f++) {
  const svg = frameSvg(f);
  await sharp(Buffer.from(svg)).png().toFile(path.join(OUT, `gradient-${String(f).padStart(3, "0")}.png`));
}

console.log(`Generated ${N} frames in ${OUT}`);
