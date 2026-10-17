"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import TopBar from "../components/TopBar";

// ─── Types (mirror of POST /explain) ─────────────────────────────────────────

interface RawTransaction {
  trans_date_trans_time: string;
  merchant: string;
  category: string;
  amt: number;
  city: string;
  state: string;
  lat: number;
  long: number;
  city_pop: number;
  job: string;
  dob: string;
  trans_num: string;
  merch_lat: number;
  merch_long: number;
}

interface Explain {
  raw: RawTransaction;
  validation: { required: string[]; present: string[] };
  engineered: {
    hour: number;
    day_of_week: number;
    month: number;
    age: number;
    distance_km: number;
    log_amt: number;
  };
  dropped: string[];
  encoding: { category: string; state: string; active: string[]; total: number };
  vector: { feature: string; value: number }[];
  model: { n_trees: number; learning_rate: number; max_depth: number; num_leaves: number };
  tree_path: {
    feature: string;
    threshold: number;
    value: number;
    is_numeric: boolean;
    go_left: boolean;
    samples: number;
  }[];
  tree_leaf: number;
  boosting: { trees: number; raw_score: number }[];
  base_value: number;
  raw_score: number;
  contributions: { feature: string; value: number }[];
  probability: number;
  threshold: number;
  is_fraud: number;
  verdict: string;
}

type Pt = { x: number; y: number };

// ─── Feedback loop / retraining (simulated MLOps process) ────────────────────

interface Metrics {
  precision: number;
  recall: number;
  f1: number;
  auc: number;
}
interface Labels {
  confirmed: number; // analyst confirmed the flagged payment was fraud
  false_positive: number; // analyst found the flagged payment was legit
  missed_fraud: number; // approved payment later disputed by the customer
}
type CaseLabel = keyof Labels | "no_complaint";
interface Nightly {
  used: Labels;
  candidate: Metrics;
  better: boolean;
  from: string;
  to: string;
  task: number;
  status: "running" | "compared" | "awaiting_ds" | "published" | "kept" | "rejected";
}

// Production benchmark = LightGBM test-set results from ML_work_and_analysis.ipynb (threshold 0.5)
const BASELINE: Metrics = { precision: 0.9257, recall: 0.8399, f1: 0.8807, auc: 0.9978 };
const HOLDOUT_TX = 67922;
const HOLDOUT_FRAUD = 356;
const TRAIN_ROWS = 271685;
const NO_LABELS: Labels = { confirmed: 0, false_positive: 0, missed_fraud: 0 };

const f1Of = (p: number, r: number) => (2 * p * r) / (p + r);
const totalLabels = (l: Labels) => l.confirmed + l.false_positive + l.missed_fraud;

/** Simulated nightly retrain: more (and more informative) labels → better candidate. */
function simulateCandidate(b: Metrics, u: Labels, seed: number): Metrics {
  const n = totalLabels(u);
  const jitter = (k: number) => (((Math.sin(seed * 9301 + k * 49297) + 1) / 2) - 0.5) * 0.002;
  let recall: number, precision: number;
  if (n < 3) {
    // too little new signal — retraining mostly adds noise
    recall = b.recall - 0.0028 + jitter(1);
    precision = b.precision - 0.0015 + jitter(2);
  } else {
    recall = b.recall + Math.min(0.03, 0.0028 * u.missed_fraud + 0.0008 * u.confirmed) + jitter(1);
    precision = b.precision + Math.min(0.03, 0.0021 * u.false_positive + 0.0004 * u.confirmed) + jitter(2);
  }
  recall = Math.min(0.99, recall);
  precision = Math.min(0.995, precision);
  const auc = Math.min(0.9995, b.auc + (n < 3 ? -0.0002 : 0.00012 * Math.sqrt(n)));
  return { precision, recall, f1: f1Of(precision, recall), auc };
}

const API = "http://127.0.0.1:5050";

// ─── Schematic geometry (SVG viewBox units) ──────────────────────────────────

const VB_W = 1240;
const VB_H = 870;
const NODE_W = 124;
const NODE_H = 104;
const ROW_Y = 118;
const CY = ROW_Y + NODE_H / 2; // 170 — the main data line
// Grid for terminals + feedback lanes (8 columns)
const gX = (col: number) => 36 + col * 152;
const C = (col: number) => gX(col) + NODE_W / 2;
// Main row: steps 01–06 + decision 07, spread so the decision sits above the fraud terminal
const MAIN_N = 6;
const DECISION = MAIN_N - 1;
// Feedback-loop stages follow right after the decision
const ST = {
  review: DECISION + 1,
  complaint: DECISION + 2,
  store: DECISION + 3,
  retrain: DECISION + 4,
  compare: DECISION + 5,
  task: DECISION + 6,
  publish: DECISION + 7,
  keep: DECISION + 8,
};
const MAIN_STEP = (C(7) - NODE_W / 2 - 36) / DECISION;
const nodeX = (i: number) => 36 + i * MAIN_STEP;
const nodeC = (i: number): Pt => ({ x: nodeX(i) + NODE_W / 2, y: CY });
const DIAMOND = { x: nodeC(DECISION).x, y: CY, r: 58 };
// Where the traced request rests inside a step (empty top-right corner, clear of the text)
const nodeStop = (i: number): Pt =>
  i === DECISION ? { x: DIAMOND.x, y: CY - DIAMOND.r + 22 } : { x: nodeX(i) + NODE_W - 18, y: ROW_Y + 18 };
const entryX = (i: number) => (i === DECISION ? DIAMOND.x - DIAMOND.r : nodeX(i));
const exitX = (i: number) => nodeX(i) + NODE_W;
const TERM_Y = 318;
const TERM_H = 74;
const TERM_BOT = TERM_Y + TERM_H;
const TERM_APPROVE = { x: gX(6), c: C(6) };
const TERM_FRAUD = { x: gX(7), c: C(7) };

const BRANCH: Record<"legit" | "fraud", Pt[]> = {
  fraud: [
    { x: DIAMOND.x, y: CY + DIAMOND.r },
    { x: TERM_FRAUD.c, y: TERM_Y + TERM_H / 2 },
  ],
  legit: [
    { x: DIAMOND.x, y: CY + DIAMOND.r },
    { x: DIAMOND.x, y: 262 },
    { x: TERM_APPROVE.c, y: 262 },
    { x: TERM_APPROVE.c, y: TERM_Y + TERM_H / 2 },
  ],
};
const START: Pt = { x: 0, y: CY };
const MAIN_PATH: Pt[] = [START, ...Array.from({ length: MAIN_N }, (_, i) => nodeC(i))];
const MODEL = 4; // index of the LightGBM step (trees + sigmoid)

// Feedback lane (light) and nightly lane (dark)
const L2 = { y: 420, end: 600 };
const L3 = { y: 612, end: 862 };
const BOX_H = 88;
const LOOP_BOX: Record<number, { x: number; y: number; h: number }> = {
  [ST.review]: { x: gX(7), y: 456, h: BOX_H }, // manual review
  [ST.complaint]: { x: gX(6), y: 456, h: BOX_H }, // customer complaint
  [ST.store]: { x: gX(5), y: 456, h: BOX_H }, // labelled feedback data
  [ST.retrain]: { x: gX(5), y: 650, h: BOX_H }, // OpenClaw retrain
  [ST.task]: { x: gX(3), y: 650, h: BOX_H }, // data scientist task
  [ST.publish]: { x: gX(2), y: 650, h: BOX_H }, // publish
  [ST.keep]: { x: gX(4), y: 790, h: 60 }, // keep current model
};
const CMP_STAGE = ST.compare;
const CMP = { x: C(4), y: 694, r: 50 }; // benchmark comparison diamond
const loopStop = (i: number): Pt =>
  i === CMP_STAGE ? { x: CMP.x, y: CMP.y - CMP.r + 18 } : { x: LOOP_BOX[i].x + NODE_W - 18, y: LOOP_BOX[i].y + 18 };

const EDGES: Record<string, Pt[]> = {
  "flag-review": [{ x: C(7), y: TERM_BOT }, { x: C(7), y: 456 }],
  "approve-complaint": [{ x: C(6), y: TERM_BOT }, { x: C(6), y: 456 }],
  "review-store": [
    { x: C(7), y: 544 },
    { x: C(7), y: 580 },
    { x: gX(5) + 94, y: 580 },
    { x: gX(5) + 94, y: 544 },
  ],
  "complaint-store": [{ x: gX(6), y: 500 }, { x: gX(5) + NODE_W, y: 500 }],
  "store-retrain": [{ x: gX(5) + 40, y: 544 }, { x: gX(5) + 40, y: 650 }],
  "retrain-compare": [{ x: gX(5), y: 694 }, { x: CMP.x + CMP.r, y: 694 }],
  "compare-task": [{ x: CMP.x - CMP.r, y: 694 }, { x: gX(3) + NODE_W, y: 694 }],
  "compare-keep": [{ x: CMP.x, y: CMP.y + CMP.r }, { x: CMP.x, y: 790 }],
  "task-publish": [{ x: gX(3), y: 694 }, { x: gX(2) + NODE_W, y: 694 }],
  "publish-model": [
    { x: C(2), y: 650 },
    { x: C(2), y: 272 },
    { x: nodeC(MODEL).x, y: 272 },
    { x: nodeC(MODEL).x, y: ROW_Y + NODE_H },
  ],
};
const EDGE_TONE: Record<string, "light" | "dark" | "cross"> = {
  "flag-review": "light",
  "approve-complaint": "light",
  "review-store": "light",
  "complaint-store": "light",
  "store-retrain": "cross",
  "retrain-compare": "dark",
  "compare-task": "dark",
  "compare-keep": "dark",
  "task-publish": "dark",
  "publish-model": "cross",
};
const STREAM_EDGES = ["flag-review", "approve-complaint", "review-store", "complaint-store"];

const PHASES = [
  { label: "Intake", from: 0, to: 1 },
  { label: "Preprocessing", from: 2, to: 3 },
  { label: "Model", from: MODEL, to: MODEL },
  { label: "Decision", from: DECISION, to: DECISION },
];

const STAGES = [
  { title: ["Transaction", "source"], desc: "A synthetic card transaction is generated from real dataset statistics and sent to the API." },
  { title: ["API &", "validation"], desc: "The API receives the JSON request and checks that every field the model needs is present." },
  { title: ["Feature", "engineering"], desc: "Raw fields are turned into signals the model was trained on: time, age, distance, log-amount." },
  { title: ["One-hot", "encoding"], desc: "Text categories (merchant category, state) become 0/1 columns aligned to the training layout." },
  { title: ["LightGBM"], desc: "300 gradient-boosted trees add up a raw score (log-odds); the sigmoid turns it into a fraud probability between 0 and 1." },
  { title: ["Decision", "p ≥ 0.5 ?"], desc: "The probability is compared to the 0.5 threshold and the final verdict is returned." },
  // ── feedback loop (simulated) ──
  { title: ["Manual", "review"], desc: "A fraud analyst reviews every flagged payment and labels it: confirmed fraud or false positive (actually legit)." },
  { title: ["Customer", "complaint?"], desc: "An approved payment can still turn out to be fraud — a customer complaint / chargeback labels it as missed fraud." },
  { title: ["Labelled", "feedback data"], desc: "Every human label is stored together with the transaction features as a new training example." },
  { title: ["OpenClaw", "nightly retrain"], desc: "At 02:00 the OpenClaw agent merges the new labels into the training set and retrains LightGBM." },
  { title: ["Better than", "benchmark?"], desc: "The candidate is evaluated on the same hold-out set and compared with the production benchmark." },
  { title: ["Data scientist", "review task"], desc: "If the candidate wins, OpenClaw opens a task for a data scientist to verify the result before anything ships." },
  { title: ["Publish", "new model"], desc: "After sign-off the new version is published and replaces the production model in the pipeline." },
  { title: ["Keep current", "model"], desc: "If the candidate is not better (or is rejected), production stays unchanged and the report is logged." },
];

const LABELS: Record<string, string> = {
  amt: "Amount",
  log_amt: "log(1+Amount)",
  hour: "Hour",
  day_of_week: "Day of week",
  month: "Month",
  age: "Age",
  distance_km: "Distance km",
  city_pop: "City population",
  lat: "Cardholder lat",
  long: "Cardholder long",
  merch_lat: "Merchant lat",
  merch_long: "Merchant long",
  category: "Category",
  state: "State",
};

const CASE_TEXT: Record<CaseLabel, string> = {
  confirmed: "confirmed fraud",
  false_positive: "false positive",
  missed_fraud: "missed fraud",
  no_complaint: "no complaint",
};

const DWELL = { slow: 2600, normal: 1500, fast: 500 } as const;
type Pace = keyof typeof DWELL;
const DONE = 99; // main trace finished

// ─── Helpers ─────────────────────────────────────────────────────────────────

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const pad = (i: number) => String(i + 1).padStart(2, "0"); // stage index → "01"

const pathLen = (pts: Pt[]) =>
  pts.slice(1).reduce((s, p, i) => s + Math.hypot(p.x - pts[i].x, p.y - pts[i].y), 0);

function pointAlong(pts: Pt[], t: number): Pt {
  const segs = pts.slice(1).map((p, i) => Math.hypot(p.x - pts[i].x, p.y - pts[i].y));
  const total = segs.reduce((a, b) => a + b, 0);
  let d = Math.min(Math.max(t, 0), 1) * total;
  for (let i = 0; i < segs.length; i++) {
    if (d <= segs[i] || i === segs.length - 1) {
      const k = segs[i] === 0 ? 1 : Math.min(d / segs[i], 1);
      return {
        x: pts[i].x + (pts[i + 1].x - pts[i].x) * k,
        y: pts[i].y + (pts[i + 1].y - pts[i].y) * k,
      };
    }
    d -= segs[i];
  }
  return pts[pts.length - 1];
}

const pathD = (pts: Pt[]) => pts.map((p, i) => `${i ? "L" : "M"}${p.x},${p.y}`).join(" ");
const pct = (p: number) =>
  p >= 0.01 ? `${(p * 100).toFixed(2)}%` : `${(p * 100).toFixed(4)}%`;
const pct1 = (p: number) => `${(p * 100).toFixed(2)}%`;
const signed = (v: number, d = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
const featLabel = (f: string, raw?: RawTransaction) =>
  f === "category" && raw
    ? `Category (${raw.category.replace("_", " ")})`
    : f === "state" && raw
    ? `State (${raw.state})`
    : LABELS[f] ?? f;

/** Short IN / OUT text shown on the travelling payload tag */
function payloadTag(i: number, raw: RawTransaction | null, d: Explain | null): [string, string] {
  switch (i) {
    case 0: return ["GET /generate", raw ? `JSON · ${Object.keys(raw).length} fields` : "…"];
    case 1: return ["POST /explain", d ? `${d.validation.present.length}/${d.validation.required.length} required ✓` : "…"];
    case 2: return ["time, dob, coords", d ? `+6 feat · dist ${d.engineered.distance_km}km` : "…"];
    case 3: return [raw ? `${raw.category.slice(0, 12)}, ${raw.state}` : "…", d ? `${d.encoding.total} cols · ${d.encoding.active.length} = 1` : "…"];
    case 4: return [d ? `vector[${d.vector.length}]` : "…", d ? `s ${signed(d.raw_score)} → p ${pct(d.probability)}` : "…"];
    case 5: return [d ? `p = ${pct(d.probability)}` : "…", d ? (d.is_fraud ? "FRAUD" : "LEGITIMATE") : "…"];
    default: return ["", ""];
  }
}

/** Live output summary written inside each main node once the packet has passed */
function nodeSummary(i: number, raw: RawTransaction | null, d: Explain | null): string {
  if (i === 0) return raw ? `$${raw.amt.toFixed(2)} · ${raw.state}` : "";
  if (!d) return "";
  switch (i) {
    case 1: return `${d.validation.present.length}/${d.validation.required.length} ✓`;
    case 2: return `${d.engineered.distance_km} km · ${d.engineered.hour}h`;
    case 3: return `${d.encoding.active.length} of ${d.encoding.total} = 1`;
    case 4: return `${signed(d.raw_score)} → ${pct(d.probability)}`;
    default: return "";
  }
}

// ─── Small UI pieces ─────────────────────────────────────────────────────────

function Block({ label, children, tone = "card" }: { label: string; children: React.ReactNode; tone?: "card" | "dark" }) {
  return (
    <div className={`p-4 ${tone === "dark" ? "bg-black text-white" : "bg-sand"}`}>
      <div
        className={`text-[10px] uppercase tracking-[0.14em] font-semibold mb-3 flex items-center gap-2 ${
          tone === "dark" ? "text-brand" : "text-black"
        }`}
      >
        <span className={`w-2 h-2 ${tone === "dark" ? "bg-brand" : "bg-black"}`} />
        {label}
      </div>
      {children}
    </div>
  );
}

function KV({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <div className="divide-y divide-line text-xs">
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-3 py-1.5">
          <span className="text-subtle font-mono">{k}</span>
          <span className="text-black font-medium text-right truncate">{v}</span>
        </div>
      ))}
    </div>
  );
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <pre className="bg-black text-white/90 font-mono text-[11px] leading-relaxed p-3 overflow-x-auto whitespace-pre">
      {children}
    </pre>
  );
}

function ActionButton({
  onClick,
  disabled,
  children,
  variant = "primary",
}: {
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
  variant?: "primary" | "outline" | "fraud" | "legit" | "dark";
}) {
  const cls = {
    primary: "bg-brand text-black hover:bg-black hover:text-brand",
    outline: "border border-black text-black hover:bg-black hover:text-white",
    fraud: "bg-fraud text-white hover:bg-black",
    legit: "bg-legit text-white hover:bg-black",
    dark: "bg-black text-brand hover:bg-neutral-800",
  }[variant];
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`h-10 px-4 font-medium text-[11px] uppercase tracking-[0.1em] transition-colors disabled:opacity-30 disabled:cursor-not-allowed ${cls}`}
    >
      {children}
    </button>
  );
}

function SimBadge() {
  return (
    <span className="inline-flex items-center gap-1.5 bg-purple text-white text-[10px] uppercase tracking-[0.12em] font-semibold px-2 py-1">
      Simulation
    </span>
  );
}

// ─── Charts ──────────────────────────────────────────────────────────────────

function BoostingChart({ data, final }: { data: Explain["boosting"]; final: number }) {
  const W = 360, H = 170, L = 40, R = 14, T = 14, B = 30;
  const ys = data.map((d) => d.raw_score).concat([0]);
  const yMin = Math.min(...ys) - 0.5, yMax = Math.max(...ys) + 0.5;
  const x = (t: number) => L + (t / 300) * (W - L - R);
  const y = (v: number) => T + ((yMax - v) / (yMax - yMin)) * (H - T - B);
  const [hover, setHover] = useState<number | null>(null);
  const line = data.map((d, i) => `${i ? "L" : "M"}${x(d.trees)},${y(d.raw_score)}`).join(" ");
  const h = hover !== null ? data[hover] : null;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" onMouseLeave={() => setHover(null)}>
      {/* zero line = threshold in log-odds space */}
      <line x1={L} x2={W - R} y1={y(0)} y2={y(0)} stroke="#000" strokeDasharray="4 3" />
      <text x={W - R} y={y(0) - 5} textAnchor="end" fontSize="9" fill="#000" fontWeight="600">
        score 0 = p 50% (threshold)
      </text>
      <line x1={L} x2={L} y1={T} y2={H - B} stroke="#e7e6e4" />
      <line x1={L} x2={W - R} y1={H - B} y2={H - B} stroke="#e7e6e4" />
      {[0, 100, 200, 300].map((t) => (
        <text key={t} x={x(t)} y={H - B + 14} textAnchor="middle" fontSize="9" fill="#808080">
          {t}
        </text>
      ))}
      <text x={(L + W - R) / 2} y={H - 3} textAnchor="middle" fontSize="9" fill="#525252">
        number of trees added
      </text>
      {[yMin + 0.5, yMax - 0.5].map((v) => (
        <text key={v} x={L - 6} y={y(v) + 3} textAnchor="end" fontSize="9" fill="#808080">
          {v.toFixed(1)}
        </text>
      ))}
      <path d={line} fill="none" stroke="#50327b" strokeWidth="2" />
      {data.map((d, i) => (
        <g key={d.trees} onMouseEnter={() => setHover(i)}>
          <circle cx={x(d.trees)} cy={y(d.raw_score)} r="12" fill="transparent" />
          <circle cx={x(d.trees)} cy={y(d.raw_score)} r={hover === i ? 5 : 3.5} fill="#50327b" stroke="#fff" strokeWidth="2" />
        </g>
      ))}
      <text x={x(300) - 8} y={y(final) - 8} textAnchor="end" fontSize="10" fontWeight="600" fill="#000">
        final {signed(final)}
      </text>
      {h && (
        <g transform={`translate(${Math.min(x(h.trees) + 8, W - 120)},${T})`}>
          <rect width="112" height="34" fill="#000" />
          <text x="8" y="14" fontSize="9" fill="#fff">after {h.trees} trees</text>
          <text x="8" y="27" fontSize="10" fill="#f7e948" fontWeight="600">score {signed(h.raw_score)}</text>
        </g>
      )}
    </svg>
  );
}

function SigmoidChart({ score, p }: { score: number; p: number }) {
  const W = 320, H = 160, L = 34, R = 10, T = 12, B = 28;
  const x = (s: number) => L + ((s + 12) / 24) * (W - L - R);
  const y = (v: number) => T + (1 - v) * (H - T - B);
  const curve = Array.from({ length: 97 }, (_, i) => {
    const s = -12 + i * 0.25;
    return `${i ? "L" : "M"}${x(s)},${y(1 / (1 + Math.exp(-s)))}`;
  }).join(" ");
  const sc = Math.max(-12, Math.min(12, score));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
      <line x1={L} x2={W - R} y1={y(0.5)} y2={y(0.5)} stroke="#000" strokeDasharray="4 3" />
      <text x={L + 4} y={y(0.5) - 5} fontSize="9" fontWeight="600">threshold 0.5</text>
      <line x1={L} x2={L} y1={T} y2={H - B} stroke="#e7e6e4" />
      <line x1={L} x2={W - R} y1={H - B} y2={H - B} stroke="#e7e6e4" />
      {[0, 1].map((v) => (
        <text key={v} x={L - 6} y={y(v) + 3} textAnchor="end" fontSize="9" fill="#808080">{v}</text>
      ))}
      {[-10, 0, 10].map((s) => (
        <text key={s} x={x(s)} y={H - B + 13} textAnchor="middle" fontSize="9" fill="#808080">{s}</text>
      ))}
      <text x={(L + W - R) / 2} y={H - 2} textAnchor="middle" fontSize="9" fill="#525252">raw score (log-odds)</text>
      <path d={curve} fill="none" stroke="#000" strokeWidth="2" />
      <line x1={x(sc)} x2={x(sc)} y1={y(p)} y2={H - B} stroke={p >= 0.5 ? "#c62828" : "#085850"} strokeDasharray="2 2" />
      <circle cx={x(sc)} cy={y(p)} r="6" fill="#f7e948" stroke="#000" strokeWidth="2" />
      <text x={x(sc) + (sc > 4 ? -10 : 10)} y={y(p) + (p > 0.8 ? 16 : -8)} textAnchor={sc > 4 ? "end" : "start"} fontSize="10" fontWeight="600">
        p = {pct(p)}
      </text>
    </svg>
  );
}

function ContributionBars({ d }: { d: Explain }) {
  const top = d.contributions.slice(0, 8);
  const max = Math.max(...top.map((c) => Math.abs(c.value)), 0.01);
  return (
    <div>
      <div className="flex justify-between text-[10px] uppercase tracking-[0.08em] font-semibold mb-2">
        <span className="text-legit">← towards legitimate</span>
        <span className="text-fraud">towards fraud →</span>
      </div>
      <div className="space-y-1.5">
        {top.map((c) => {
          const w = (Math.abs(c.value) / max) * 50;
          const pos = c.value >= 0;
          return (
            <div key={c.feature} className="grid grid-cols-[136px_1fr] items-center gap-2 text-[11px]" title={`${featLabel(c.feature, d.raw)}: ${signed(c.value, 3)} log-odds`}>
              <span className="text-black truncate">{featLabel(c.feature, d.raw)}</span>
              <div className="relative h-5">
                <div className="absolute top-0 bottom-0 left-1/2 w-px bg-black" />
                <div
                  className={`absolute top-1 h-3 ${pos ? "bg-fraud rounded-r-sm" : "bg-legit rounded-l-sm"}`}
                  style={pos ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }}
                />
                <span
                  className="absolute top-0.5 font-mono text-[10px] text-black"
                  style={pos ? { left: `calc(50% + ${w}% + 4px)` } : { right: `calc(50% + ${w}% + 4px)` }}
                >
                  {signed(c.value)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-[10px] text-subtle mt-3 leading-relaxed">
        SHAP contributions from LightGBM (<span className="font-mono">pred_contrib=True</span>), in log-odds.
        Start {signed(d.base_value)} + all contributions = score {signed(d.raw_score)}.
      </p>
    </div>
  );
}

function MetricsTable({ bench, cand, benchName, candName }: { bench: Metrics; cand: Metrics; benchName: string; candName: string }) {
  const rows: [string, (m: Metrics) => number, (v: number) => string][] = [
    ["Recall", (m) => m.recall, pct1],
    ["Precision", (m) => m.precision, pct1],
    ["F1-score", (m) => m.f1, (v) => v.toFixed(4)],
    ["ROC-AUC", (m) => m.auc, (v) => v.toFixed(4)],
    ["Fraud caught", (m) => Math.round(m.recall * HOLDOUT_FRAUD), (v) => `${v} / ${HOLDOUT_FRAUD}`],
    [
      "False alarms",
      (m) => {
        const caught = Math.round(m.recall * HOLDOUT_FRAUD);
        return Math.round((caught * (1 - m.precision)) / m.precision);
      },
      (v) => `${v}`,
    ],
  ];
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-[10px] uppercase tracking-[0.08em] text-subtle">
          <th className="text-left font-medium pb-2">Metric</th>
          <th className="text-right font-medium pb-2">{benchName}</th>
          <th className="text-right font-medium pb-2">{candName}</th>
          <th className="text-right font-medium pb-2">Δ</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-line">
        {rows.map(([name, get, fmt]) => {
          const b = get(bench), c = get(cand);
          const lowerIsBetter = name === "False alarms";
          const diff = c - b;
          const good = lowerIsBetter ? diff < 0 : diff > 0;
          const isCount = name === "Fraud caught" || name === "False alarms";
          return (
            <tr key={name}>
              <td className="py-1.5 text-black">{name}</td>
              <td className="py-1.5 text-right font-mono text-muted">{fmt(b)}</td>
              <td className="py-1.5 text-right font-mono text-black font-semibold">{fmt(c)}</td>
              <td className={`py-1.5 text-right font-mono font-semibold ${diff === 0 ? "text-subtle" : good ? "text-legit" : "text-fraud"}`}>
                {diff === 0 ? "0" : isCount ? signed(diff, 0) : name === "Recall" || name === "Precision" ? `${signed(diff * 100, 2)}pp` : signed(diff, 4)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ─── Stage detail: IN → PROCESS → OUT ────────────────────────────────────────

interface LoopCtx {
  outcome: "legit" | "fraud" | null;
  caseLabel: CaseLabel | null;
  caseAwaiting: boolean;
  labels: Labels;
  nightly: Nightly | null;
  benchmark: Metrics;
  version: string;
  busy: boolean;
  onLabel: (l: CaseLabel) => void;
  onNightly: () => void;
  onDs: (approve: boolean) => void;
}

function LoopDetail({ stage, d, raw, ctx }: { stage: number; d: Explain | null; raw: RawTransaction | null; ctx: LoopCtx }) {
  const n = ctx.nightly;
  let IN: React.ReactNode = null, PROC: React.ReactNode = null, OUT: React.ReactNode = null;
  const traced = d && raw;

  switch (stage) {
    case ST.review: // manual review
      IN = traced && ctx.outcome === "fraud" ? (
        <KV rows={[["trans_num", raw.trans_num.slice(0, 12) + "…"], ["merchant", raw.merchant], ["amount", `$${raw.amt.toFixed(2)}`], ["fraud probability", pct(d.probability)], ["status", "blocked"]]} />
      ) : (
        <p className="text-xs text-muted">Every payment with p ≥ 0.5 lands in the analyst&apos;s review queue. Send a transaction that gets flagged to follow one case.</p>
      );
      PROC = (
        <>
          <ul className="text-xs text-muted space-y-2 leading-relaxed">
            <li>• contact the cardholder (SMS / call)</li>
            <li>• check location, merchant and spending history</li>
            <li>• label the case — the model&apos;s guess is never the final truth</li>
          </ul>
          {ctx.caseAwaiting && ctx.outcome === "fraud" && (
            <div className="flex flex-wrap gap-2 mt-4">
              <ActionButton variant="fraud" disabled={ctx.busy} onClick={() => ctx.onLabel("confirmed")}>Confirm fraud</ActionButton>
              <ActionButton variant="legit" disabled={ctx.busy} onClick={() => ctx.onLabel("false_positive")}>Actually legit</ActionButton>
            </div>
          )}
        </>
      );
      OUT = (
        <Code>{`{ "trans_num": "${raw ? raw.trans_num.slice(0, 10) + "…" : "…"}",
  "label": ${ctx.caseLabel === "confirmed" ? 1 : ctx.caseLabel === "false_positive" ? 0 : "?"},   // 1 = fraud, 0 = legit
  "reason": "${ctx.caseLabel ? CASE_TEXT[ctx.caseLabel] : "pending"}",
  "source": "manual_review" }`}</Code>
      );
      break;

    case ST.complaint: // customer complaint
      IN = traced && ctx.outcome === "legit" ? (
        <KV rows={[["trans_num", raw.trans_num.slice(0, 12) + "…"], ["merchant", raw.merchant], ["amount", `$${raw.amt.toFixed(2)}`], ["fraud probability", pct(d.probability)], ["status", "approved"]]} />
      ) : (
        <p className="text-xs text-muted">Approved payments stay open for disputes. Send a transaction that gets approved to follow one case.</p>
      );
      PROC = (
        <>
          <ul className="text-xs text-muted space-y-2 leading-relaxed">
            <li>• customer reports &quot;I didn&apos;t make this payment&quot;</li>
            <li>• chargeback is investigated and confirmed</li>
            <li>• the payment is re-labelled as fraud the model missed</li>
          </ul>
          {ctx.caseAwaiting && ctx.outcome === "legit" && (
            <div className="flex flex-wrap gap-2 mt-4">
              <ActionButton variant="fraud" disabled={ctx.busy} onClick={() => ctx.onLabel("missed_fraud")}>Complaint received</ActionButton>
              <ActionButton variant="outline" disabled={ctx.busy} onClick={() => ctx.onLabel("no_complaint")}>No complaint</ActionButton>
            </div>
          )}
        </>
      );
      OUT = ctx.caseLabel === "no_complaint" ? (
        <p className="text-xs text-muted">No complaint — the payment keeps its legit verdict, nothing new to learn.</p>
      ) : (
        <Code>{`{ "trans_num": "${raw ? raw.trans_num.slice(0, 10) + "…" : "…"}",
  "label": ${ctx.caseLabel === "missed_fraud" ? 1 : "?"},   // 1 = fraud
  "reason": "${ctx.caseLabel === "missed_fraud" ? "missed fraud (chargeback)" : "pending"}",
  "source": "customer_complaint" }`}</Code>
      );
      break;

    case ST.store: // feedback store
      IN = (
        <KV rows={[["from manual review", "confirmed fraud / false positive"], ["from complaints", "missed fraud"], ["+ features", "same 39 columns as the model"]]} />
      );
      PROC = (
        <>
          <Code>{`INSERT INTO feedback_labels
  (trans_num, features, label, source, labelled_at)`}</Code>
          <p className="text-xs text-muted mt-3">Labels wait here until the nightly run picks them up.</p>
          <div className="mt-4">
            <ActionButton variant="dark" disabled={ctx.busy || totalLabels(ctx.labels) === 0} onClick={ctx.onNightly}>Run nightly retraining</ActionButton>
          </div>
        </>
      );
      OUT = (
        <>
          <KV
            rows={[
              ["confirmed fraud", ctx.labels.confirmed],
              ["false positives (legit)", ctx.labels.false_positive],
              ["missed fraud (complaints)", ctx.labels.missed_fraud],
              ["waiting for 02:00", <strong key="t">{totalLabels(ctx.labels)} labels</strong>],
            ]}
          />
        </>
      );
      break;

    case ST.retrain: // retrain
      IN = (
        <KV
          rows={[
            ["training set", `${TRAIN_ROWS.toLocaleString()} rows`],
            ["new labels", n ? totalLabels(n.used) : totalLabels(ctx.labels)],
            ["current model", ctx.version],
          ]}
        />
      );
      PROC = (
        <Code>{`openclaw run nightly-retrain   # 02:00
 1. pull new labels from feedback store
 2. merge + de-duplicate with training set
 3. same feature pipeline (FE → OHE → scale)
 4. train LightGBM (same hyper-parameters)
 5. evaluate on hold-out (${HOLDOUT_TX.toLocaleString()} tx)`}</Code>
      );
      OUT = n ? (
        <KV rows={[["candidate model", n.to], ["recall", pct1(n.candidate.recall)], ["precision", pct1(n.candidate.precision)], ["F1", n.candidate.f1.toFixed(4)]]} />
      ) : (
        <p className="text-xs text-muted">Run the nightly retraining to produce a candidate model.</p>
      );
      break;

    case ST.compare: // compare
      IN = (
        <KV rows={[["benchmark", `${n ? n.from : ctx.version} (production)`], ["candidate", n ? n.to : "—"], ["hold-out set", `${HOLDOUT_TX.toLocaleString()} tx · ${HOLDOUT_FRAUD} fraud`]]} />
      );
      PROC = (
        <>
          <Code>{`promote = candidate.f1 > benchmark.f1
      and candidate.recall >= benchmark.recall`}</Code>
          {n && (
            <div className="mt-3 bg-card p-3">
              <MetricsTable bench={BASELINE_OF(n)} cand={n.candidate} benchName={n.from} candName={n.to} />
            </div>
          )}
        </>
      );
      OUT = n ? (
        <div className={`p-4 text-white ${n.better ? "bg-legit" : "bg-black"}`}>
          <div className="text-[10px] uppercase tracking-[0.14em] opacity-80">Result</div>
          <div className="text-xl font-medium mt-1">{n.better ? "Candidate is better" : "Not better"}</div>
          <div className="text-xs mt-2 opacity-90">{n.better ? "→ task for a data scientist" : "→ keep the current model"}</div>
        </div>
      ) : (
        <p className="text-xs text-muted">No candidate yet.</p>
      );
      break;

    case ST.task: // DS task
      IN = n ? <KV rows={[["report", `${n.from} vs ${n.to}`], ["F1", `${signed((n.candidate.f1 - BASELINE_OF(n).f1), 4)}`], ["new labels used", totalLabels(n.used)]]} /> : <p className="text-xs text-muted">Waiting for a candidate that beats the benchmark.</p>;
      PROC = n ? (
        <div className="bg-card border border-line p-4">
          <div className="flex justify-between items-start gap-3">
            <div>
              <div className="font-mono text-[10px] text-subtle">TASK-{n.task}</div>
              <div className="text-sm font-medium text-black mt-0.5">Verify candidate LightGBM {n.to}</div>
            </div>
            <span className="text-[10px] uppercase tracking-[0.08em] bg-brand px-1.5 py-0.5 font-semibold">{n.status === "awaiting_ds" ? "Open" : n.status === "published" ? "Done" : n.status === "rejected" ? "Rejected" : "—"}</span>
          </div>
          <div className="text-[11px] text-muted mt-1">Assignee: Data Scientist · created by OpenClaw</div>
          <ul className="text-xs text-black mt-3 space-y-1">
            <li>☐ re-check metrics on the hold-out set</li>
            <li>☐ spot-check new false positives / negatives</li>
            <li>☐ check for data drift and bias</li>
            <li>☐ approve publishing</li>
          </ul>
          {n.status === "awaiting_ds" && (
            <div className="flex flex-wrap gap-2 mt-4">
              <ActionButton variant="primary" disabled={ctx.busy} onClick={() => ctx.onDs(true)}>Approve &amp; publish</ActionButton>
              <ActionButton variant="outline" disabled={ctx.busy} onClick={() => ctx.onDs(false)}>Reject</ActionButton>
            </div>
          )}
        </div>
      ) : null;
      OUT = n && (n.status === "published" || n.status === "rejected") ? (
        <p className="text-sm text-black">{n.status === "published" ? `Signed off — ${n.to} goes to production.` : `Rejected — ${n.from} stays in production.`}</p>
      ) : (
        <p className="text-xs text-muted">Waiting for the data scientist&apos;s decision.</p>
      );
      break;

    case ST.publish: // publish
      IN = n ? <KV rows={[["approved model", n.to], ["approved by", "Data Scientist"]]} /> : <p className="text-xs text-muted">Nothing approved yet.</p>;
      PROC = (
        <Code>{`registry.publish("lgbm_fraud_model", "${n ? n.to : "vX"}")
benchmark = candidate.metrics
api.reload_model()   # step ${pad(MODEL)} now uses the new version`}</Code>
      );
      OUT = (
        <KV rows={[["production model", ctx.version], ["new benchmark F1", ctx.benchmark.f1.toFixed(4)], ["new benchmark recall", pct1(ctx.benchmark.recall)]]} />
      );
      break;

    case ST.keep: // keep
      IN = n ? <KV rows={[["candidate", n.to], ["reason", n.status === "rejected" ? "rejected by data scientist" : "did not beat benchmark"]]} /> : <p className="text-xs text-muted">—</p>;
      PROC = <Code>{`log_report(candidate, benchmark)
# production model unchanged`}</Code>;
      OUT = <KV rows={[["production model", ctx.version], ["benchmark F1", ctx.benchmark.f1.toFixed(4)]]} />;
      break;
  }

  return (
    <div className="bg-card border border-line p-5 md:p-6 animate-slide-in" key={stage}>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-5">
        <span className={`font-mono text-xs px-2 py-1 ${stage >= ST.retrain ? "bg-black text-brand" : "bg-sand text-black border border-black"}`}>{String(stage + 1).padStart(2, "0")}</span>
        <h2 className="text-xl font-medium text-black">{STAGES[stage].title.join(" ")}</h2>
        <SimBadge />
        <p className="text-sm text-muted basis-full">{STAGES[stage].desc}</p>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1.35fr_auto_1fr] gap-3 items-stretch">
        <Block label="Data in">{IN}</Block>
        <div className="hidden lg:flex items-center text-2xl text-black">→</div>
        <Block label="Process">{PROC}</Block>
        <div className="hidden lg:flex items-center text-2xl text-black">→</div>
        <Block label="Data out">{OUT}</Block>
      </div>
      <p className="text-[10px] text-subtle mt-4">
        Feedback loop and retraining are simulated for the demo — labels, metrics and model versions only live in this page; the production model file is not changed.
      </p>
    </div>
  );
}

// benchmark the candidate was compared against (kept on the nightly record)
const BASELINE_OF = (n: Nightly & { bench?: Metrics }) => n.bench ?? BASELINE;

function StageDetail({ stage, d, raw, ctx }: { stage: number; d: Explain | null; raw: RawTransaction | null; ctx: LoopCtx }) {
  if (stage > DECISION) return <LoopDetail stage={stage} d={d} raw={raw} ctx={ctx} />;
  if (stage < 0 || !raw) {
    return (
      <div className="bg-card border border-line p-10 text-center">
        <div className="w-10 h-10 bg-brand mx-auto mb-4" />
        <p className="text-sm text-black">Run a transaction to see the real data entering and leaving every step.</p>
        <p className="text-xs text-subtle mt-1">Click any step in the schematic to inspect it.</p>
      </div>
    );
  }
  if (!d && stage > 0) return null;

  let IN: React.ReactNode = null, PROC: React.ReactNode = null, OUT: React.ReactNode = null;

  switch (stage) {
    case 0:
      IN = (
        <>
          <Code>GET /generate</Code>
          <p className="text-xs text-muted mt-3 leading-relaxed">
            Request from the browser. The generator samples category, state, location and amount from distributions of the Kaggle dataset; ~5% follow a suspicious pattern.
          </p>
        </>
      );
      PROC = (
        <ul className="text-xs text-muted space-y-2 leading-relaxed">
          <li>• category / state picked with real dataset frequencies</li>
          <li>• cardholder lat/long inside the dataset bounding box</li>
          <li>• normal: lognormal amount ≤ $500, merchant within ~50 km</li>
          <li>• suspicious: $800–5000, merchant 500+ km away, 23:00–04:00</li>
        </ul>
      );
      OUT = <KV rows={Object.entries(raw).map(([k, v]) => [k, String(v)])} />;
      break;

    case 1:
      IN = (
        <>
          <Code>{`POST /explain\nContent-Type: application/json`}</Code>
          <div className="flex flex-wrap gap-1 mt-3">
            {Object.keys(raw).map((k) => (
              <span key={k} className="font-mono text-[10px] bg-card border border-line px-1.5 py-0.5">{k}</span>
            ))}
          </div>
        </>
      );
      PROC = (
        <>
          <Code>{`missing = [k for k in REQUIRED\n           if k not in payload]\nif missing: return 422`}</Code>
          <p className="text-xs text-muted mt-3">Requests with missing fields are rejected before reaching the model.</p>
        </>
      );
      OUT = (
        <>
          <div className="space-y-1">
            {d!.validation.required.map((k) => (
              <div key={k} className="flex justify-between text-xs">
                <span className="font-mono text-black">{k}</span>
                <span className="text-legit font-semibold">✓ present</span>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-subtle mt-3">Passed through, not used by the model: merchant, city, job, trans_num</p>
        </>
      );
      break;

    case 2: {
      const e = d!.engineered;
      IN = (
        <KV
          rows={[
            ["trans_date_trans_time", raw.trans_date_trans_time],
            ["dob", raw.dob],
            ["lat, long", `${raw.lat}, ${raw.long}`],
            ["merch_lat, merch_long", `${raw.merch_lat}, ${raw.merch_long}`],
            ["amt", `$${raw.amt.toFixed(2)}`],
          ]}
        />
      );
      PROC = (
        <Code>{`hour        = dt.hour
day_of_week = dt.dayofweek
month       = dt.month
age         = (dt − dob).days // 365
distance_km = haversine(card, merchant)
log_amt     = ln(1 + amt)`}</Code>
      );
      OUT = (
        <>
          <KV
            rows={[
              ["hour", e.hour],
              ["day_of_week", e.day_of_week],
              ["month", e.month],
              ["age", `${e.age} years`],
              ["distance_km", `${e.distance_km} km`],
              ["log_amt", e.log_amt],
            ]}
          />
          <div className="flex flex-wrap gap-1 mt-3">
            {d!.dropped.map((k) => (
              <span key={k} className="font-mono text-[10px] text-subtle line-through">{k}</span>
            ))}
          </div>
        </>
      );
      break;
    }

    case 3: {
      const ohe = d!.vector.filter((v) => v.feature.startsWith("category_") || v.feature.startsWith("state_"));
      IN = (
        <KV rows={[["category", raw.category], ["state", raw.state]]} />
      );
      PROC = (
        <>
          <Code>{`pd.get_dummies(df, columns=["category", "state"])
df.reindex(columns=feature_cols, fill_value=0)`}</Code>
          <p className="text-xs text-muted mt-3">Every possible value gets its own column, so the model sees numbers, never text.</p>
        </>
      );
      OUT = (
        <div className="grid grid-cols-3 gap-1">
          {ohe.map((v) => (
            <div
              key={v.feature}
              className={`font-mono text-[9px] px-1.5 py-1 flex justify-between ${
                v.value === 1 ? "bg-black text-brand font-semibold" : "bg-card text-subtle"
              }`}
            >
              <span className="truncate">{v.feature.replace("category_", "").replace("state_", "")}</span>
              <span>{v.value}</span>
            </div>
          ))}
        </div>
      );
      break;
    }

    case 4: {
      const m = d!.model;
      IN = (
        <>
          {/* 12 numeric features in raw units + 27 one-hot columns */}
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px] font-mono">
            {d!.vector
              .filter((v) => !v.feature.startsWith("category_") && !v.feature.startsWith("state_"))
              .map((v) => (
                <div key={v.feature} className="flex justify-between gap-2">
                  <span className="text-subtle truncate">{v.feature}</span>
                  <span className="text-black">{v.value}</span>
                </div>
              ))}
          </div>
          <div className="grid gap-0.5 mt-3" style={{ gridTemplateColumns: "repeat(14, minmax(0, 1fr))" }}>
            {d!.vector
              .filter((v) => v.feature.startsWith("category_") || v.feature.startsWith("state_"))
              .map((v) => (
                <div
                  key={v.feature}
                  title={`${v.feature} = ${v.value}`}
                  className={`aspect-square ${v.value === 1 ? "bg-black" : "bg-card border border-line"}`}
                />
              ))}
          </div>
          <p className="text-xs text-muted mt-3">
            Feature vector of {d!.vector.length} values: 12 numeric features in their original units (no scaling) + 27 one-hot columns (black = 1). Hover a square to see its column.
          </p>
        </>
      );
      PROC = (
        <>
          <div className="text-[10px] text-subtle uppercase tracking-[0.1em] mb-2">
            Tree #1 of {m.n_trees} — the actual path this transaction took
          </div>
          <div className="space-y-1">
            {d!.tree_path.map((s, i) => {
              const name = s.feature.startsWith("category_")
                ? `category = ${s.feature.slice(9).replace("_", " ")}`
                : s.feature.startsWith("state_")
                ? `state = ${s.feature.slice(6)}`
                : null;
              const q = name
                ? `Is ${name}?`
                : `Is ${s.feature} ≤ ${s.threshold}${s.feature === "log_amt" ? ` (≈ $${Math.expm1(s.threshold).toFixed(0)})` : ""}?`;
              const answer = name ? (s.go_left ? "No" : "Yes") : s.go_left ? "Yes" : "No";
              return (
                <div key={i} className="flex items-center gap-2 text-[11px]" style={{ paddingLeft: i * 8 }}>
                  <span className="font-mono text-subtle w-4">{i + 1}</span>
                  <span className="flex-1 bg-card px-2 py-1 truncate">
                    <span className="text-black">{q}</span>{" "}
                    <span className="text-subtle font-mono">value {name ? (s.go_left ? 0 : 1) : s.value}</span>
                  </span>
                  <span className="font-semibold text-black w-8">{answer}</span>
                </div>
              );
            })}
            <div className="flex items-center gap-2 text-[11px]" style={{ paddingLeft: d!.tree_path.length * 8 }}>
              <span className="w-4" />
              <span className="bg-black text-brand font-mono px-2 py-1">leaf value {signed(d!.tree_leaf)}</span>
            </div>
          </div>
          <p className="text-[10px] text-subtle mt-3">
            {m.n_trees} trees · max depth {m.max_depth} · {m.num_leaves} leaves · learning rate {m.learning_rate}. Each next tree corrects the error of the previous ones.
          </p>
        </>
      );
      // ② sum of the trees → raw score, ③ sigmoid → probability
      PROC = (
        <>
          {PROC}
          <div className="text-[10px] text-subtle uppercase tracking-[0.1em] mt-5 mb-1">
            Sum of all {m.n_trees} trees = raw score
          </div>
          <div className="bg-card p-2">
            <BoostingChart data={d!.boosting} final={d!.raw_score} />
          </div>
        </>
      );
      OUT = (
        <div>
          <div className="text-[10px] text-subtle uppercase tracking-[0.1em]">Raw score (log-odds)</div>
          <div className="text-2xl font-medium tabular-nums mt-1">{signed(d!.raw_score, 4)}</div>
          <div className="text-[10px] text-subtle uppercase tracking-[0.1em] mt-4 mb-1">Sigmoid</div>
          <Code>{`p = 1 / (1 + e^(−score))
  = 1 / (1 + e^(${(-d!.raw_score).toFixed(3)}))
  = ${d!.probability}`}</Code>
          <div className="mt-2 bg-card p-2">
            <SigmoidChart score={d!.raw_score} p={d!.probability} />
          </div>
          <div className="text-[10px] text-subtle uppercase tracking-[0.1em] mt-4">Fraud probability</div>
          <div className={`text-3xl font-medium tabular-nums mt-1 ${d!.is_fraud ? "text-fraud" : "text-legit"}`}>
            {pct(d!.probability)}
          </div>
        </div>
      );
      break;
    }

    case 5:
      IN = (
        <KV rows={[["probability", pct(d!.probability)], ["threshold", "0.50"]]} />
      );
      PROC = (
        <>
          <Code>{`if p >= 0.5:  verdict = "FRAUD"
else:         verdict = "Legitimate"`}</Code>
          <div className="mt-4 bg-card p-3">
            <div className="text-[10px] uppercase tracking-[0.12em] font-semibold text-black mb-3">
              Why? What pushed this decision
            </div>
            <ContributionBars d={d!} />
          </div>
        </>
      );
      OUT = (
        <div>
          <div className={`p-4 ${d!.is_fraud ? "bg-fraud" : "bg-legit"} text-white`}>
            <div className="text-[10px] uppercase tracking-[0.14em] opacity-80">Verdict</div>
            <div className="text-2xl font-medium mt-1">{d!.is_fraud ? "Fraud" : "Legitimate"}</div>
            <div className="text-xs mt-2 opacity-90">
              {d!.is_fraud ? "Payment blocked and sent for manual review" : "Payment approved"}
            </div>
          </div>
          <div className="mt-3">
            <Code>{`{ "verdict": "${d!.verdict}",
  "fraud_probability": ${d!.probability},
  "threshold": 0.5 }`}</Code>
          </div>
          <p className="text-[10px] text-subtle mt-2">
            Response returned to the client. Next: {d!.is_fraud ? `manual review (step ${pad(ST.review)})` : `open for customer complaints (step ${pad(ST.complaint)})`} ↓
          </p>
        </div>
      );
      break;
  }

  return (
    <div className="bg-card border border-line p-5 md:p-6 animate-slide-in" key={stage}>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 mb-5">
        <span className="bg-black text-brand font-mono text-xs px-2 py-1">{String(stage + 1).padStart(2, "0")}</span>
        <h2 className="text-xl font-medium text-black">{STAGES[stage].title.join(" ")}</h2>
        <p className="text-sm text-muted basis-full md:basis-auto">{STAGES[stage].desc}</p>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1.35fr_auto_1fr] gap-3 items-stretch">
        <Block label="Data in">{IN}</Block>
        <div className="hidden lg:flex items-center text-2xl text-black">→</div>
        <Block label="Process">{PROC}</Block>
        <div className="hidden lg:flex items-center text-2xl text-black">→</div>
        <Block label="Data out">{OUT}</Block>
      </div>
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

interface Dot {
  id: number;
  start: number;
  path: Pt[];
  dur: number;
  termT: number; // fraction of the path at which the dot reaches its terminal
  counted: boolean;
  route: "legit" | "fraud";
  label: keyof Labels | null; // label it becomes in the feedback store (if any)
}
const STREAM_MS = 3200; // time to travel request → approve terminal
const STREAM_SPEED = pathLen([...MAIN_PATH, ...BRANCH.legit]) / STREAM_MS;
const COMPLAINT_RATE = 0.06; // share of approved payments that get disputed (exaggerated for the demo)
const FP_RATE = 0.25; // share of flagged payments that the analyst finds legit

export default function PipelinePage() {
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [raw, setRaw] = useState<RawTransaction | null>(null);
  const [data, setData] = useState<Explain | null>(null);
  const [active, setActive] = useState(-1); // main stage the packet is at (DONE = finished)
  const [loopActive, setLoopActive] = useState<number | null>(null); // loop stage the packet is at
  const [visited, setVisited] = useState<number[]>([]); // loop stages passed
  const [litEdges, setLitEdges] = useState<string[]>([]); // loop edges travelled
  const [selected, setSelected] = useState<number | null>(null);
  const [packet, setPacket] = useState<Pt | null>(null);
  const [outcome, setOutcome] = useState<"legit" | "fraud" | null>(null);
  const [busy, setBusy] = useState(false);
  const [pace, setPace] = useState<Pace>("slow");
  const paceRef = useRef<Pace>("slow");
  const packetRef = useRef<Pt>(START);

  // feedback loop / model lifecycle (simulated)
  const [caseLabel, setCaseLabel] = useState<CaseLabel | null>(null);
  const [caseAwaiting, setCaseAwaiting] = useState(false);
  const [labels, setLabels] = useState<Labels>(NO_LABELS);
  const [nightly, setNightly] = useState<(Nightly & { bench: Metrics }) | null>(null);
  const [benchmark, setBenchmark] = useState<Metrics>(BASELINE);
  const [minor, setMinor] = useState(0);
  const [flash, setFlash] = useState(false);
  const taskRef = useRef(1041);
  const version = `v1.${minor}`;

  const [streaming, setStreaming] = useState(false);
  const streamingRef = useRef(false);
  const dotsRef = useRef<Dot[]>([]);
  const [dotPos, setDotPos] = useState<{ id: number; p: Pt; color: string }[]>([]);
  const [counts, setCounts] = useState({ legit: 0, fraud: 0 });
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    paceRef.current = pace;
  }, [pace]);

  // Stop the animation loop when leaving the page
  useEffect(
    () => () => {
      streamingRef.current = false;
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    },
    []
  );

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false));
  }, []);

  // ── Packet animation ──
  const moveTo = (pts: Pt[], ms: number) =>
    new Promise<void>((resolve) => {
      const path = [packetRef.current, ...pts];
      const t0 = performance.now();
      const tick = (now: number) => {
        const t = Math.min((now - t0) / ms, 1);
        const p = pointAlong(path, t);
        packetRef.current = p;
        setPacket(p);
        if (t < 1) requestAnimationFrame(tick);
        else resolve();
      };
      requestAnimationFrame(tick);
    });
  const travel = () => (paceRef.current === "fast" ? 350 : 650);
  const dwell = (k = 1) => sleep(DWELL[paceRef.current] * k);
  const lightEdge = (key: string) => setLitEdges((e) => (e.includes(key) ? e : [...e, key]));
  const reachLoop = (i: number) => {
    setLoopActive(i);
    setVisited((v) => (v.includes(i) ? v : [...v, i]));
  };
  const travelEdge = async (key: string, stop: Pt, k = 1.4) => {
    lightEdge(key);
    await moveTo([...EDGES[key], stop], travel() * k);
  };

  // ── Single-request trace ──
  const runTrace = async (suspicious: boolean) => {
    if (busy) return;
    setBusy(true);
    setSelected(null);
    setData(null);
    setRaw(null);
    setOutcome(null);
    setActive(-1);
    setLoopActive(null);
    setVisited([]);
    setLitEdges([]);
    setCaseLabel(null);
    setCaseAwaiting(false);
    packetRef.current = START;
    setPacket(START);

    try {
      const rawP = fetch(`${API}/generate${suspicious ? "?fraud=1" : ""}`).then((r) => r.json());
      await moveTo([{ x: entryX(0), y: CY }, nodeStop(0)], travel());
      const r: RawTransaction = await rawP;
      setRaw(r);
      setActive(0);
      const expP = fetch(`${API}/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(r),
      }).then((res) => res.json());
      await dwell();
      const d: Explain = await expP;
      setData(d);

      // follow the arrow: leave step i−1 on its right edge, enter step i on the left
      for (let i = 1; i < MAIN_N; i++) {
        await moveTo([{ x: exitX(i - 1), y: CY }, { x: entryX(i), y: CY }, nodeStop(i)], travel());
        setActive(i);
        await dwell();
      }
      const route = d.is_fraud ? "fraud" : "legit";
      await moveTo([{ x: DIAMOND.x + DIAMOND.r, y: CY }, ...BRANCH[route]], travel() * 1.6);
      setOutcome(route);
      setActive(DONE);
      setCaseAwaiting(true);
      setSelected(DECISION);
    } catch {
      setApiOk(false);
    } finally {
      setBusy(false);
    }
  };

  // ── Feedback: analyst label / customer complaint ──
  const labelCase = async (label: CaseLabel) => {
    if (busy || !outcome || !caseAwaiting) return;
    setBusy(true);
    setSelected(null);
    setCaseAwaiting(false);
    if (outcome === "fraud") {
      await travelEdge("flag-review", loopStop(ST.review));
      reachLoop(ST.review);
      await dwell(0.8);
      setCaseLabel(label);
      await dwell(0.6);
      await travelEdge("review-store", loopStop(ST.store), 2);
    } else {
      await travelEdge("approve-complaint", loopStop(ST.complaint));
      reachLoop(ST.complaint);
      await dwell(0.8);
      setCaseLabel(label);
      if (label === "no_complaint") {
        setSelected(ST.complaint);
        setBusy(false);
        return;
      }
      await dwell(0.6);
      await travelEdge("complaint-store", loopStop(ST.store), 1.2);
    }
    reachLoop(ST.store);
    if (label !== "no_complaint") setLabels((l) => ({ ...l, [label]: l[label] + 1 }));
    setSelected(ST.store);
    setBusy(false);
  };

  // ── Nightly retraining (OpenClaw) ──
  const runNightly = async () => {
    if (busy || totalLabels(labels) === 0) return;
    setBusy(true);
    setSelected(null);
    setVisited((v) => v.filter((s) => s <= ST.store));
    setLitEdges((e) => e.filter((k) => EDGE_TONE[k] === "light"));
    const used = { ...labels };
    const bench = benchmark;
    const from = version;
    const to = `v1.${minor + 1}`;
    const candidate = simulateCandidate(bench, used, totalLabels(used) + minor * 7);
    const better = candidate.f1 > bench.f1 && candidate.recall >= bench.recall;
    const task = ++taskRef.current;

    packetRef.current = loopStop(ST.store);
    setPacket(loopStop(ST.store));
    reachLoop(ST.store);
    await dwell(0.6);

    await travelEdge("store-retrain", loopStop(ST.retrain));
    reachLoop(ST.retrain);
    setLabels(NO_LABELS); // consumed by the nightly run
    setNightly({ used, candidate, better, from, to, task, status: "running", bench });
    await dwell(1.4);

    await travelEdge("retrain-compare", loopStop(ST.compare), 1);
    reachLoop(ST.compare);
    setNightly((n) => n && { ...n, status: "compared" });
    await dwell(1.2);

    if (better) {
      await travelEdge("compare-task", loopStop(ST.task), 1);
      reachLoop(ST.task);
      setNightly((n) => n && { ...n, status: "awaiting_ds" });
      setSelected(ST.task);
    } else {
      await travelEdge("compare-keep", loopStop(ST.keep), 1);
      reachLoop(ST.keep);
      setNightly((n) => n && { ...n, status: "kept" });
      setSelected(ST.compare);
    }
    setBusy(false);
  };

  // ── Data scientist decision ──
  const dsDecision = async (approve: boolean) => {
    if (busy || !nightly || nightly.status !== "awaiting_ds") return;
    setBusy(true);
    setSelected(null);
    if (!approve) {
      setNightly({ ...nightly, status: "rejected" });
      reachLoop(ST.keep);
      setSelected(ST.keep);
      setBusy(false);
      return;
    }
    await travelEdge("task-publish", loopStop(ST.publish), 1);
    reachLoop(ST.publish);
    await dwell(0.8);
    lightEdge("publish-model");
    await moveTo([...EDGES["publish-model"], nodeStop(MODEL)], travel() * 3);
    setMinor((m) => m + 1);
    setBenchmark(nightly.candidate);
    setNightly({ ...nightly, status: "published" });
    setFlash(true);
    setTimeout(() => setFlash(false), 1600);
    setSelected(ST.publish);
    setBusy(false);
  };

  // ── Stream mode: many requests flowing at once (incl. the feedback lane) ──
  const animateDots = useCallback(() => {
    const now = performance.now();
    const alive: Dot[] = [];
    const pos: typeof dotPos = [];
    let legit = 0, fraud = 0;
    const newLabels: Labels = { ...NO_LABELS };
    for (const dot of dotsRef.current) {
      const t = (now - dot.start) / dot.dur;
      if (!dot.counted && t >= dot.termT) {
        dot.counted = true;
        if (dot.route === "fraud") fraud++;
        else legit++;
      }
      if (t >= 1) {
        if (dot.label) newLabels[dot.label]++;
        continue;
      }
      alive.push(dot);
      const p = pointAlong(dot.path, t);
      const decided = p.y > CY + 0.5; // only the branches run below the main line
      const color = !decided
        ? "#000"
        : dot.label === "missed_fraud" && t > dot.termT
        ? "#c62828" // approved, then disputed → turns into fraud
        : dot.route === "fraud"
        ? dot.label === "false_positive" && t > dot.termT + (1 - dot.termT) * 0.3
          ? "#085850" // analyst found it legit
          : "#c62828"
        : "#085850";
      pos.push({ id: dot.id, p, color });
    }
    dotsRef.current = alive;
    setDotPos(pos);
    if (legit || fraud) setCounts((c) => ({ legit: c.legit + legit, fraud: c.fraud + fraud }));
    if (totalLabels(newLabels))
      setLabels((l) => ({
        confirmed: l.confirmed + newLabels.confirmed,
        false_positive: l.false_positive + newLabels.false_positive,
        missed_fraud: l.missed_fraud + newLabels.missed_fraud,
      }));
    if (alive.length || streamingRef.current) rafRef.current = requestAnimationFrame(animateDots);
    else rafRef.current = null;
  }, []);

  useEffect(() => {
    if (!streaming) return;
    let id = 0;
    let stopped = false;
    const spawn = async () => {
      while (!stopped) {
        try {
          const r = await (await fetch(`${API}/generate`)).json();
          const p = await (
            await fetch(`${API}/predict`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(r),
            })
          ).json();
          const route: "legit" | "fraud" = p.is_fraud ? "fraud" : "legit";
          const head = [...MAIN_PATH, ...BRANCH[route]];
          let tail: Pt[] = [];
          let label: keyof Labels | null = null;
          if (route === "fraud") {
            label = Math.random() < FP_RATE ? "false_positive" : "confirmed";
            tail = [...EDGES["flag-review"], ...EDGES["review-store"]];
          } else if (Math.random() < COMPLAINT_RATE) {
            label = "missed_fraud";
            tail = [...EDGES["approve-complaint"], { x: C(6), y: 500 }, ...EDGES["complaint-store"]];
          }
          const path = [...head, ...tail];
          const len = pathLen(path);
          dotsRef.current.push({
            id: id++,
            start: performance.now(),
            path,
            dur: len / STREAM_SPEED,
            termT: pathLen(head) / len,
            counted: false,
            route,
            label,
          });
          if (rafRef.current === null) rafRef.current = requestAnimationFrame(animateDots);
        } catch {
          // API hiccup — keep streaming
        }
        await sleep(260);
      }
    };
    spawn();
    return () => {
      stopped = true;
    };
  }, [streaming, animateDots]);

  const toggleStream = () => {
    streamingRef.current = !streaming;
    setStreaming(!streaming);
  };

  const mainShown = active === DONE ? DECISION : active >= 0 ? active : -1;
  const shown = selected ?? loopActive ?? mainShown;
  const tagStage = active >= 0 && active < MAIN_N ? active : -1;
  const [tagIn, tagOut] = tagStage >= 0 ? payloadTag(tagStage, raw, data) : ["", ""];
  const tagX = tagStage >= 0 ? Math.min(Math.max(nodeC(tagStage).x - 80, 4), VB_W - 164) : 0;
  const passed = (i: number) => active > i;
  const edgeOn = (key: string) => litEdges.includes(key) || (streaming && STREAM_EDGES.includes(key));

  const ctx: LoopCtx = {
    outcome,
    caseLabel,
    caseAwaiting,
    labels,
    nightly,
    benchmark,
    version,
    busy,
    onLabel: labelCase,
    onNightly: runNightly,
    onDs: dsDecision,
  };

  /** Summary line written inside each loop node */
  const loopSummary = (i: number): string => {
    switch (i) {
      case ST.review: return caseLabel && outcome === "fraud" ? CASE_TEXT[caseLabel] : `${labels.confirmed + labels.false_positive} queued`;
      case ST.complaint: return caseLabel && outcome === "legit" ? CASE_TEXT[caseLabel] : `${labels.missed_fraud} disputes`;
      case ST.store: return `${totalLabels(labels)} new labels`;
      case ST.retrain: return nightly ? `${totalLabels(nightly.used)} labels → ${nightly.to}` : "runs at 02:00";
      case ST.task: return nightly?.status === "awaiting_ds" ? `TASK-${nightly.task} open` : nightly?.status === "published" ? "approved" : nightly?.status === "rejected" ? "rejected" : "";
      case ST.publish: return nightly?.status === "published" ? `${version} live` : "";
      case ST.keep: return nightly?.status === "kept" || nightly?.status === "rejected" ? `stays ${version}` : "";
      default: return "";
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <TopBar apiOk={apiOk} />

      <main className="flex-1 w-full max-w-7xl mx-auto px-4 md:px-6 py-8 md:py-10">
        {/* Hero */}
        <section className="mb-6 flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6">
          <div>
            <p className="text-[11px] uppercase tracking-[0.16em] text-muted font-medium mb-3">
              End-to-end · real data from the API
            </p>
            <h1 className="text-3xl md:text-5xl font-medium leading-[1.1] tracking-tight text-black">
              How the AI pipeline{" "}
              <span className="bg-brand px-2 -mx-0.5 whitespace-nowrap">makes a decision</span>
            </h1>
            <p className="text-muted text-sm mt-4 max-w-2xl">
              Follow one request through every step: what data it brings in, what each step hands to the next, how the model arrives at the verdict — and how human feedback retrains the model every night.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => runTrace(false)}
              disabled={busy || apiOk !== true}
              className="h-12 px-5 font-medium text-xs uppercase tracking-[0.1em] bg-brand text-black hover:bg-black hover:text-brand transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              ▶ Send transaction
            </button>
            <button
              onClick={() => runTrace(true)}
              disabled={busy || apiOk !== true}
              className="h-12 px-5 font-medium text-xs uppercase tracking-[0.1em] border border-black text-black hover:bg-black hover:text-white transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Send suspicious
            </button>
            <button
              onClick={toggleStream}
              disabled={apiOk !== true}
              className={`h-12 px-5 font-medium text-xs uppercase tracking-[0.1em] transition-colors disabled:opacity-30 disabled:cursor-not-allowed ${
                streaming ? "bg-black text-brand" : "border border-line bg-card text-black hover:border-black"
              }`}
            >
              {streaming ? "■ Stop stream" : "Stream mode"}
            </button>
            <div className="h-12 flex items-center bg-card border border-line text-[11px] uppercase tracking-[0.08em]">
              {(Object.keys(DWELL) as Pace[]).map((k) => (
                <button
                  key={k}
                  onClick={() => setPace(k)}
                  className={`h-full px-3 ${pace === k ? "bg-black text-white" : "text-muted hover:text-black"}`}
                >
                  {k}
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* ── Schematic ── */}
        <div className="bg-card border border-line overflow-x-auto">
          <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full min-w-[900px] h-auto block select-none" role="img" aria-label="Fraud detection pipeline with feedback loop and nightly retraining">
            <defs>
              {[
                ["arr", "#000"],
                ["arr-idle", "#bdbdbd"],
                ["arr-white", "#fff"],
                ["arr-dim", "#525252"],
                ["arr-yellow", "#f7e948"],
              ].map(([id, c]) => (
                <marker key={id} id={id} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                  <path d="M0,0 L10,5 L0,10 z" fill={c} />
                </marker>
              ))}
            </defs>

            {/* Lane backgrounds */}
            <rect x={0} y={L2.y} width={VB_W} height={L2.end - L2.y} fill="#f2f0ed" />
            <rect x={0} y={L3.y} width={VB_W} height={L3.end - L3.y} fill="#000" />

            {/* Phase bands */}
            {PHASES.map((ph) => {
              const x1 = nodeX(ph.from);
              const x2 = ph.label === "Decision" ? VB_W - 12 : nodeX(ph.to) + NODE_W;
              return (
                <g key={ph.label}>
                  <text x={x1} y={24} fontSize="10" letterSpacing="1.6" fontWeight="600" fill="#525252">
                    {ph.label.toUpperCase()}
                  </text>
                  <line x1={x1} x2={x2} y1={32} y2={32} stroke="#000" strokeWidth="2" />
                </g>
              );
            })}

            {/* Lane titles */}
            <g transform={`translate(${nodeX(0)}, ${L2.y + 30})`}>
              <text fontSize="10" letterSpacing="1.6" fontWeight="600" fill="#000">FEEDBACK LOOP · HUMAN IN THE LOOP</text>
              {["Flagged payments are reviewed by an analyst;", "approved ones can be disputed by the customer.", "Every human label becomes new training data."].map((t, k) => (
                <text key={k} y={22 + k * 16} fontSize="11" fill="#525252">{t}</text>
              ))}
              <rect y={80} width="88" height="18" fill="#50327b" />
              <text x="8" y="93" fontSize="9" letterSpacing="1.2" fontWeight="600" fill="#fff">SIMULATION</text>
            </g>
            <g transform={`translate(${nodeX(0)}, ${L3.y + 30})`}>
              <text fontSize="10" letterSpacing="1.6" fontWeight="600" fill="#f7e948">NIGHTLY RETRAINING · OPENCLAW · 02:00</text>
              {["The OpenClaw agent retrains on the new labels,", "compares the candidate with the production", "benchmark and asks a data scientist to sign off."].map((t, k) => (
                <text key={k} y={22 + k * 16} fontSize="11" fill="#ffffffaa">{t}</text>
              ))}
            </g>
            {/* Live benchmark card */}
            <g transform={`translate(${nodeX(0)}, ${L3.y + 112})`}>
              <rect width="250" height="104" fill="#1e1e1e" stroke={flash ? "#f7e948" : "#333"} strokeWidth="1.5" />
              <text x="14" y="22" fontSize="9" letterSpacing="1.4" fontWeight="600" fill="#ffffff99">PRODUCTION BENCHMARK</text>
              <text x="14" y="48" fontSize="20" fontWeight="500" fill="#f7e948">LightGBM {version}</text>
              <text x="14" y="70" fontSize="11" fill="#fff" fontFamily="var(--font-geist-mono)">
                recall {pct1(benchmark.recall)} · prec {pct1(benchmark.precision)}
              </text>
              <text x="14" y="88" fontSize="11" fill="#fff" fontFamily="var(--font-geist-mono)">
                F1 {benchmark.f1.toFixed(4)} · AUC {benchmark.auc.toFixed(4)}
              </text>
            </g>

            {/* Request enters */}
            <text x={2} y={CY - 10} fontSize="9" fill="#808080">request</text>
            <line x1={0} x2={nodeX(0) - 2} y1={CY} y2={CY} stroke="#000" strokeWidth="2" markerEnd="url(#arr)" />

            {/* Edges between main steps */}
            {Array.from({ length: MAIN_N - 1 }, (_, i) => {
              const x1 = nodeX(i) + NODE_W;
              const x2 = i === DECISION - 1 ? DIAMOND.x - DIAMOND.r : nodeX(i + 1);
              const on = passed(i) || streaming;
              return (
                <line
                  key={i}
                  x1={x1}
                  x2={x2 - 2}
                  y1={CY}
                  y2={CY}
                  stroke={on ? "#000" : "#bdbdbd"}
                  strokeWidth="2"
                  strokeDasharray={streaming ? "5 4" : undefined}
                  className={streaming ? "animate-flow" : undefined}
                  markerEnd={on ? "url(#arr)" : "url(#arr-idle)"}
                />
              );
            })}

            {/* Decision branches */}
            {(["fraud", "legit"] as const).map((r) => {
              const pts = BRANCH[r];
              const on = outcome === r || streaming;
              const end = { ...pts[pts.length - 1], y: TERM_Y - 2 };
              return (
                <path
                  key={r}
                  d={pathD([...pts.slice(0, -1), end])}
                  fill="none"
                  stroke={on ? (r === "fraud" ? "#c62828" : "#085850") : "#bdbdbd"}
                  strokeWidth="2"
                  strokeDasharray={streaming ? "5 4" : undefined}
                  className={streaming ? "animate-flow" : undefined}
                  markerEnd={on ? "url(#arr)" : "url(#arr-idle)"}
                />
              );
            })}

            {/* Feedback + nightly edges */}
            {Object.entries(EDGES).map(([key, pts]) => {
              const on = edgeOn(key);
              const tone = EDGE_TONE[key];
              const flow = streaming && STREAM_EDGES.includes(key);
              const common = {
                d: pathD(pts),
                fill: "none",
                strokeDasharray: flow ? "5 4" : undefined,
                className: flow ? "animate-flow" : undefined,
              };
              if (tone === "cross") {
                // casing: black outline keeps the yellow line visible on light and dark lanes
                return (
                  <g key={key}>
                    {on && <path d={common.d} fill="none" stroke="#000" strokeWidth="5" />}
                    <path {...common} stroke={on ? "#f7e948" : "#bdbdbd"} strokeWidth="2" markerEnd={on ? "url(#arr-yellow)" : "url(#arr-idle)"} />
                  </g>
                );
              }
              const color = tone === "dark" ? (on ? "#fff" : "#525252") : on ? "#000" : "#bdbdbd";
              const marker = tone === "dark" ? (on ? "url(#arr-white)" : "url(#arr-dim)") : on ? "url(#arr)" : "url(#arr-idle)";
              return <path key={key} {...common} stroke={color} strokeWidth="2" markerEnd={marker} />;
            })}

            {/* Edge labels: what data moves along the loop */}
            <text x={C(7) - 8} y={430} textAnchor="end" fontSize="9" fill="#525252">flagged payment</text>
            <text x={C(6) - 8} y={430} textAnchor="end" fontSize="9" fill="#525252">approved payment</text>
            <text x={(C(7) + nodeX(5) + 94) / 2} y={574} textAnchor="middle" fontSize="9" fill="#525252">label + features</text>
            <text x={nodeX(5) + 32} y={594} textAnchor="end" fontSize="9" fill="#525252">new labels</text>
            <text x={CMP.x - CMP.r - 6} y={684} textAnchor="end" fontSize="10" fontWeight="600" fill="#fff">yes</text>
            <text x={CMP.x + 8} y={772} fontSize="10" fontWeight="600" fill="#fff">no</text>
            <text x={C(2) + 8} y={292} fontSize="9" fontWeight="600" fill="#000">deploy new model version → step {pad(MODEL)}</text>

            {/* Stream dots — drawn before the steps so they pass "through" each box */}
            {dotPos.map((dot) => (
              <circle key={dot.id} cx={dot.p.x} cy={dot.p.y} r="5" fill={dot.color} stroke="#fff" strokeWidth="1.5" />
            ))}

            <text x={DIAMOND.x + 8} y={296} fontSize="10" fontWeight="600" fill="#c62828">p ≥ 0.5</text>
            <text x={TERM_APPROVE.c + 8} y={296} fontSize="10" fontWeight="600" fill="#085850">p &lt; 0.5</text>

            {/* Main step nodes 01–07 */}
            {STAGES.slice(0, DECISION).map((s, i) => {
              const isActive = active === i || (i === MODEL && flash);
              const done = passed(i);
              const isSel = shown === i;
              const sum = done || active === i ? nodeSummary(i, raw, data) : "";
              return (
                <g key={i} onClick={() => setSelected(i)} className="cursor-pointer">
                  <rect
                    x={nodeX(i)}
                    y={ROW_Y}
                    width={NODE_W}
                    height={NODE_H}
                    fill={isActive ? "#f7e948" : "#fff"}
                    stroke={isActive || done || isSel ? "#000" : "#e7e6e4"}
                    strokeWidth={isSel ? 3 : 1.5}
                    style={{ transition: "fill 0.3s" }}
                  />
                  <text x={nodeX(i) + 12} y={ROW_Y + 20} fontSize="10" fontFamily="var(--font-geist-mono)" fill={isActive || done ? "#000" : "#808080"}>
                    {String(i + 1).padStart(2, "0")}
                    {i === MODEL ? ` · ${version}` : ""}
                  </text>
                  {s.title.map((t, k) => (
                    <text key={k} x={nodeX(i) + 12} y={ROW_Y + 44 + k * 16} fontSize="13" fontWeight="500" fill={isActive || done ? "#000" : "#525252"}>
                      {t}
                    </text>
                  ))}
                  <text x={nodeX(i) + 12} y={ROW_Y + NODE_H - 12} fontSize="10" fontFamily="var(--font-geist-mono)" fill="#50327b" fontWeight="600">
                    {sum}
                  </text>
                </g>
              );
            })}

            {/* Decision diamond */}
            <g onClick={() => setSelected(DECISION)} className="cursor-pointer">
              <polygon
                points={`${DIAMOND.x},${CY - DIAMOND.r} ${DIAMOND.x + DIAMOND.r},${CY} ${DIAMOND.x},${CY + DIAMOND.r} ${DIAMOND.x - DIAMOND.r},${CY}`}
                fill={active === DECISION ? "#f7e948" : active === DONE ? "#000" : "#fff"}
                stroke={active >= DECISION || shown === DECISION ? "#000" : "#e7e6e4"}
                strokeWidth={shown === DECISION ? 3 : 1.5}
              />
              <text x={DIAMOND.x} y={CY - 6} textAnchor="middle" fontSize="12" fontWeight="600" fill={active === DONE ? "#f7e948" : "#000"}>
                Decision
              </text>
              <text x={DIAMOND.x} y={CY + 12} textAnchor="middle" fontSize="11" fontFamily="var(--font-geist-mono)" fill={active === DONE ? "#fff" : "#000"}>
                p ≥ 0.5 ?
              </text>
              <text x={DIAMOND.x - DIAMOND.r + 4} y={ROW_Y - 2} fontSize="10" fontFamily="var(--font-geist-mono)" fill={active >= DECISION ? "#000" : "#808080"}>
                {pad(DECISION)}
              </text>
            </g>

            {/* Terminals */}
            {([
              { key: "legit", x: TERM_APPROVE.x, title: "Approve", sub: "payment authorised", color: "#085850", n: counts.legit },
              { key: "fraud", x: TERM_FRAUD.x, title: "Flag as fraud", sub: "blocked for review", color: "#c62828", n: counts.fraud },
            ] as const).map((t) => {
              const hit = outcome === t.key;
              return (
                <g key={t.key} onClick={() => setSelected(DECISION)} className="cursor-pointer">
                  <rect x={t.x} y={TERM_Y} width={NODE_W} height={TERM_H} fill={hit ? t.color : "#fff"} stroke={t.color} strokeWidth="2" />
                  <text x={t.x + 12} y={TERM_Y + 24} fontSize="13" fontWeight="600" fill={hit ? "#fff" : t.color}>
                    {t.title}
                  </text>
                  <text x={t.x + 12} y={TERM_Y + 41} fontSize="10" fill={hit ? "#fff" : "#525252"}>
                    {t.sub}
                  </text>
                  {(streaming || t.n > 0) && (
                    <text x={t.x + 12} y={TERM_Y + 62} fontSize="12" fontWeight="600" fill={hit ? "#fff" : "#000"} fontFamily="var(--font-geist-mono)">
                      {t.n.toLocaleString()} req
                    </text>
                  )}
                </g>
              );
            })}

            {/* Loop nodes 09–16 */}
            {Object.entries(LOOP_BOX).map(([k, b]) => {
              const i = Number(k);
              const night = i >= ST.retrain;
              const isActive = loopActive === i || (i === ST.store && streaming && totalLabels(labels) > 0 && loopActive === null);
              const done = visited.includes(i);
              const isSel = shown === i;
              const sum = loopSummary(i);
              const stroke = isSel ? (night ? "#f7e948" : "#000") : done || isActive ? (night ? "#f7e948" : "#000") : night ? "#525252" : "#bdbdbd";
              return (
                <g key={i} onClick={() => setSelected(i)} className="cursor-pointer">
                  <rect
                    x={b.x}
                    y={b.y}
                    width={NODE_W}
                    height={b.h}
                    fill={loopActive === i ? "#f7e948" : "#fff"}
                    stroke={stroke}
                    strokeWidth={isSel ? 3 : 1.5}
                    style={{ transition: "fill 0.3s" }}
                  />
                  <text x={b.x + 12} y={b.y + 18} fontSize="10" fontFamily="var(--font-geist-mono)" fill={done || isActive ? "#000" : "#808080"}>
                    {String(i + 1).padStart(2, "0")}
                  </text>
                  {STAGES[i].title.map((t, j) => (
                    <text key={j} x={b.x + 12} y={b.y + (b.h > 70 ? 40 : 34) + j * 15} fontSize="12" fontWeight="500" fill="#000">
                      {t}
                    </text>
                  ))}
                  {b.h > 70 && (
                    <text x={b.x + 12} y={b.y + b.h - 10} fontSize="10" fontFamily="var(--font-geist-mono)" fill="#50327b" fontWeight="600">
                      {sum}
                    </text>
                  )}
                </g>
              );
            })}

            {/* Benchmark comparison diamond */}
            <g onClick={() => setSelected(CMP_STAGE)} className="cursor-pointer">
              <polygon
                points={`${CMP.x},${CMP.y - CMP.r} ${CMP.x + CMP.r},${CMP.y} ${CMP.x},${CMP.y + CMP.r} ${CMP.x - CMP.r},${CMP.y}`}
                fill={loopActive === CMP_STAGE ? "#f7e948" : "#fff"}
                stroke={shown === CMP_STAGE || visited.includes(CMP_STAGE) ? "#f7e948" : "#525252"}
                strokeWidth={shown === CMP_STAGE ? 3 : 1.5}
              />
              <text x={CMP.x} y={CMP.y - 4} textAnchor="middle" fontSize="11" fontWeight="600" fill="#000">Better than</text>
              <text x={CMP.x} y={CMP.y + 11} textAnchor="middle" fontSize="11" fontWeight="600" fill="#000">benchmark?</text>
              <text x={CMP.x - CMP.r + 2} y={CMP.y - CMP.r + 4} fontSize="10" fontFamily="var(--font-geist-mono)" fill="#ffffffaa">{pad(ST.compare)}</text>
            </g>

            {/* Payload tag above the active main step */}
            {tagStage >= 0 && (
              <g style={{ transform: `translate(${tagX}px, 44px)`, transition: "transform 0.5s ease" }}>
                <rect width="160" height="52" fill="#000" />
                <text x="10" y="20" fontSize="10" fontFamily="var(--font-geist-mono)" fill="#ffffff99">
                  IN&nbsp; {tagIn}
                </text>
                <text x="10" y="38" fontSize="10" fontFamily="var(--font-geist-mono)" fill="#f7e948" fontWeight="600">
                  OUT {tagOut}
                </text>
                <polygon
                  points={`${nodeC(tagStage).x - tagX - 7},52 ${nodeC(tagStage).x - tagX + 7},52 ${nodeC(tagStage).x - tagX},60`}
                  fill="#000"
                  style={{ transition: "all 0.5s ease" }}
                />
              </g>
            )}

            {/* The traced request */}
            {packet && (
              <g transform={`translate(${packet.x}, ${packet.y})`}>
                <circle r="16" fill="#f7e948" opacity="0.35" className="animate-ping-slow" />
                <circle r="9" fill="#f7e948" stroke="#000" strokeWidth="2.5" />
              </g>
            )}
          </svg>

          {/* Legend + feedback controls */}
          <div className="border-t border-line px-4 md:px-6 py-4 flex flex-col xl:flex-row xl:items-center xl:justify-between gap-4">
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-muted">
              <span className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-brand border-2 border-black" />one request</span>
              <span className="flex items-center gap-2"><span className="w-4 h-0.5 bg-black" />data handed to the next step</span>
              <span className="flex items-center gap-2"><span className="w-3 h-3 bg-black" />IN / OUT payload tag</span>
              <span className="flex items-center gap-2"><span className="w-3 h-3 bg-brand border border-black rotate-45 scale-75" />decision point</span>
              <span>Click any step to inspect it.</span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] uppercase tracking-[0.08em] text-muted mr-1">
                Feedback store: <strong className="text-black">{totalLabels(labels)}</strong> labels
              </span>
              {caseAwaiting && outcome === "fraud" && (
                <>
                  <ActionButton variant="fraud" disabled={busy} onClick={() => labelCase("confirmed")}>Analyst: confirm fraud</ActionButton>
                  <ActionButton variant="legit" disabled={busy} onClick={() => labelCase("false_positive")}>Analyst: actually legit</ActionButton>
                </>
              )}
              {caseAwaiting && outcome === "legit" && (
                <>
                  <ActionButton variant="fraud" disabled={busy} onClick={() => labelCase("missed_fraud")}>Customer complaint</ActionButton>
                  <ActionButton variant="outline" disabled={busy} onClick={() => labelCase("no_complaint")}>No complaint</ActionButton>
                </>
              )}
              {nightly?.status === "awaiting_ds" ? (
                <>
                  <ActionButton variant="primary" disabled={busy} onClick={() => dsDecision(true)}>Data scientist: approve &amp; publish</ActionButton>
                  <ActionButton variant="outline" disabled={busy} onClick={() => dsDecision(false)}>Reject</ActionButton>
                </>
              ) : (
                <ActionButton variant="dark" disabled={busy || totalLabels(labels) === 0} onClick={runNightly}>Run nightly retraining</ActionButton>
              )}
            </div>
          </div>
        </div>

        {/* ── Stage detail ── */}
        <div className="mt-6">
          <StageDetail stage={shown} d={data} raw={raw} ctx={ctx} />
        </div>

        {/* Stage quick-nav */}
        <div className="mt-4 flex flex-wrap gap-1">
          {STAGES.map((s, i) => (
            <button
              key={i}
              onClick={() => setSelected(i)}
              disabled={i <= DECISION && (!raw || (i > 0 && !data))}
              className={`h-9 px-3 text-[11px] uppercase tracking-[0.08em] font-medium ${
                shown === i
                  ? "bg-black text-brand"
                  : i > DECISION
                  ? "bg-sand border border-line text-muted hover:text-black hover:border-black"
                  : "bg-card border border-line text-muted hover:text-black hover:border-black"
              } disabled:opacity-30`}
            >
              <span className="font-mono mr-1.5">{String(i + 1).padStart(2, "0")}</span>
              {s.title.join(" ")}
            </button>
          ))}
        </div>
      </main>

      <footer className="bg-black text-white/50 text-[10px] uppercase tracking-[0.08em] mt-8">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-5 text-center">
          Educational PoC — RSU AI &amp; ML Final Project · LightGBM Fraud Detection ·
          Dataset: Kaggle dhruvb2028/credit-card-fraud-dataset
        </div>
      </footer>
    </div>
  );
}
