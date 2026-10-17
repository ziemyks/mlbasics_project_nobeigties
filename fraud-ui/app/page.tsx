"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import TopBar from "./components/TopBar";

// ─── Types ───────────────────────────────────────────────────────────────────

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

interface EngineeredFeatures {
  hour: number;
  day_of_week: number;
  month: number;
  age: number;
  distance_km: number;
  log_amt: number;
  category: string;
  state: string;
}

interface PredictionResult {
  is_fraud: number;
  fraud_probability: number;
  verdict: string;
  threshold: number;
  engineered_features: EngineeredFeatures;
}

interface ProcessedTransaction {
  id: string;
  raw: RawTransaction;
  prediction: PredictionResult | null;
  stage: "incoming" | "engineering" | "scoring" | "done";
  timestamp: number;
}

const API = "http://127.0.0.1:5050";

// Interval between transactions (ms) — slider positions, slow → fast
const INTERVALS = [5000, 3000, 2000, 1000, 500, 200, 100, 50, 20, 10];
const MAX_IN_FLIGHT = 16; // max transactions processed in parallel

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

// ─── Helper Components ───────────────────────────────────────────────────────

function Modal({
  isOpen,
  onClose,
  title,
  children,
}: {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}) {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 animate-in fade-in duration-200">
      <div className="bg-card w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col shadow-2xl">
        <div className="px-6 py-4 bg-black text-white flex justify-between items-center gap-4">
          <h3 className="font-medium text-sm uppercase tracking-[0.08em] truncate">
            {title}
          </h3>
          <button
            onClick={onClose}
            className="w-8 h-8 shrink-0 flex items-center justify-center bg-brand text-black font-semibold hover:bg-white transition-colors"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div className="p-6 overflow-y-auto flex-1">{children}</div>
      </div>
    </div>
  );
}

function Badge({
  children,
  variant,
}: {
  children: React.ReactNode;
  variant: "fraud" | "legit" | "neutral" | "processing";
}) {
  const cls = {
    fraud: "bg-fraud text-white",
    legit: "bg-legit text-white",
    neutral: "bg-sand text-muted",
    processing: "bg-brand text-black",
  }[variant];
  return (
    <span
      className={`px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] ${cls}`}
    >
      {children}
    </span>
  );
}

function Card({
  title,
  step,
  children,
  className = "",
}: {
  title: string;
  step?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`bg-card border border-line p-5 ${className}`}>
      <h2 className="text-[11px] font-semibold text-black uppercase tracking-[0.12em] mb-4 flex items-center gap-3">
        {step ? (
          <span className="bg-black text-brand font-mono text-[10px] px-1.5 py-0.5">
            {step}
          </span>
        ) : (
          <span className="w-2.5 h-2.5 bg-brand" />
        )}
        {title}
      </h2>
      {children}
    </div>
  );
}

function ProgressPipeline({ stage }: { stage: ProcessedTransaction["stage"] }) {
  const steps = [
    { key: "incoming", label: "Incoming" },
    { key: "engineering", label: "Feature Engineering" },
    { key: "scoring", label: "Model Scoring" },
    { key: "done", label: "Result" },
  ];
  const stageIdx = steps.findIndex((s) => s.key === stage);

  return (
    <div className="grid grid-cols-4 gap-1 mb-4">
      {steps.map((step, i) => (
        <div
          key={step.key}
          className={`flex items-center gap-2 px-3 py-2 text-[11px] font-medium uppercase tracking-[0.06em] transition-colors ${
            i < stageIdx
              ? "bg-black text-white"
              : i === stageIdx
              ? "bg-brand text-black"
              : "bg-sand text-subtle"
          }`}
        >
          <span className="font-mono">{String(i + 1).padStart(2, "0")}</span>
          <span className="hidden sm:inline truncate">{step.label}</span>
        </div>
      ))}
    </div>
  );
}

function RawTransactionPanel({ tx }: { tx: RawTransaction }) {
  const fields = [
    { label: "Merchant", value: tx.merchant },
    { label: "Category", value: tx.category.replace("_", " ") },
    { label: "Amount", value: `$${tx.amt.toFixed(2)}` },
    { label: "Date/Time", value: tx.trans_date_trans_time },
    { label: "City", value: `${tx.city}, ${tx.state}` },
    { label: "Population", value: tx.city_pop.toLocaleString() },
    { label: "Cardholder Location", value: `${tx.lat}°, ${tx.long}°` },
    { label: "Merchant Location", value: `${tx.merch_lat}°, ${tx.merch_long}°` },
    { label: "DOB", value: tx.dob },
    { label: "Job", value: tx.job },
  ];

  return (
    <div className="divide-y divide-line text-xs">
      {fields.map((f) => (
        <div key={f.label} className="flex justify-between py-1.5">
          <span className="text-subtle">{f.label}</span>
          <span className="text-black font-medium text-right ml-2 truncate max-w-[180px]">
            {f.value}
          </span>
        </div>
      ))}
    </div>
  );
}

function EngineeringPanel({ features }: { features: EngineeredFeatures }) {
  const rows = [
    {
      label: "hour",
      raw: "trans_date_trans_time",
      computed: features.hour.toString(),
      desc: "Extract hour from timestamp",
    },
    {
      label: "day_of_week",
      raw: "trans_date_trans_time",
      computed: `${features.day_of_week} (${DAY_NAMES[features.day_of_week]})`,
      desc: "Day of week (0=Mon)",
    },
    {
      label: "month",
      raw: "trans_date_trans_time",
      computed: features.month.toString(),
      desc: "Month extracted",
    },
    {
      label: "age",
      raw: "dob → now",
      computed: `${features.age} years`,
      desc: "(trans_time − dob) / 365",
    },
    {
      label: "distance_km",
      raw: "lat/long + merch_lat/long",
      computed: `${features.distance_km} km`,
      desc: "Haversine distance",
    },
    {
      label: "log_amt",
      raw: `amt`,
      computed: features.log_amt.toFixed(4),
      desc: "ln(1 + amt)",
    },
    {
      label: "category OHE",
      raw: features.category,
      computed: `category_${features.category} = 1`,
      desc: "One-hot encoded",
    },
    {
      label: "state OHE",
      raw: features.state,
      computed: `state_${features.state} = 1`,
      desc: "One-hot encoded",
    },
  ];

  return (
    <div>
      <p className="text-[10px] text-purple mb-2 font-mono">
        {"// Same pipeline as training: datetime → age → haversine → log → OHE"}
      </p>
      <div className="space-y-1">
        {rows.map((r) => (
          <div
            key={r.label}
            className="flex items-center text-xs gap-2 bg-sand px-2 py-1"
          >
            <span className="text-black font-mono font-medium w-28 shrink-0">
              {r.label}
            </span>
            <span className="text-subtle text-[10px] w-32 shrink-0 truncate">
              {r.desc}
            </span>
            <span className="text-legit font-mono font-semibold ml-auto truncate">
              {r.computed}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ProbabilityBar({ probability }: { probability: number }) {
  const pct = (probability * 100).toFixed(2);
  const isFraud = probability >= 0.5;

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-baseline text-xs">
        <span className="text-muted uppercase tracking-[0.08em] text-[11px]">
          Fraud Probability
        </span>
        <span
          className={`tabular-nums font-semibold text-lg ${
            isFraud ? "text-fraud" : "text-legit"
          }`}
        >
          {pct}%
        </span>
      </div>
      <div className="w-full h-3 bg-sand overflow-hidden relative mt-5">
        {/* Threshold marker at 50% */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-black z-10"
          style={{ left: "50%" }}
        />
        <div
          className={`h-full transition-all duration-700 ${
            isFraud ? "bg-fraud" : "bg-legit"
          }`}
          style={{ width: `${Math.max(probability * 100, 1)}%` }}
        />
      </div>
      <div className="relative flex justify-between text-[10px] text-subtle font-mono">
        <span>0% Legitimate</span>
        <span
          className="absolute text-black font-semibold"
          style={{ left: "50%", transform: "translateX(-50%)" }}
        >
          threshold 50%
        </span>
        <span>100% Fraud</span>
      </div>
    </div>
  );
}

interface Stats {
  total: number;
  frauds: number;
  probSum: number;
}

function StatsPanel({
  stats,
  throughput,
}: {
  stats: Stats;
  throughput: number;
}) {
  const { total, frauds } = stats;
  const legits = total - frauds;
  const avgProb = total > 0 ? stats.probSum / total : 0;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-5 gap-px bg-line border border-line">
      {[
        { label: "Processed", value: total.toLocaleString(), color: "text-black" },
        { label: "Legitimate", value: legits.toLocaleString(), color: "text-legit" },
        { label: "Fraudulent", value: frauds.toLocaleString(), color: "text-fraud" },
        {
          label: "Avg Probability",
          value: `${(avgProb * 100).toFixed(1)}%`,
          color: "text-purple",
        },
      ].map((s) => (
        <div key={s.label} className="bg-card p-5">
          <div className="text-[10px] text-subtle uppercase tracking-[0.12em] font-medium">
            {s.label}
          </div>
          <div className={`text-3xl font-medium tabular-nums mt-2 ${s.color}`}>
            {s.value}
          </div>
        </div>
      ))}
      <div className="bg-black p-5 col-span-2 sm:col-span-1">
        <div className="text-[10px] text-white/60 uppercase tracking-[0.12em] font-medium">
          Throughput
        </div>
        <div className="text-3xl font-medium tabular-nums mt-2 text-brand">
          {throughput.toFixed(1)}
          <span className="text-sm text-white/70 ml-1.5">tx/s</span>
        </div>
      </div>
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────

export default function FraudDetectionPage() {
  const [history, setHistory] = useState<ProcessedTransaction[]>([]);
  const [current, setCurrent] = useState<ProcessedTransaction | null>(null);
  const [selectedTx, setSelectedTx] = useState<ProcessedTransaction | null>(
    null
  );
  const [running, setRunning] = useState(false);
  const runningRef = useRef(false);
  const [speedIdx, setSpeedIdx] = useState(1); // index into INTERVALS (3s)
  const speed = INTERVALS[speedIdx]; // ms between transactions
  const speedRef = useRef(speed);
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const intervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [stats, setStats] = useState<Stats>({ total: 0, frauds: 0, probSum: 0 });
  const [throughput, setThroughput] = useState(0);
  const completedAtRef = useRef<number[]>([]);
  const inFlightRef = useRef(0);
  const latestIdRef = useRef<string | null>(null);

  // Keep the running loop in sync with the slider
  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  // Measured throughput: completions over the last 3 seconds
  useEffect(() => {
    const id = setInterval(() => {
      const now = Date.now();
      completedAtRef.current = completedAtRef.current.filter(
        (t) => now - t < 3000
      );
      setThroughput(completedAtRef.current.length / 3);
    }, 500);
    return () => clearInterval(id);
  }, []);

  // Check API on mount
  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false));
  }, []);

  const processOne = useCallback(async () => {
    // Visual stage delays shrink with the interval, so the animation
    // never limits how many transactions per second we can push through
    const step = (ms: number) => sleep(Math.min(ms, speedRef.current / 5));
    // With parallel transactions, only the newest one drives the live panel
    const show = (tx: ProcessedTransaction) => {
      if (latestIdRef.current === tx.id) setCurrent({ ...tx });
    };

    // 1. Generate transaction
    const genRes = await fetch(`${API}/generate`);
    const raw: RawTransaction = await genRes.json();
    const txId = raw.trans_num;
    latestIdRef.current = txId;

    const tx: ProcessedTransaction = {
      id: txId,
      raw,
      prediction: null,
      stage: "incoming",
      timestamp: Date.now(),
    };
    show(tx);

    // 2. Feature engineering (visual delay)
    await step(600);
    tx.stage = "engineering";
    show(tx);

    // 3. Model scoring
    await step(600);
    tx.stage = "scoring";
    show(tx);

    const predRes = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(raw),
    });
    const pred: PredictionResult = await predRes.json();

    await step(400);
    tx.prediction = pred;
    tx.stage = "done";
    show(tx);

    // 4. Move to history
    await step(500);
    setHistory((prev) => [tx, ...prev].slice(0, 50)); // keep last 50
    setStats((s) => ({
      total: s.total + 1,
      frauds: s.frauds + pred.is_fraud,
      probSum: s.probSum + pred.fraud_probability,
    }));
    completedAtRef.current.push(Date.now());
    // We intentionally do NOT set current to null here.
  }, []);

  const startStream = useCallback(() => {
    setRunning(true);
    runningRef.current = true;
    // Fire a new transaction every `speed` ms without waiting for the
    // previous one to finish, capped at MAX_IN_FLIGHT parallel requests
    const loop = () => {
      if (!runningRef.current) return;
      if (inFlightRef.current < MAX_IN_FLIGHT) {
        inFlightRef.current++;
        processOne()
          .catch(() => {
            // API might be down, keep trying
          })
          .finally(() => {
            inFlightRef.current--;
          });
      }
      intervalRef.current = setTimeout(loop, speedRef.current);
    };
    loop();
  }, [processOne]);

  const stopStream = useCallback(() => {
    setRunning(false);
    runningRef.current = false;
    if (intervalRef.current) {
      clearTimeout(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <TopBar apiOk={apiOk} />

      <main className="flex-1 w-full max-w-7xl mx-auto px-4 md:px-6 py-8 md:py-10">
        {/* Hero */}
        <section className="mb-8 flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6">
          <div>
            <p className="text-[11px] uppercase tracking-[0.16em] text-muted font-medium mb-3">
              LightGBM · Educational PoC
            </p>
            <h1 className="text-3xl md:text-5xl font-medium leading-[1.1] tracking-tight text-black">
              Real-time{" "}
              <span className="bg-brand px-2 -mx-0.5 whitespace-nowrap">
                fraud detection
              </span>
            </h1>
            <p className="text-muted text-sm mt-4 max-w-xl">
              AI generates synthetic transactions → Feature engineering →
              LightGBM scores → Fraud/Legitimate verdict
            </p>
          </div>

          {/* Controls */}
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={running ? stopStream : startStream}
              disabled={apiOk !== true}
              className={`h-12 px-6 font-medium text-xs uppercase tracking-[0.1em] transition-colors ${
                running
                  ? "bg-black text-brand hover:bg-neutral-800"
                  : "bg-brand text-black hover:bg-black hover:text-brand"
              } disabled:opacity-30 disabled:cursor-not-allowed`}
            >
              {running ? "■ Stop Stream" : "▶ Start Auto-Stream"}
            </button>

            <button
              onClick={processOne}
              disabled={apiOk !== true || running}
              className="h-12 px-6 font-medium text-xs uppercase tracking-[0.1em] bg-transparent text-black border border-black hover:bg-black hover:text-white transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Process One
            </button>

            <div className="h-12 flex items-center gap-3 px-4 bg-card border border-line text-[11px] uppercase tracking-[0.08em] text-muted">
              <label htmlFor="speed">Speed</label>
              <input
                id="speed"
                type="range"
                min={0}
                max={INTERVALS.length - 1}
                step={1}
                value={speedIdx}
                onChange={(e) => setSpeedIdx(Number(e.target.value))}
                className="w-28 accent-black"
              />
              <span className="font-mono text-black font-semibold w-16 normal-case tracking-normal">
                {1000 / speed < 1
                  ? (1000 / speed).toFixed(1)
                  : (1000 / speed).toFixed(0)}{" "}
                tx/s
              </span>
            </div>
          </div>
        </section>

        {/* Stats Bar */}
        <div className="mb-6">
          <StatsPanel stats={stats} throughput={throughput} />
        </div>

        {/* Main Layout: Live Processing (left) + History (right) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* ── LIVE PROCESSING PANEL ── */}
          <div className="lg:col-span-2 space-y-4">
            {current ? (
              <div className="animate-slide-in">
                {/* Pipeline Progress */}
                <ProgressPipeline stage={current.stage} />

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Raw Transaction */}
                  <Card title="Raw Transaction" step="01">
                    <RawTransactionPanel tx={current.raw} />
                  </Card>

                  {/* Feature Engineering */}
                  <Card
                    title="Feature Engineering"
                    step="02"
                    className={
                      current.stage === "engineering"
                        ? "animate-pulse-glow"
                        : ""
                    }
                  >
                    {current.prediction?.engineered_features ? (
                      <EngineeringPanel
                        features={current.prediction.engineered_features}
                      />
                    ) : current.stage === "incoming" ? (
                      <p className="text-xs text-subtle italic">
                        Waiting for incoming transaction…
                      </p>
                    ) : (
                      <div className="flex items-center gap-2 text-xs text-black">
                        <span className="w-2 h-2 bg-brand animate-pulse" />
                        Processing features…
                      </div>
                    )}
                  </Card>

                  {/* Model Scoring */}
                  <Card
                    title="LightGBM Scoring"
                    step="03"
                    className={`md:col-span-2 ${
                      current.stage === "scoring" ? "animate-pulse-glow" : ""
                    }`}
                  >
                    {current.prediction ? (
                      <div className="space-y-4">
                        <ProbabilityBar
                          probability={current.prediction.fraud_probability}
                        />
                        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                          <div className="text-xs text-muted">
                            <span className="font-mono">
                              model.predict_proba(features)
                            </span>{" "}
                            →{" "}
                            <span className="font-semibold text-black font-mono">
                              {(
                                current.prediction.fraud_probability * 100
                              ).toFixed(4)}
                              %
                            </span>
                          </div>
                          <Badge
                            variant={
                              current.prediction.is_fraud ? "fraud" : "legit"
                            }
                          >
                            {current.prediction.is_fraud
                              ? "Fraud detected"
                              : "Legitimate"}
                          </Badge>
                        </div>

                        {/* Educational explanation */}
                        <div className="p-4 bg-sand border-l-4 border-brand text-[11px] text-muted leading-relaxed">
                          <strong className="text-black uppercase tracking-[0.08em] text-[10px]">
                            How it works
                          </strong>
                          <p className="mt-1">
                            The LightGBM model analyzes 39 engineered features
                            (numerical + one-hot encoded) and outputs a fraud
                            probability between 0 and 1. If probability ≥{" "}
                            <span className="font-semibold text-black">0.50</span>{" "}
                            (threshold), the transaction is flagged as fraud.
                          </p>
                          {current.prediction.is_fraud === 1 && (
                            <span className="text-fraud font-medium block mt-1">
                              Suspicious signals: high amount ($
                              {current.raw.amt.toFixed(2)}), distance{" "}
                              {current.prediction.engineered_features.distance_km}{" "}
                              km, hour{" "}
                              {current.prediction.engineered_features.hour}:00
                            </span>
                          )}
                          <button
                            onClick={() => setSelectedTx(current)}
                            className="mt-3 block w-full h-10 bg-black text-white text-[11px] uppercase tracking-[0.1em] font-medium hover:bg-brand hover:text-black transition-colors"
                          >
                            View Deep Dive Details →
                          </button>
                        </div>
                      </div>
                    ) : current.stage === "scoring" ? (
                      <div className="flex items-center gap-2 text-xs text-black">
                        <span className="w-2 h-2 bg-brand animate-pulse" />
                        Running LightGBM inference…
                      </div>
                    ) : (
                      <p className="text-xs text-subtle italic">
                        Waiting for features…
                      </p>
                    )}
                  </Card>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-72 bg-card border border-line text-center px-6">
                <div className="w-12 h-12 bg-brand mb-5" />
                <p className="text-sm text-black">
                  Press <strong>&quot;Start Auto-Stream&quot;</strong> or{" "}
                  <strong>&quot;Process One&quot;</strong> to begin
                </p>
                <p className="text-xs text-subtle mt-2">
                  Synthetic transactions will be generated and analyzed in real
                  time
                </p>
              </div>
            )}
          </div>

          {/* ── HISTORY PANEL ── */}
          <Card
            title="Transaction History"
            className="max-h-[640px] overflow-hidden flex flex-col"
          >
            <div className="overflow-y-auto flex-1 space-y-1.5 pr-1">
              {history.length === 0 ? (
                <p className="text-xs text-subtle italic text-center py-8">
                  No transactions processed yet
                </p>
              ) : (
                history.map((tx) => (
                  <div
                    key={tx.id}
                    onClick={() => setSelectedTx(tx)}
                    className={`p-2.5 border-l-4 bg-sand text-xs cursor-pointer hover:bg-brand/40 transition-colors animate-slide-in ${
                      tx.prediction?.is_fraud ? "border-fraud" : "border-legit"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-black truncate max-w-[140px]">
                        {tx.raw.merchant}
                      </span>
                      <Badge
                        variant={tx.prediction?.is_fraud ? "fraud" : "legit"}
                      >
                        {tx.prediction?.is_fraud ? "Fraud" : "OK"}
                      </Badge>
                    </div>
                    <div className="flex justify-between text-[10px] text-subtle">
                      <span>
                        {tx.raw.category.replace("_", " ")} · ${tx.raw.amt.toFixed(2)}
                      </span>
                      <span className="font-semibold text-black uppercase tracking-[0.08em]">
                        Details →
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </Card>
        </div>

        <Modal
          isOpen={!!selectedTx}
          onClose={() => setSelectedTx(null)}
          title={`Transaction Details: ${selectedTx?.id}`}
        >
          {selectedTx && (
            <div className="space-y-6">
              <div className="flex flex-wrap items-center justify-between gap-4 bg-sand p-5">
                <div>
                  <div className="text-[10px] text-subtle uppercase tracking-[0.12em] font-medium">
                    Prediction
                  </div>
                  <div
                    className={`text-2xl font-medium mt-1 ${
                      selectedTx.prediction?.is_fraud
                        ? "text-fraud"
                        : "text-legit"
                    }`}
                  >
                    {selectedTx.prediction?.is_fraud === 1 ? "Fraudulent" : "Legitimate"}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[10px] text-subtle uppercase tracking-[0.12em] font-medium">
                    Probability Score
                  </div>
                  <div className="text-2xl font-medium tabular-nums text-black mt-1">
                    {((selectedTx.prediction?.fraud_probability || 0) * 100).toFixed(4)}%
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <h4 className="text-[11px] text-black font-semibold uppercase tracking-[0.12em] mb-3 flex items-center gap-2 border-b border-line pb-2">
                    <span className="w-2.5 h-2.5 bg-brand" /> Raw Transaction Data
                  </h4>
                  <div className="bg-black p-4 font-mono text-[11px] h-[300px] overflow-auto">
                    <pre className="text-white/90">
                      {JSON.stringify(selectedTx.raw, null, 2)}
                    </pre>
                  </div>
                  <p className="mt-2 text-[10px] text-subtle italic">
                    Note: This is the raw JSON returned by the generator.
                  </p>
                </div>

                <div>
                  <h4 className="text-[11px] text-black font-semibold uppercase tracking-[0.12em] mb-3 flex items-center gap-2 border-b border-line pb-2">
                    <span className="w-2.5 h-2.5 bg-brand" /> Engineered Model Features
                  </h4>
                  <div className="bg-sand p-4 font-mono text-[11px] h-[300px] overflow-y-auto">
                    <div className="space-y-2">
                      {Object.entries(selectedTx.prediction?.engineered_features || {}).map(([key, value]) => (
                        <div key={key} className="flex justify-between border-b border-line pb-1">
                          <span className="text-muted">{key}</span>
                          <span className="text-black font-semibold">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <p className="mt-2 text-[10px] text-subtle italic">
                    Note: Features after datetime extraction, haversine distance, and log transformation.
                  </p>
                </div>
              </div>

              <div className="bg-brand p-5">
                <h5 className="text-black text-[11px] font-semibold uppercase tracking-[0.12em] mb-1">
                  Analysis Insight
                </h5>
                <p className="text-black text-xs leading-relaxed">
                  The LightGBM model evaluated this transaction.
                  {selectedTx.prediction?.fraud_probability && selectedTx.prediction.fraud_probability > 0.8 ? (
                    " The extremely high probability score suggests multiple severe anomalies in purchase location and transaction amount."
                  ) : selectedTx.prediction?.is_fraud ? (
                    " While the indicators were mixed, the mathematical combination of variables surpassed the risk threshold."
                  ) : (
                    " The behavior pattern is consistent with historical legitimate user habits in this geographical region."
                  )}
                </p>
              </div>
            </div>
          )}
        </Modal>

        {/* ── Architecture Diagram ── */}
        <div className="mt-8">
          <Card title="System Architecture">
            <div className="flex flex-wrap items-stretch justify-center gap-2 text-xs py-4">
              {[
                { label: "Synthetic\nGenerator", sub: "Python / Real stats" },
                { label: "Raw\nTransaction", sub: "15 fields" },
                { label: "Feature\nEngineering", sub: "→ 39 features" },
                { label: "LightGBM\nModel", sub: "predict_proba()" },
                { label: "Result\nVerdict", sub: "Fraud / Legit" },
              ].map((item, i, arr) => (
                <div key={item.label} className="flex items-center gap-2">
                  <div
                    className={`p-4 min-w-[120px] ${
                      i === arr.length - 1 ? "bg-black text-white" : "bg-sand text-black"
                    }`}
                  >
                    <div
                      className={`font-mono text-[10px] mb-2 ${
                        i === arr.length - 1 ? "text-brand" : "text-subtle"
                      }`}
                    >
                      {String(i + 1).padStart(2, "0")}
                    </div>
                    <div className="font-medium whitespace-pre-line leading-tight">
                      {item.label}
                    </div>
                    <div
                      className={`text-[10px] mt-1 ${
                        i === arr.length - 1 ? "text-white/60" : "text-subtle"
                      }`}
                    >
                      {item.sub}
                    </div>
                  </div>
                  {i < arr.length - 1 && (
                    <span className="text-black text-lg">→</span>
                  )}
                </div>
              ))}
            </div>
          </Card>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-black text-white/50 text-[10px] uppercase tracking-[0.08em]">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-5 text-center">
          Educational PoC — RSU AI &amp; ML Final Project · LightGBM Fraud Detection ·
          Dataset: Kaggle dhruvb2028/credit-card-fraud-dataset
        </div>
      </footer>
    </div>
  );
}
