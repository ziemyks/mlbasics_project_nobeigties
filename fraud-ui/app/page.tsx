"use client";

import { useState, useCallback, useRef, useEffect } from "react";

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

const DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

// ─── Helper Components ───────────────────────────────────────────────────────

function Badge({
  children,
  variant,
}: {
  children: React.ReactNode;
  variant: "fraud" | "legit" | "neutral" | "processing";
}) {
  const cls = {
    fraud: "bg-red-500/20 text-red-400 border-red-500/40",
    legit: "bg-green-500/20 text-green-400 border-green-500/40",
    neutral: "bg-slate-500/20 text-slate-400 border-slate-500/40",
    processing: "bg-blue-500/20 text-blue-400 border-blue-500/40",
  }[variant];
  return (
    <span className={`px-2 py-0.5 text-xs font-semibold rounded border ${cls}`}>
      {children}
    </span>
  );
}

function Card({
  title,
  icon,
  children,
  className = "",
}: {
  title: string;
  icon: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`bg-[#111827] border border-slate-700/50 rounded-xl p-4 ${className}`}
    >
      <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
        <span>{icon}</span>
        {title}
      </h2>
      {children}
    </div>
  );
}

function ProgressPipeline({ stage }: { stage: ProcessedTransaction["stage"] }) {
  const steps = [
    { key: "incoming", label: "Incoming", icon: "📥" },
    { key: "engineering", label: "Feature Engineering", icon: "⚙️" },
    { key: "scoring", label: "Model Scoring", icon: "🧠" },
    { key: "done", label: "Result", icon: "✅" },
  ];
  const stageIdx = steps.findIndex((s) => s.key === stage);

  return (
    <div className="flex items-center gap-1 mb-4">
      {steps.map((step, i) => (
        <div key={step.key} className="flex items-center gap-1">
          <div
            className={`flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium transition-all ${
              i < stageIdx
                ? "bg-green-500/20 text-green-400"
                : i === stageIdx
                ? "bg-blue-500/20 text-blue-300 animate-pulse-glow"
                : "bg-slate-800 text-slate-500"
            }`}
          >
            <span>{step.icon}</span>
            <span className="hidden sm:inline">{step.label}</span>
          </div>
          {i < steps.length - 1 && (
            <div
              className={`w-4 h-0.5 ${
                i < stageIdx ? "bg-green-500" : "bg-slate-700"
              }`}
            />
          )}
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
    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
      {fields.map((f) => (
        <div key={f.label} className="flex justify-between">
          <span className="text-slate-500">{f.label}</span>
          <span className="text-slate-200 font-mono text-right ml-2 truncate max-w-[140px]">
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
    <div className="space-y-1">
      <p className="text-[10px] text-blue-400 mb-2 font-mono">
        {"// Same pipeline as training: datetime → age → haversine → log → OHE → scale"}
      </p>
      <div className="space-y-1">
        {rows.map((r) => (
          <div
            key={r.label}
            className="flex items-center text-xs gap-2 bg-slate-800/60 rounded px-2 py-1"
          >
            <span className="text-blue-400 font-mono w-28 shrink-0">
              {r.label}
            </span>
            <span className="text-slate-500 text-[10px] w-32 shrink-0 truncate">
              {r.desc}
            </span>
            <span className="text-yellow-300 font-mono ml-auto">
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
      <div className="flex justify-between text-xs">
        <span className="text-slate-400">Fraud Probability</span>
        <span
          className={`font-mono font-bold ${
            isFraud ? "text-red-400" : "text-green-400"
          }`}
        >
          {pct}%
        </span>
      </div>
      <div className="w-full h-4 bg-slate-800 rounded-full overflow-hidden relative">
        {/* Threshold marker at 50% */}
        <div
          className="absolute top-0 bottom-0 w-px bg-yellow-500 z-10"
          style={{ left: "50%" }}
        />
        <div
          className="absolute -top-4 text-[9px] text-yellow-500 font-mono"
          style={{ left: "50%", transform: "translateX(-50%)" }}
        >
          threshold 50%
        </div>
        <div
          className={`h-full rounded-full transition-all duration-700 ${
            isFraud
              ? "bg-gradient-to-r from-red-600 to-red-400"
              : "bg-gradient-to-r from-green-700 to-green-400"
          }`}
          style={{ width: `${Math.max(probability * 100, 1)}%` }}
        />
      </div>
      <div className="flex justify-between text-[9px] text-slate-500 font-mono">
        <span>0% (Legitimate)</span>
        <span>100% (Fraud)</span>
      </div>
    </div>
  );
}

function StatsPanel({ history }: { history: ProcessedTransaction[] }) {
  const done = history.filter((t) => t.stage === "done" && t.prediction);
  const total = done.length;
  const frauds = done.filter((t) => t.prediction!.is_fraud === 1).length;
  const legits = total - frauds;
  const avgProb =
    total > 0
      ? done.reduce((s, t) => s + t.prediction!.fraud_probability, 0) / total
      : 0;

  return (
    <div className="grid grid-cols-4 gap-3 text-center">
      {[
        { label: "Processed", value: total.toString(), color: "text-blue-400" },
        { label: "Legitimate", value: legits.toString(), color: "text-green-400" },
        { label: "Fraudulent", value: frauds.toString(), color: "text-red-400" },
        {
          label: "Avg Prob",
          value: `${(avgProb * 100).toFixed(1)}%`,
          color: "text-yellow-400",
        },
      ].map((s) => (
        <div key={s.label} className="bg-slate-800/60 rounded-lg p-3">
          <div className={`text-xl font-bold font-mono ${s.color}`}>
            {s.value}
          </div>
          <div className="text-[10px] text-slate-500 uppercase mt-1">
            {s.label}
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────

export default function FraudDetectionPage() {
  const [history, setHistory] = useState<ProcessedTransaction[]>([]);
  const [current, setCurrent] = useState<ProcessedTransaction | null>(null);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(3000); // ms between transactions
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const intervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const historyEndRef = useRef<HTMLDivElement>(null);

  // Check API on mount
  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false));
  }, []);

  // Auto-scroll history
  useEffect(() => {
    historyEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history]);

  const processOne = useCallback(async () => {
    // 1. Generate transaction
    const genRes = await fetch(`${API}/generate`);
    const raw: RawTransaction = await genRes.json();
    const txId = raw.trans_num;

    const tx: ProcessedTransaction = {
      id: txId,
      raw,
      prediction: null,
      stage: "incoming",
      timestamp: Date.now(),
    };
    setCurrent({ ...tx });

    // 2. Feature engineering (visual delay)
    await new Promise((r) => setTimeout(r, 600));
    tx.stage = "engineering";
    setCurrent({ ...tx });

    // 3. Model scoring
    await new Promise((r) => setTimeout(r, 600));
    tx.stage = "scoring";
    setCurrent({ ...tx });

    const predRes = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(raw),
    });
    const pred: PredictionResult = await predRes.json();

    await new Promise((r) => setTimeout(r, 400));
    tx.prediction = pred;
    tx.stage = "done";
    setCurrent({ ...tx });

    // 4. Move to history
    await new Promise((r) => setTimeout(r, 500));
    setHistory((prev) => [tx, ...prev].slice(0, 50)); // keep last 50
    setCurrent(null);
  }, []);

  const startStream = useCallback(() => {
    setRunning(true);
    const loop = async () => {
      try {
        await processOne();
      } catch {
        // API might be down, keep trying
      }
      intervalRef.current = setTimeout(loop, speed);
    };
    loop();
  }, [processOne, speed]);

  const stopStream = useCallback(() => {
    setRunning(false);
    if (intervalRef.current) clearTimeout(intervalRef.current);
  }, []);

  return (
    <main className="min-h-screen p-4 md:p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6 text-center">
        <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
          🛡️ LightGBM Fraud Detection — Educational PoC
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          AI generates synthetic transactions → Feature engineering → LightGBM
          scores → Fraud/Legitimate verdict
        </p>

        {/* API Status */}
        <div className="mt-3 flex items-center justify-center gap-2 text-xs">
          <span
            className={`w-2 h-2 rounded-full ${
              apiOk === true
                ? "bg-green-400"
                : apiOk === false
                ? "bg-red-400"
                : "bg-yellow-400 animate-pulse"
            }`}
          />
          <span className="text-slate-500">
            {apiOk === true
              ? "API Connected (localhost:5050)"
              : apiOk === false
              ? "API Offline — run: python api.py"
              : "Checking API…"}
          </span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center justify-center gap-3 mb-6">
        <button
          onClick={running ? stopStream : startStream}
          disabled={apiOk !== true}
          className={`px-5 py-2 rounded-lg font-semibold text-sm transition-all ${
            running
              ? "bg-red-500/20 text-red-400 border border-red-500/40 hover:bg-red-500/30"
              : "bg-blue-500/20 text-blue-400 border border-blue-500/40 hover:bg-blue-500/30"
          } disabled:opacity-30 disabled:cursor-not-allowed`}
        >
          {running ? "⏹ Stop Stream" : "▶ Start Auto-Stream"}
        </button>

        <button
          onClick={processOne}
          disabled={apiOk !== true || running}
          className="px-5 py-2 rounded-lg font-semibold text-sm bg-slate-700/50 text-slate-300 border border-slate-600 hover:bg-slate-600/50 disabled:opacity-30 disabled:cursor-not-allowed"
        >
          🔄 Process One
        </button>

        <div className="flex items-center gap-2 text-xs text-slate-400">
          <label>Speed:</label>
          <input
            type="range"
            min={500}
            max={5000}
            step={500}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="w-24 accent-blue-500"
          />
          <span className="font-mono text-blue-400 w-12">
            {(speed / 1000).toFixed(1)}s
          </span>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="mb-6">
        <StatsPanel history={history} />
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
                <Card title="Step 1 — Raw Transaction" icon="📥">
                  <RawTransactionPanel tx={current.raw} />
                </Card>

                {/* Feature Engineering */}
                <Card
                  title="Step 2 — Feature Engineering"
                  icon="⚙️"
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
                    <p className="text-xs text-slate-500 italic">
                      Waiting for incoming transaction…
                    </p>
                  ) : (
                    <div className="flex items-center gap-2 text-xs text-blue-400">
                      <span className="animate-spin">⏳</span>
                      Processing features…
                    </div>
                  )}
                </Card>

                {/* Model Scoring */}
                <Card
                  title="Step 3 — LightGBM Scoring"
                  icon="🧠"
                  className={`md:col-span-2 ${
                    current.stage === "scoring" ? "animate-pulse-glow" : ""
                  }`}
                >
                  {current.prediction ? (
                    <div className="space-y-3">
                      <ProbabilityBar
                        probability={current.prediction.fraud_probability}
                      />
                      <div className="flex items-center justify-between">
                        <div className="text-xs text-slate-400">
                          <span className="font-mono">
                            model.predict_proba(features)
                          </span>{" "}
                          →{" "}
                          <span className="font-bold text-white">
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
                            ? "🚨 FRAUD DETECTED"
                            : "✅ LEGITIMATE"}
                        </Badge>
                      </div>

                      {/* Educational explanation */}
                      <div className="mt-2 p-3 rounded-lg bg-slate-800/80 border border-slate-700 text-[11px] text-slate-400 leading-relaxed">
                        <strong className="text-slate-300">
                          📚 How it works:
                        </strong>{" "}
                        The LightGBM model analyzes 39 engineered features
                        (numerical + one-hot encoded) and outputs a fraud
                        probability between 0 and 1. If probability ≥{" "}
                        <span className="text-yellow-400">0.50</span>{" "}
                        (threshold), the transaction is flagged as fraud.
                        {current.prediction.is_fraud === 1 && (
                          <span className="text-red-400 block mt-1">
                            ⚠️ Suspicious signals: high amount ($
                            {current.raw.amt.toFixed(2)}), distance{" "}
                            {current.prediction.engineered_features.distance_km}{" "}
                            km, hour{" "}
                            {current.prediction.engineered_features.hour}:00
                          </span>
                        )}
                      </div>
                    </div>
                  ) : current.stage === "scoring" ? (
                    <div className="flex items-center gap-2 text-xs text-blue-400">
                      <span className="animate-spin">⏳</span>
                      Running LightGBM inference…
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500 italic">
                      Waiting for features…
                    </p>
                  )}
                </Card>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-slate-500">
              <div className="text-4xl mb-3">🔍</div>
              <p className="text-sm">
                Press <strong>&quot;Start Auto-Stream&quot;</strong> or{" "}
                <strong>&quot;Process One&quot;</strong> to begin
              </p>
              <p className="text-xs text-slate-600 mt-1">
                Synthetic transactions will be generated and analyzed in real
                time
              </p>
            </div>
          )}
        </div>

        {/* ── HISTORY PANEL ── */}
        <Card title="Transaction History" icon="📋" className="max-h-[600px] overflow-hidden flex flex-col">
          <div className="overflow-y-auto flex-1 space-y-2 pr-1">
            {history.length === 0 ? (
              <p className="text-xs text-slate-500 italic text-center py-8">
                No transactions processed yet
              </p>
            ) : (
              history.map((tx) => (
                <div
                  key={tx.id}
                  className={`p-2.5 rounded-lg border text-xs animate-slide-in ${
                    tx.prediction?.is_fraud
                      ? "bg-red-500/5 border-red-500/30"
                      : "bg-green-500/5 border-green-500/30"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-slate-200 truncate max-w-[120px]">
                      {tx.raw.merchant}
                    </span>
                    <Badge
                      variant={tx.prediction?.is_fraud ? "fraud" : "legit"}
                    >
                      {tx.prediction?.is_fraud ? "FRAUD" : "OK"}
                    </Badge>
                  </div>
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span>
                      {tx.raw.category.replace("_", " ")} · ${tx.raw.amt.toFixed(2)}
                    </span>
                    <span className="font-mono">
                      {((tx.prediction?.fraud_probability ?? 0) * 100).toFixed(
                        1
                      )}
                      %
                    </span>
                  </div>
                  <div className="text-[10px] text-slate-600 mt-0.5">
                    {tx.raw.city}, {tx.raw.state} ·{" "}
                    {tx.prediction?.engineered_features.distance_km} km ·{" "}
                    {tx.prediction?.engineered_features.hour}:00
                  </div>
                </div>
              ))
            )}
            <div ref={historyEndRef} />
          </div>
        </Card>
      </div>

      {/* ── Architecture Diagram ── */}
      <div className="mt-8 mb-4">
        <Card title="System Architecture" icon="🏗️">
          <div className="flex flex-wrap items-center justify-center gap-3 text-xs py-4">
            {[
              { icon: "🎲", label: "Synthetic\nGenerator", sub: "Python / Real stats" },
              { icon: "→", label: "", sub: "" },
              { icon: "📥", label: "Raw\nTransaction", sub: "15 fields" },
              { icon: "→", label: "", sub: "" },
              { icon: "⚙️", label: "Feature\nEngineering", sub: "→ 39 features" },
              { icon: "→", label: "", sub: "" },
              { icon: "🧠", label: "LightGBM\nModel", sub: "predict_proba()" },
              { icon: "→", label: "", sub: "" },
              { icon: "📊", label: "Result\nVerdict", sub: "Fraud / Legit" },
            ].map((item, i) =>
              item.label === "" ? (
                <span key={i} className="text-slate-600 text-lg">
                  →
                </span>
              ) : (
                <div
                  key={i}
                  className="bg-slate-800 border border-slate-700 rounded-lg p-3 text-center min-w-[90px]"
                >
                  <div className="text-xl mb-1">{item.icon}</div>
                  <div className="text-slate-300 font-semibold whitespace-pre-line leading-tight">
                    {item.label}
                  </div>
                  <div className="text-[10px] text-slate-500 mt-1">
                    {item.sub}
                  </div>
                </div>
              )
            )}
          </div>
        </Card>
      </div>

      {/* Footer */}
      <footer className="text-center text-[10px] text-slate-600 py-4">
        Educational PoC — RSU AI &amp; ML Final Project · LightGBM Fraud Detection ·
        Dataset: Kaggle dhruvb2028/credit-card-fraud-dataset
      </footer>
    </main>
  );
}
