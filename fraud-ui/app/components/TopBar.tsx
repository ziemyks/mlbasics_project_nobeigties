"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Live Monitor" },
  { href: "/pipeline", label: "AI Pipeline" },
];

export default function TopBar({ apiOk }: { apiOk: boolean | null }) {
  const pathname = usePathname();

  return (
    <header className="bg-black text-white">
      <div className="max-w-7xl mx-auto px-4 md:px-6 h-14 flex items-center justify-between gap-4">
        <div className="flex items-center gap-6 md:gap-10 min-w-0">
          <div className="flex items-center gap-3 text-sm font-medium uppercase tracking-[0.12em] shrink-0">
            <span className="w-3 h-3 bg-brand" />
            <span className="hidden sm:inline">Fraud Monitor</span>
          </div>
          <nav className="flex items-center h-14">
            {NAV.map((item) => {
              const active = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`h-14 flex items-center px-3 md:px-4 text-[11px] uppercase tracking-[0.1em] border-b-2 transition-colors ${
                    active
                      ? "border-brand text-white"
                      : "border-transparent text-white/50 hover:text-white"
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.08em] min-w-0">
          <span
            className={`w-2 h-2 rounded-full shrink-0 ${
              apiOk === true
                ? "bg-brand"
                : apiOk === false
                ? "bg-fraud"
                : "bg-white/40 animate-pulse"
            }`}
          />
          <span className="text-white/70 truncate hidden md:inline">
            {apiOk === true
              ? "API Connected (localhost:5050)"
              : apiOk === false
              ? "API Offline — run: python api.py"
              : "Checking API…"}
          </span>
        </div>
      </div>
    </header>
  );
}
