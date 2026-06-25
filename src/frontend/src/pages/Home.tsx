import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";

// ── Types ──────────────────────────────────────────────────────────

interface HealthResponse {
  status: string;
  vector_db: { healthy: boolean; error: string | null };
  reranker: {
    status: string;
    model: string;
    device: string;
    enabled: boolean;
    loaded: boolean;
    latency_ms?: number;
    warm?: boolean;
    error?: string;
  };
  llm_client: { healthy: boolean; error: string | null };
}

interface ServiceMeta {
  label: string;
  description: string;
  icon: React.ReactNode;
}

// ── Constants ──────────────────────────────────────────────────────

type ServiceKey = "vector_db" | "reranker" | "llm_client";

const SERVICE_META: Record<ServiceKey, ServiceMeta> = {
  vector_db: {
    label: "Vector DB",
    description: "Chroma DB",
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <ellipse cx="12" cy="5" rx="9" ry="3" />
        <path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5" />
        <path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3" />
      </svg>
    ),
  },
  reranker: {
    label: "Reranker",
    description: "Cross-encoder reranker",
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2a4 4 0 0 1 4 4c0 1.95-1.4 3.58-3.25 3.93" />
        <path d="M8.75 9.93A4.002 4.002 0 0 1 12 2" />
        <path d="M12 9v13" />
        <path d="M5.2 18.8A6.98 6.98 0 0 1 5 17c0-2.83 1.68-5.27 4.1-6.38" />
        <path d="M14.9 10.62A7.002 7.002 0 0 1 19 17c0 .62-.07 1.22-.2 1.8" />
        <path d="M7 22h10" />
      </svg>
    ),
  },
  llm_client: {
    label: "LLM",
    description: "Generation model",
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 2a7 7 0 1 0 10 10" />
        <path d="M12 12 2 22l5-5 5 5 5-5 5 5" />
        <circle cx="12" cy="7" r="1" />
      </svg>
    ),
  },
};

// ── Helpers ────────────────────────────────────────────────────────

function statusColor(healthy: boolean | null): string {
  if (healthy === null) return "bg-muted text-muted-foreground";
  if (healthy === true) return "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400";
  return "bg-red-500/10 text-red-600 dark:text-red-400";
}

function statusDot(healthy: boolean | null): string {
  if (healthy === null) return "bg-muted-foreground/30";
  if (healthy === true) return "bg-emerald-500";
  return "bg-red-500";
}

function rerankerStatusColor(status: string): string {
  if (status === "healthy" || status === "disabled") {
    return "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400";
  }
  return "bg-red-500/10 text-red-600 dark:text-red-400";
}

function rerankerStatusDot(status: string): string {
  if (status === "healthy" || status === "disabled") return "bg-emerald-500";
  return "bg-red-500";
}

// ── Component ──────────────────────────────────────────────────────

export default function Home() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [checked, setChecked] = useState(false);

  const checkHealth = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("http://localhost:8000/health");
      if (!res.ok) {
        throw new Error(`Health check failed with status ${res.status}`);
      }
      const data: HealthResponse = await res.json();
      setHealth(data);
      setChecked(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred");
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-svh justify-center p-6 pt-20">
      <div className="flex w-full max-w-xl flex-col items-center gap-8">
        {/* Hero */}
        <div className="flex flex-col items-center gap-2 text-center">
          <div className="mb-2 flex size-16 items-center justify-center rounded-2xl bg-primary/10">
            <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              <path d="M8 10h.01" />
              <path d="M12 10h.01" />
              <path d="M16 10h.01" />
            </svg>
          </div>
          <h1 className="text-3xl font-semibold tracking-tight">RAG-Chatbot</h1>
          <p className="text-muted-foreground max-w-sm text-sm">
            Retrieval-Augmented Generation chatbot powered by ChromaDB, Llama,
            and your uploaded documents.
          </p>
        </div>

        {/* Health check button */}
        <Button
          id="health-check-btn"
          onClick={checkHealth}
          disabled={loading}
          size="lg"
          className="w-full max-w-xs"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="size-4 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Checking…
            </span>
          ) : checked ? (
            <span className="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2" />
              </svg>
              Re-check Health
            </span>
          ) : (
            <span className="flex items-center gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
              </svg>
              Check System Health
            </span>
          )}
        </Button>

        {/* Error */}
        {error && (
          <div
            id="health-error"
            className="flex w-full items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive animate-in fade-in slide-in-from-top-2 duration-200"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mt-0.5 shrink-0">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <p>{error}</p>
          </div>
        )}

        {/* Service status cards */}
        {health && (
          <div className="grid w-full gap-3 animate-in fade-in slide-in-from-bottom-3 duration-300">
            {/* Vector DB card */}
            <Card size="sm" className="transition-shadow hover:shadow-md">
              <CardHeader className="flex-row items-center gap-4">
                <div className={`flex size-10 shrink-0 items-center justify-center rounded-lg ${statusColor(health.vector_db.healthy)}`}>
                  {SERVICE_META.vector_db.icon}
                </div>
                <div className="min-w-0 flex-1">
                  <CardTitle>{SERVICE_META.vector_db.label}</CardTitle>
                  <p className="text-muted-foreground text-xs">{SERVICE_META.vector_db.description}</p>
                  {health.vector_db.error && (
                    <p className="text-destructive mt-0.5 text-xs">{health.vector_db.error}</p>
                  )}
                </div>
                <div className="flex items-center gap-2 rounded-full border border-border px-3 py-1">
                  <span className={`size-2 rounded-full ${statusDot(health.vector_db.healthy)}`} />
                  <span className="text-xs font-medium capitalize">
                    {health.vector_db.healthy ? "healthy" : "unhealthy"}
                  </span>
                </div>
              </CardHeader>
            </Card>

            {/* Reranker card */}
            <Card size="sm" className="transition-shadow hover:shadow-md">
              <CardHeader className="flex-row items-center gap-4">
                <div className={`flex size-10 shrink-0 items-center justify-center rounded-lg ${rerankerStatusColor(health.reranker.status)}`}>
                  {SERVICE_META.reranker.icon}
                </div>
                <div className="min-w-0 flex-1">
                  <CardTitle>{SERVICE_META.reranker.label}</CardTitle>
                  <p className="text-muted-foreground text-xs">{SERVICE_META.reranker.description}</p>
                  <p className="text-muted-foreground mt-0.5 text-xs">
                    {health.reranker.model} • {health.reranker.device}
                  </p>
                  {health.reranker.error && (
                    <p className="text-destructive mt-0.5 text-xs">{health.reranker.error}</p>
                  )}
                </div>
                <div className="flex items-center gap-2 rounded-full border border-border px-3 py-1">
                  <span className={`size-2 rounded-full ${rerankerStatusDot(health.reranker.status)}`} />
                  <span className="text-xs font-medium capitalize">{health.reranker.status}</span>
                </div>
              </CardHeader>
            </Card>

            {/* LLM client card */}
            <Card size="sm" className="transition-shadow hover:shadow-md">
              <CardHeader className="flex-row items-center gap-4">
                <div className={`flex size-10 shrink-0 items-center justify-center rounded-lg ${statusColor(health.llm_client.healthy)}`}>
                  {SERVICE_META.llm_client.icon}
                </div>
                <div className="min-w-0 flex-1">
                  <CardTitle>{SERVICE_META.llm_client.label}</CardTitle>
                  <p className="text-muted-foreground text-xs">{SERVICE_META.llm_client.description}</p>
                  {health.llm_client.error && (
                    <p className="text-destructive mt-0.5 text-xs">{health.llm_client.error}</p>
                  )}
                </div>
                <div className="flex items-center gap-2 rounded-full border border-border px-3 py-1">
                  <span className={`size-2 rounded-full ${statusDot(health.llm_client.healthy)}`} />
                  <span className="text-xs font-medium capitalize">
                    {health.llm_client.healthy ? "healthy" : "unhealthy"}
                  </span>
                </div>
              </CardHeader>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}