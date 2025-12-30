"use client";

import { useMemo, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  useTrajectories,
  useTrajectoryDetail,
} from "@/hooks/use-ops-dashboard";

const DEMO_TENANT_ID = "00000000-0000-0000-0000-000000000001";

function formatTimestamp(value: string | null | undefined) {
  if (!value) return "n/a";
  return new Date(value).toLocaleString();
}

function TrajectoryViewer() {
  const [statusFilter, setStatusFilter] = useState("all");
  const [agentFilter, setAgentFilter] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const trajectoriesQuery = useTrajectories(DEMO_TENANT_ID, {
    status: statusFilter === "all" ? undefined : statusFilter,
    agentType: agentFilter || undefined,
    limit: 50,
  });

  const detailQuery = useTrajectoryDetail(DEMO_TENANT_ID, selectedId);

  const selectedTrajectory = useMemo(() => {
    return trajectoriesQuery.data?.find((item) => item.id === selectedId);
  }, [trajectoriesQuery.data, selectedId]);

  return (
    <main className="min-h-screen bg-slate-50">
      <div className="container mx-auto py-10 space-y-6">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold text-slate-900">
            Trajectory Debugging
          </h1>
          <p className="text-slate-600">
            Inspect past agent sessions, filter by status, and review event
            timelines.
          </p>
        </header>

        <section className="bg-white border border-slate-200 rounded-xl p-4 space-y-4">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2 text-sm">
              <label className="text-slate-500">Status</label>
              <select
                value={statusFilter}
                onChange={(event) => setStatusFilter(event.target.value)}
                className="border border-slate-200 rounded-md px-2 py-1"
              >
                <option value="all">All</option>
                <option value="ok">OK</option>
                <option value="error">Error</option>
              </select>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <label className="text-slate-500">Agent</label>
              <input
                value={agentFilter}
                onChange={(event) => setAgentFilter(event.target.value)}
                className="border border-slate-200 rounded-md px-2 py-1"
                placeholder="orchestrator"
              />
            </div>
          </div>
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-3">
            <h2 className="text-sm font-semibold text-slate-700">
              Trajectories
            </h2>
            {trajectoriesQuery.isLoading ? (
              <p className="text-sm text-slate-500">Loading trajectories...</p>
            ) : trajectoriesQuery.error ? (
              <p className="text-sm text-red-600">
                Failed to load: {trajectoriesQuery.error.message}
              </p>
            ) : trajectoriesQuery.data?.length ? (
              <div className="space-y-2">
                {trajectoriesQuery.data.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setSelectedId(item.id)}
                    className={`w-full text-left border rounded-lg px-3 py-2 text-sm ${
                      item.id === selectedId
                        ? "border-indigo-500 bg-indigo-50"
                        : "border-slate-100 hover:border-slate-200"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium text-slate-700">
                        {item.agent_type ?? "unknown"}
                      </span>
                      <span
                        className={
                          item.has_error ? "text-red-600" : "text-emerald-600"
                        }
                      >
                        {item.has_error ? "error" : "ok"}
                      </span>
                    </div>
                    <div className="text-xs text-slate-500">
                      {formatTimestamp(item.created_at)} · {item.event_count}{" "}
                      events
                    </div>
                  </button>
                ))}
              </div>
            ) : (
              <p className="text-sm text-slate-500">No trajectories found.</p>
            )}
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-4 lg:col-span-2">
            <h2 className="text-sm font-semibold text-slate-700">
              Timeline
            </h2>
            {!selectedId ? (
              <p className="text-sm text-slate-500">
                Select a trajectory to view events.
              </p>
            ) : detailQuery.isLoading ? (
              <p className="text-sm text-slate-500">Loading timeline...</p>
            ) : detailQuery.error ? (
              <p className="text-sm text-red-600">
                Failed to load: {detailQuery.error.message}
              </p>
            ) : detailQuery.data ? (
              <div className="space-y-4">
                <div className="text-xs text-slate-500">
                  Session: {detailQuery.data.trajectory.session_id ?? "n/a"} ·{" "}
                  Started: {formatTimestamp(detailQuery.data.trajectory.created_at)}{" "}
                  · Duration:{" "}
                  {detailQuery.data.duration_ms
                    ? `${detailQuery.data.duration_ms} ms`
                    : "n/a"}
                </div>
                <div className="space-y-2">
                  {detailQuery.data.events.map((event) => (
                    <div
                      key={event.id}
                      className="border border-slate-100 rounded-lg p-3 text-sm"
                    >
                      <div className="flex items-center justify-between text-xs text-slate-500">
                        <span className="uppercase">{event.event_type}</span>
                        <span>{formatTimestamp(event.created_at)}</span>
                      </div>
                      <p className="mt-2 text-slate-700 whitespace-pre-wrap">
                        {event.content}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </section>

        {selectedTrajectory?.has_error && (
          <section className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 text-sm">
            This trajectory contains error events. Filter by status to compare
            with successful runs.
          </section>
        )}
      </div>
    </main>
  );
}

export default function TrajectoriesPage() {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 30000,
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <TrajectoryViewer />
    </QueryClientProvider>
  );
}
