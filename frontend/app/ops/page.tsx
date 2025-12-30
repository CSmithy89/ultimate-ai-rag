"use client";

import { useMemo, useState, useEffect } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  useCostSummary,
  useCostEvents,
  useCostAlerts,
  useUpdateCostAlerts,
} from "@/hooks/use-ops-dashboard";

const DEMO_TENANT_ID = "00000000-0000-0000-0000-000000000001";
function formatCurrency(value: number) {
  return `$${value.toFixed(4)}`;
}

function TrendBars({ values }: { values: number[] }) {
  const max = Math.max(...values, 0);
  return (
    <div className="flex items-end gap-2 h-24">
      {values.map((value, idx) => {
        const height = max === 0 ? 4 : Math.max(4, (value / max) * 96);
        return (
          <div
            key={idx}
            className="w-3 rounded bg-indigo-500/80"
            style={{ height }}
            title={formatCurrency(value)}
          />
        );
      })}
    </div>
  );
}

function OpsDashboard() {
  const [window, setWindow] = useState("day");
  const summaryQuery = useCostSummary(DEMO_TENANT_ID, window);
  const eventsQuery = useCostEvents(DEMO_TENANT_ID);
  const alertsQuery = useCostAlerts(DEMO_TENANT_ID);
  const updateAlerts = useUpdateCostAlerts();

  const trendValues = useMemo(() => {
    return summaryQuery.data?.trend.map((point) => point.total_cost_usd) ?? [];
  }, [summaryQuery.data]);

  const [dailyThreshold, setDailyThreshold] = useState("");
  const [monthlyThreshold, setMonthlyThreshold] = useState("");
  const [alertsEnabled, setAlertsEnabled] = useState(true);

  useEffect(() => {
    if (alertsQuery.data) {
      setDailyThreshold(
        alertsQuery.data.daily_threshold_usd?.toString() ?? ""
      );
      setMonthlyThreshold(
        alertsQuery.data.monthly_threshold_usd?.toString() ?? ""
      );
      setAlertsEnabled(alertsQuery.data.enabled ?? true);
    }
  }, [alertsQuery.data]);

  return (
    <main className="min-h-screen bg-slate-50">
      <div className="container mx-auto py-10 space-y-8">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold text-slate-900">
            Operations &amp; Observability
          </h1>
          <p className="text-slate-600">
            Monitor LLM usage, costs, and operational thresholds.
          </p>
        </header>

        <section className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-slate-800">
              Cost Summary
            </h2>
            <select
              value={window}
              onChange={(event) => setWindow(event.target.value)}
              className="border border-slate-200 rounded-md px-3 py-1 text-sm text-slate-700"
            >
              <option value="day">Last 24h</option>
              <option value="week">Last 7d</option>
              <option value="month">Last 30d</option>
            </select>
          </div>

          {summaryQuery.isLoading ? (
            <p className="text-slate-500">Loading cost summary...</p>
          ) : summaryQuery.error ? (
            <p className="text-red-600">
              Failed to load summary: {summaryQuery.error.message}
            </p>
          ) : summaryQuery.data ? (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="border border-slate-100 rounded-lg p-4">
                  <p className="text-xs uppercase text-slate-500">Total Cost</p>
                  <p className="text-2xl font-semibold text-slate-900">
                    {formatCurrency(summaryQuery.data.total_cost_usd)}
                  </p>
                </div>
                <div className="border border-slate-100 rounded-lg p-4">
                  <p className="text-xs uppercase text-slate-500">
                    Total Savings
                  </p>
                  <p className="text-2xl font-semibold text-emerald-600">
                    {formatCurrency(summaryQuery.data.total_savings_usd)}
                  </p>
                </div>
                <div className="border border-slate-100 rounded-lg p-4">
                  <p className="text-xs uppercase text-slate-500">
                    Total Tokens
                  </p>
                  <p className="text-2xl font-semibold text-slate-900">
                    {summaryQuery.data.total_tokens.toLocaleString()}
                  </p>
                </div>
                <div className="border border-slate-100 rounded-lg p-4">
                  <p className="text-xs uppercase text-slate-500">
                    Total Requests
                  </p>
                  <p className="text-2xl font-semibold text-slate-900">
                    {summaryQuery.data.total_requests}
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="border border-slate-100 rounded-lg p-4 space-y-3">
                  <h3 className="text-sm font-semibold text-slate-700">
                    Cost Trend
                  </h3>
                  {trendValues.length ? (
                    <TrendBars values={trendValues} />
                  ) : (
                    <p className="text-sm text-slate-500">No data yet.</p>
                  )}
                </div>
                <div className="border border-slate-100 rounded-lg p-4 space-y-3">
                  <h3 className="text-sm font-semibold text-slate-700">
                    Cost by Model
                  </h3>
                  <div className="space-y-2 text-sm">
                    {summaryQuery.data.by_model.length ? (
                      summaryQuery.data.by_model.map((model) => (
                        <div
                          key={model.model_id}
                          className="flex items-center justify-between"
                        >
                          <span className="text-slate-600">
                            {model.model_id}
                          </span>
                          <span className="font-medium text-slate-800">
                            {formatCurrency(model.total_cost_usd)}
                          </span>
                        </div>
                      ))
                    ) : (
                      <p className="text-slate-500">No model data yet.</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : null}
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
            <h2 className="text-lg font-semibold text-slate-800">
              Alerts &amp; Thresholds
            </h2>
            {summaryQuery.data?.alerts?.enabled ? (
              <div className="text-sm text-slate-600 space-y-1">
                <p>
                  Daily total:{" "}
                  <span className="font-medium text-slate-800">
                    {formatCurrency(summaryQuery.data.alerts.daily_total_usd ?? 0)}
                  </span>
                </p>
                <p>
                  Monthly total:{" "}
                  <span className="font-medium text-slate-800">
                    {formatCurrency(summaryQuery.data.alerts.monthly_total_usd ?? 0)}
                  </span>
                </p>
              </div>
            ) : (
              <p className="text-sm text-slate-500">
                Alerts are currently disabled.
              </p>
            )}

            <form
              className="space-y-3"
              onSubmit={(event) => {
                event.preventDefault();
                const dailyValue =
                  dailyThreshold === "" ? null : Number(dailyThreshold);
                const monthlyValue =
                  monthlyThreshold === "" ? null : Number(monthlyThreshold);
                updateAlerts.mutate({
                  tenant_id: DEMO_TENANT_ID,
                  daily_threshold_usd:
                    dailyValue !== null && Number.isFinite(dailyValue)
                      ? dailyValue
                      : null,
                  monthly_threshold_usd:
                    monthlyValue !== null && Number.isFinite(monthlyValue)
                      ? monthlyValue
                      : null,
                  enabled: alertsEnabled,
                });
              }}
            >
              <div className="flex items-center gap-3 text-sm">
                <input
                  id="alerts-enabled"
                  type="checkbox"
                  checked={alertsEnabled}
                  onChange={(event) => setAlertsEnabled(event.target.checked)}
                />
                <label htmlFor="alerts-enabled" className="text-slate-700">
                  Enable alerts
                </label>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-slate-500">Daily threshold</label>
                  <input
                    className="mt-1 w-full border border-slate-200 rounded-md px-3 py-2 text-sm"
                    value={dailyThreshold}
                    onChange={(event) => setDailyThreshold(event.target.value)}
                    placeholder="e.g. 25.00"
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-500">
                    Monthly threshold
                  </label>
                  <input
                    className="mt-1 w-full border border-slate-200 rounded-md px-3 py-2 text-sm"
                    value={monthlyThreshold}
                    onChange={(event) => setMonthlyThreshold(event.target.value)}
                    placeholder="e.g. 500.00"
                  />
                </div>
              </div>
              <button
                type="submit"
                className="bg-indigo-600 text-white text-sm px-4 py-2 rounded-md"
                disabled={updateAlerts.isPending}
              >
                {updateAlerts.isPending ? "Saving..." : "Save thresholds"}
              </button>
            </form>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
            <h2 className="text-lg font-semibold text-slate-800">
              Recent Requests
            </h2>
            {eventsQuery.isLoading ? (
              <p className="text-slate-500">Loading events...</p>
            ) : eventsQuery.error ? (
              <p className="text-red-600">
                Failed to load events: {eventsQuery.error.message}
              </p>
            ) : eventsQuery.data?.length ? (
              <div className="space-y-3">
                {eventsQuery.data.slice(0, 6).map((event) => (
                  <div
                    key={event.id}
                    className="border border-slate-100 rounded-lg px-3 py-2 text-sm"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-slate-600">{event.model_id}</span>
                      <span className="text-slate-800 font-medium">
                        {formatCurrency(event.total_cost_usd)}
                      </span>
                    </div>
                    <div className="text-xs text-slate-500">
                      {event.total_tokens.toLocaleString()} tokens ·{" "}
                      {new Date(event.created_at).toLocaleTimeString()}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-slate-500">No usage events yet.</p>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

export default function OpsPage() {
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
      <OpsDashboard />
    </QueryClientProvider>
  );
}
