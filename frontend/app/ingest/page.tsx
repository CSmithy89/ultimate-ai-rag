"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";
const tenantId = process.env.NEXT_PUBLIC_TENANT_ID ?? "";

type IngestJob = {
  job_id?: string;
  id?: string;
  job_type?: string;
  status?: string;
  created_at?: string;
  error_message?: string | null;
};

type JobListResponse = {
  data?: { jobs?: IngestJob[] };
};

type JobCreateResponse = {
  data?: {
    job_id?: string;
    status?: string;
    filename?: string;
  };
};

function IngestPageContent() {
  const searchParams = useSearchParams();
  const urlInputRef = useRef<HTMLInputElement | null>(null);
  const pdfInputRef = useRef<HTMLInputElement | null>(null);
  const [url, setUrl] = useState("");
  const [maxDepth, setMaxDepth] = useState(3);
  const [file, setFile] = useState<File | null>(null);
  const [jobs, setJobs] = useState<IngestJob[]>([]);
  const [urlStatus, setUrlStatus] = useState<string | null>(null);
  const [pdfStatus, setPdfStatus] = useState<string | null>(null);
  const [jobsStatus, setJobsStatus] = useState<string | null>(null);
  const [isSubmittingUrl, setIsSubmittingUrl] = useState(false);
  const [isSubmittingPdf, setIsSubmittingPdf] = useState(false);

  const canSubmit = Boolean(tenantId);

  const jobRows = useMemo(() => {
    return jobs.map((job) => {
      const id = job.job_id ?? job.id ?? "unknown";
      return { ...job, id };
    });
  }, [jobs]);

  const loadJobs = useCallback(async () => {
    if (!tenantId) {
      setJobs([]);
      setJobsStatus("Set NEXT_PUBLIC_TENANT_ID to load ingestion jobs.");
      return;
    }

    setJobsStatus(null);
    try {
      const response = await fetch(
        `${API_BASE_URL}/ingest/jobs?tenant_id=${tenantId}&limit=10`
      );
      const payload = (await response.json()) as JobListResponse;
      if (!response.ok) {
        throw new Error(
          (payload as { detail?: string }).detail || "Failed to load jobs."
        );
      }
      setJobs(payload.data?.jobs ?? []);
    } catch (error) {
      setJobsStatus(error instanceof Error ? error.message : "Failed to load jobs.");
    }
  }, []);

  useEffect(() => {
    void loadJobs();
  }, [loadJobs]);

  useEffect(() => {
    jobs.forEach((job) => {
      if (!job.error_message) {
        return;
      }
      const jobId = job.job_id ?? job.id ?? "unknown";
      console.error("Ingestion job failed", {
        jobId,
        jobType: job.job_type ?? "unknown",
        status: job.status ?? "unknown",
        error: job.error_message,
      });
    });
  }, [jobs]);

  useEffect(() => {
    const focus = searchParams.get("focus");
    if (focus === "url") {
      urlInputRef.current?.focus();
      urlInputRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    if (focus === "pdf") {
      pdfInputRef.current?.focus();
      pdfInputRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [searchParams]);

  useEffect(() => {
    if (!tenantId) {
      return undefined;
    }

    const interval = setInterval(() => {
      void loadJobs();
    }, 15000);

    return () => clearInterval(interval);
  }, [loadJobs]);

  const submitUrl = async () => {
    if (!canSubmit || !url) {
      setUrlStatus("Provide a URL and tenant ID before submitting.");
      return;
    }

    setIsSubmittingUrl(true);
    setUrlStatus(null);
    try {
      const response = await fetch(`${API_BASE_URL}/ingest/url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url,
          tenant_id: tenantId,
          max_depth: maxDepth,
          options: {},
        }),
      });
      const payload = (await response.json()) as JobCreateResponse;
      if (!response.ok) {
        throw new Error(
          (payload as { detail?: string }).detail || "Failed to start crawl."
        );
      }
      setUrlStatus(
        `Queued crawl job ${payload.data?.job_id ?? "unknown"} (${payload.data?.status ?? "queued"}).`
      );
      setUrl("");
      void loadJobs();
    } catch (error) {
      setUrlStatus(error instanceof Error ? error.message : "Failed to start crawl.");
    } finally {
      setIsSubmittingUrl(false);
    }
  };

  const submitPdf = async () => {
    if (!canSubmit || !file) {
      setPdfStatus("Choose a PDF and tenant ID before submitting.");
      return;
    }

    setIsSubmittingPdf(true);
    setPdfStatus(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("tenant_id", tenantId);

      const response = await fetch(`${API_BASE_URL}/ingest/document`, {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json()) as JobCreateResponse;
      if (!response.ok) {
        throw new Error(
          (payload as { detail?: string }).detail || "Failed to upload PDF."
        );
      }
      setPdfStatus(
        `Queued parse job ${payload.data?.job_id ?? "unknown"} (${payload.data?.status ?? "queued"}).`
      );
      setFile(null);
      void loadJobs();
    } catch (error) {
      setPdfStatus(error instanceof Error ? error.message : "Failed to upload PDF.");
    } finally {
      setIsSubmittingPdf(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-50">
      <div className="container mx-auto py-10 space-y-8">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold text-slate-900">
            Ingest Content
          </h1>
          <p className="text-slate-600">
            Start ingestion jobs to populate the knowledge base and graph.
          </p>
          {!tenantId ? (
            <p className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
              Set NEXT_PUBLIC_TENANT_ID to enable ingestion.
            </p>
          ) : null}
        </header>

        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
            <h2 className="text-lg font-semibold text-slate-800">
              Ingest a URL
            </h2>
            <p className="text-sm text-slate-500">
              Crawl a documentation site and ingest discovered pages.
            </p>
            <div className="space-y-3">
              <input
                ref={urlInputRef}
                className="w-full border border-slate-200 rounded-md px-3 py-2 text-sm"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="https://docs.example.com"
              />
              <div className="flex items-center gap-3 text-sm">
                <label className="text-slate-600">Max depth</label>
                <input
                  className="w-24 border border-slate-200 rounded-md px-3 py-1"
                  type="number"
                  min={1}
                  max={10}
                  value={maxDepth}
                  onChange={(event) => setMaxDepth(Number(event.target.value))}
                />
              </div>
              <button
                type="button"
                onClick={submitUrl}
                disabled={!canSubmit || isSubmittingUrl}
                className="bg-indigo-600 text-white text-sm px-4 py-2 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSubmittingUrl ? "Starting..." : "Start crawl"}
              </button>
              {urlStatus ? (
                <p className="text-sm text-slate-600">{urlStatus}</p>
              ) : null}
            </div>
          </div>

          <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
            <h2 className="text-lg font-semibold text-slate-800">
              Upload a PDF
            </h2>
            <p className="text-sm text-slate-500">
              Parse a PDF with Docling and ingest the extracted content.
            </p>
            <div className="space-y-3">
              <input
                ref={pdfInputRef}
                type="file"
                accept="application/pdf"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                className="block w-full text-sm text-slate-600 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:bg-slate-100 file:text-slate-700"
              />
              <button
                type="button"
                onClick={submitPdf}
                disabled={!canSubmit || isSubmittingPdf}
                className="bg-emerald-600 text-white text-sm px-4 py-2 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSubmittingPdf ? "Uploading..." : "Upload PDF"}
              </button>
              {pdfStatus ? (
                <p className="text-sm text-slate-600">{pdfStatus}</p>
              ) : null}
            </div>
          </div>
        </section>

        <section className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-slate-800">
              Recent Ingestion Jobs
            </h2>
            <button
              type="button"
              onClick={loadJobs}
              className="text-sm text-slate-600 hover:text-slate-900"
            >
              Refresh
            </button>
          </div>
          {jobsStatus ? (
            <p className="text-sm text-slate-500">{jobsStatus}</p>
          ) : jobRows.length ? (
            <div className="space-y-2">
              {jobRows.map((job) => (
                <div
                  key={job.id}
                  className="border border-slate-100 rounded-lg px-3 py-2 text-sm"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-slate-700">
                      {job.job_type ?? "job"}
                    </span>
                    <span className="text-slate-500">{job.status ?? "unknown"}</span>
                  </div>
                  <div className="text-xs text-slate-500">
                    {job.id} · {job.created_at ? new Date(job.created_at).toLocaleString() : "n/a"}
                  </div>
                  {job.error_message ? (
                    <p className="text-xs text-red-600 mt-1">
                      Job failed. Check the logs for details.
                    </p>
                  ) : null}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-500">No jobs yet.</p>
          )}

          <div className="text-sm text-slate-600">
            Ready to explore results?{" "}
            <Link href="/knowledge" className="text-indigo-600 hover:text-indigo-700">
              Open the knowledge graph
            </Link>
            .
          </div>
        </section>
      </div>
    </main>
  );
}

function IngestPageFallback() {
  return (
    <main className="min-h-screen bg-slate-50">
      <div className="container mx-auto py-10 space-y-8">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold text-slate-900">
            Ingest Content
          </h1>
          <p className="text-slate-600">Loading...</p>
        </header>
      </div>
    </main>
  );
}

export default function IngestPage() {
  return (
    <Suspense fallback={<IngestPageFallback />}>
      <IngestPageContent />
    </Suspense>
  );
}
