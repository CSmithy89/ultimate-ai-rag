/**
 * Home Page - Server Component
 *
 * This page is a React Server Component (RSC) that renders client components
 * as children. The ChatSidebar component is a client component (marked with
 * "use client") that handles CopilotKit integration and browser-only features.
 *
 * Server/Client Boundary:
 * - This file: Server Component - can access server-only features, no "use client"
 * - ChatSidebar: Client Component - handles interactive CopilotKit UI
 *
 * This pattern allows us to leverage server-side rendering for the static
 * content while delegating interactive features to client components.
 */
import Link from "next/link";
import { ChatSidebar } from "@/components/copilot/ChatSidebar";

export default function Home() {
  const workflowEnabled = process.env.NEXT_PUBLIC_VISUAL_WORKFLOW_ENABLED === "true";

  return (
    <main className="min-h-screen bg-slate-50">
      {/* Main content area */}
      <div className="container mx-auto py-8">
        <div className="space-y-3">
          <h1 className="text-3xl font-semibold text-slate-900">Ultimate AI RAG</h1>
          <p className="text-base text-slate-600">
            Agentic RAG and GraphRAG with CopilotKit experiences.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link
              href="/chat"
              className="bg-indigo-600 text-white text-sm px-4 py-2 rounded-md"
            >
              Open Chat
            </Link>
            <Link
              href="/ingest"
              className="bg-white text-slate-700 text-sm px-4 py-2 rounded-md border border-slate-200"
            >
              Ingest Content
            </Link>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 mt-8">
          <Link
            href="/chat"
            className="bg-white border border-slate-200 rounded-xl p-5 hover:border-slate-300"
          >
            <h2 className="text-lg font-semibold text-slate-900">AI Chat</h2>
            <p className="text-sm text-slate-600 mt-1">
              Query the knowledge base with sources, actions, and HITL validation.
            </p>
          </Link>
          <Link
            href="/ingest"
            className="bg-white border border-slate-200 rounded-xl p-5 hover:border-slate-300"
          >
            <h2 className="text-lg font-semibold text-slate-900">Ingestion</h2>
            <p className="text-sm text-slate-600 mt-1">
              Crawl URLs or upload PDFs to populate the graph.
            </p>
          </Link>
          <Link
            href="/knowledge"
            className="bg-white border border-slate-200 rounded-xl p-5 hover:border-slate-300"
          >
            <h2 className="text-lg font-semibold text-slate-900">Knowledge Graph</h2>
            <p className="text-sm text-slate-600 mt-1">
              Explore entities, relationships, and graph stats.
            </p>
          </Link>
          <Link
            href="/ops"
            className="bg-white border border-slate-200 rounded-xl p-5 hover:border-slate-300"
          >
            <h2 className="text-lg font-semibold text-slate-900">Ops Dashboard</h2>
            <p className="text-sm text-slate-600 mt-1">
              Monitor costs, alerts, and recent requests.
            </p>
          </Link>
          <Link
            href="/ops/trajectories"
            className="bg-white border border-slate-200 rounded-xl p-5 hover:border-slate-300"
          >
            <h2 className="text-lg font-semibold text-slate-900">Trajectories</h2>
            <p className="text-sm text-slate-600 mt-1">
              Inspect agent timelines and debugging events.
            </p>
          </Link>
          <Link
            href="/workflow"
            className="bg-white border border-slate-200 rounded-xl p-5 hover:border-slate-300"
          >
            <h2 className="text-lg font-semibold text-slate-900">Workflow Editor</h2>
            <p className="text-sm text-slate-600 mt-1">
              {workflowEnabled
                ? "Design pipeline steps and test execution paths."
                : "Enable NEXT_PUBLIC_VISUAL_WORKFLOW_ENABLED to use this."}
            </p>
          </Link>
        </div>
      </div>

      {/* Chat Sidebar - Client component for CopilotKit interactivity */}
      <ChatSidebar />
    </main>
  );
}
