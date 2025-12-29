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
import { ChatSidebar } from "@/components/copilot/ChatSidebar";

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-50">
      {/* Main content area */}
      <div className="container mx-auto py-8">
        <h1 className="text-3xl font-semibold text-slate-900">Ultimate AI RAG</h1>
        <p className="text-base text-slate-600 mt-2">
          Next.js App Router foundation with CopilotKit AI chat.
        </p>
      </div>

      {/* Chat Sidebar - Client component for CopilotKit interactivity */}
      <ChatSidebar />
    </main>
  );
}
