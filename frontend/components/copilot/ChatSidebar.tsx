"use client";

import { CopilotSidebar } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { ThoughtTraceStepper } from "./ThoughtTraceStepper";
import { CopilotErrorBoundary } from "./CopilotErrorBoundary";
import { GenerativeUIRenderer } from "./GenerativeUIRenderer";
import { ActivityProgress } from "./ActivityProgress";
import { useChatSuggestions } from "@/hooks/use-chat-suggestions";
import { usePathname } from "next/navigation";
import { QuickActions } from "./QuickActions";
import type { QuickActionConfig } from "@/types/copilot";

/**
 * ChatSidebar component wrapping CopilotKit's CopilotSidebar
 * with custom styling following the project's design system.
 *
 * Story 6-2: Chat Sidebar Interface
 * Story 6-3: Generative UI Components
 * Story 21-A5: Static suggestions to avoid AG-UI protocol errors
 *
 * Design System:
 * - Primary (Indigo-600): #4F46E5
 * - Secondary (Emerald-500): #10B981
 * - Neutral: Slate colors
 */
export function ChatSidebar() {
  const pathname = usePathname() ?? "/";
  const isHome = pathname === "/";
  // Get static page-context suggestions (avoids AG-UI protocol errors)
  const pageSuggestions = useChatSuggestions();
  const suggestions = isHome ? [] : pageSuggestions;
  const homeActions: QuickActionConfig[] = [
    {
      label: "Search KB",
      message:
        "Search the knowledge base for an overview of our RAG architecture. Use retrieval and include sources. If no data is available, say so and direct me to /ingest.",
      action: "send",
      icon: "Search",
      description: "Run a knowledge base search with citations",
    },
    {
      label: "Ingest URL",
      action: "navigate",
      href: "/ingest?focus=url",
      icon: "FileText",
      description: "Start a crawl ingestion job",
    },
    {
      label: "Upload PDF",
      action: "navigate",
      href: "/ingest?focus=pdf",
      icon: "FileText",
      description: "Upload a PDF for ingestion",
    },
    {
      label: "Explore Graph",
      action: "navigate",
      href: "/knowledge",
      icon: "Search",
      description: "View the knowledge graph",
    },
  ];

  return (
    <CopilotErrorBoundary>
      <CopilotSidebar
        defaultOpen={true}
        labels={{
          title: "AI Copilot",
          initial: "How can I help you today?",
        }}
        className="copilot-sidebar"
        suggestions={suggestions}
      >
        {isHome ? (
          <QuickActions
            actions={homeActions}
            orientation="vertical"
            size="sm"
            className="px-4 pt-2"
          />
        ) : null}
        <ActivityProgress />
        <ThoughtTraceStepper />
        <GenerativeUIRenderer />
      </CopilotSidebar>
    </CopilotErrorBoundary>
  );
}
