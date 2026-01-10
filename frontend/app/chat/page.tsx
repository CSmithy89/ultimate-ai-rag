/**
 * Dedicated Chat Page Example.
 *
 * Story 21-F2: Implement CopilotChat Embedded Component
 * AC5: Dedicated chat page example
 *
 * This page demonstrates using EmbeddedChat in a full-page layout.
 * Access via: /chat
 */

import { EmbeddedChat } from "@/components/copilot/EmbeddedChat";

export const metadata = {
  title: "AI Chat | Agentic RAG",
  description: "Chat with the AI assistant about your documents",
};

export default function ChatPage() {
  return (
    <div className="container mx-auto h-screen py-4 px-4 flex flex-col">
      <header className="flex-none mb-4">
        <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
          AI Assistant
        </h1>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Ask questions about your documents and knowledge base
        </p>
      </header>

      <main className="flex-1 min-h-0">
        <EmbeddedChat
          className="h-full border border-slate-200 dark:border-slate-700 rounded-lg shadow-sm"
          welcomeMessage="Welcome! I can help you explore your documents, answer questions, and provide insights from your knowledge base. What would you like to know?"
          title="RAG Assistant"
        />
      </main>
    </div>
  );
}
