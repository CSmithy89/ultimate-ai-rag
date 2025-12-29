import "./globals.css";
import { CopilotProvider } from "../components/copilot/CopilotProvider";

export const metadata = {
  title: "Ultimate AI RAG",
  description: "Agentic RAG + GraphRAG with CopilotKit",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}): React.ReactElement {
  return (
    <html lang="en">
      <body>
        <CopilotProvider>
          {children}
        </CopilotProvider>
      </body>
    </html>
  );
}
