import "./globals.css";

export const metadata = {
  title: "Ultimate AI RAG",
  description: "Agentic RAG + GraphRAG with CopilotKit",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
