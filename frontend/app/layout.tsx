import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { CopilotProvider } from "../components/copilot/CopilotProvider";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

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
      <body className={`${inter.variable} ${jetbrainsMono.variable} font-sans`}>
        <CopilotProvider>
          {children}
        </CopilotProvider>
      </body>
    </html>
  );
}
