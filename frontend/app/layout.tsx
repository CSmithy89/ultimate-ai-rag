import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { CopilotProvider } from "../components/copilot/CopilotProvider";
import { Toaster } from "../components/ui/Toaster";
import { AppHeader } from "../components/layout/AppHeader";

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
  const hideCopilotInspector =
    process.env.NEXT_PUBLIC_SHOW_COPILOT_INSPECTOR !== "true";

  return (
    <html lang="en">
      <body
        data-hide-copilotkit-inspector={hideCopilotInspector ? "true" : "false"}
        className={`${inter.variable} ${jetbrainsMono.variable} font-sans`}
      >
        <CopilotProvider>
          <AppHeader />
          {children}
          <Toaster />
        </CopilotProvider>
      </body>
    </html>
  );
}
