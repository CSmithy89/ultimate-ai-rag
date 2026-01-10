/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  /**
   * Story 21-E1, 21-E2: Proxy voice endpoints to backend
   *
   * Routes /api/copilot/* to backend /api/v1/copilot/*
   * This enables the VoiceInput and SpeakButton components to call
   * the backend transcription and TTS endpoints.
   */
  async rewrites() {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
    return [
      {
        source: "/api/copilot/:path*",
        destination: `${backendUrl}/api/v1/copilot/:path*`,
      },
    ];
  },
};

export default nextConfig;
