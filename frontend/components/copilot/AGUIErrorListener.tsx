"use client";

import { useEffect, useRef } from "react";
import { useCopilotChat } from "@copilotkit/react-core";
import {
  AGUIErrorCode,
  parseAGUIError,
  useAGUIErrorHandler,
  type AGUIErrorData,
} from "./ErrorHandler";

function safeParseJson(value: string | undefined): unknown {
  if (!value) {
    return undefined;
  }
  try {
    return JSON.parse(value);
  } catch {
    return undefined;
  }
}

function buildFallbackError(reason?: string): AGUIErrorData {
  return {
    code: AGUIErrorCode.AGENT_EXECUTION_ERROR,
    message: reason || "An unexpected error occurred.",
    http_status: 500,
  };
}

/**
 * AGUIErrorListener wires AG-UI error handling into the CopilotKit message flow.
 * It watches for failed messages and surfaces them via useAGUIErrorHandler.
 */
export function AGUIErrorListener(): null {
  const { visibleMessages } = useCopilotChat();
  const { handleError } = useAGUIErrorHandler();
  const handledRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!visibleMessages || !Array.isArray(visibleMessages)) {
      return;
    }
    visibleMessages.forEach((message: any) => {
      const status = message?.status;
      if (!status || status.code !== "Failed") {
        return;
      }

      const messageId = message?.id || `${message?.createdAt || "unknown"}:${message?.type}`;
      if (handledRef.current.has(messageId)) {
        return;
      }
      handledRef.current.add(messageId);

      const parsed = safeParseJson(status.reason);
      const errorEvent = parseAGUIError(parsed as { type?: string; event?: string; data?: unknown });
      if (errorEvent) {
        handleError(errorEvent);
        return;
      }

      handleError(buildFallbackError(status.reason));
    });
  }, [visibleMessages, handleError]);

  return null;
}

export default AGUIErrorListener;
