"use client";

/**
 * useMultimodalInput - AG-UI Enhancement (Phase 7.2)
 *
 * Provides ability to attach files, images, and audio to messages.
 * Handles base64 encoding and content type detection.
 *
 * @example
 * ```tsx
 * function ChatInput() {
 *   const { attachments, addFile, removeAttachment, sendMultimodalMessage, clearAttachments } = useMultimodalInput();
 *
 *   const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
 *     const file = e.target.files?.[0];
 *     if (file) await addFile(file);
 *   };
 *
 *   const handleSubmit = async (text: string) => {
 *     await sendMultimodalMessage(text);
 *     clearAttachments();
 *   };
 *
 *   return (
 *     <div>
 *       <input type="file" onChange={handleFileSelect} />
 *       <AttachmentList attachments={attachments} onRemove={removeAttachment} />
 *       <textarea onSubmit={handleSubmit} />
 *     </div>
 *   );
 * }
 * ```
 */

import { useState, useCallback, useMemo } from "react";
import {
  SUPPORTED_MEDIA_TYPES,
  MAX_FILE_SIZE,
  MAX_ATTACHMENTS,
  type SupportedMediaType,
  type TextInputContent,
  type BinaryInputContent,
  type MultimodalContentPart,
  type Attachment,
} from "@/types/ag-ui";

// Re-export types for backward compatibility
export {
  SUPPORTED_MEDIA_TYPES,
  type SupportedMediaType,
  type TextInputContent,
  type BinaryInputContent,
  type MultimodalContentPart,
  type Attachment,
} from "@/types/ag-ui";

/**
 * Return type for useMultimodalInput hook.
 */
export interface UseMultimodalInputResult {
  /** List of attached files */
  attachments: Attachment[];
  /** Add a file attachment */
  addFile: (file: File) => Promise<void>;
  /** Remove an attachment by ID */
  removeAttachment: (id: string) => void;
  /** Clear all attachments */
  clearAttachments: () => void;
  /** Build multimodal content array from text and attachments */
  buildMultimodalContent: (text: string) => MultimodalContentPart[];
  /** Send a multimodal message (requires onSend callback) */
  sendMultimodalMessage: (text: string) => Promise<void>;
  /** Whether an attachment is being processed */
  isProcessing: boolean;
  /** Error message if any operation failed */
  error: string | null;
  /** Clear the error */
  clearError: () => void;
  /** Check if a file type is supported */
  isSupported: (file: File) => boolean;
  /** Get attachment stats */
  stats: {
    count: number;
    totalSize: number;
    hasImages: boolean;
    hasAudio: boolean;
    hasDocuments: boolean;
  };
}

/**
 * Configuration for useMultimodalInput.
 */
export interface UseMultimodalInputConfig {
  /** Maximum file size in bytes (default: 10MB) */
  maxFileSize?: number;
  /** Maximum number of attachments (default: 5) */
  maxAttachments?: number;
  /** Callback when sending multimodal message */
  onSend?: (content: MultimodalContentPart[]) => Promise<void>;
  /** Callback on error */
  onError?: (error: Error) => void;
}

/**
 * Convert a File to base64 string.
 */
async function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Remove the data URL prefix (e.g., "data:image/png;base64,")
      const base64 = result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

/**
 * Generate a unique ID for attachments.
 */
function generateId(): string {
  return `attachment-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
}

/**
 * Hook to handle multimodal input (files, images, audio) for messages.
 */
export function useMultimodalInput(
  config: UseMultimodalInputConfig = {}
): UseMultimodalInputResult {
  const {
    maxFileSize = MAX_FILE_SIZE,
    maxAttachments = MAX_ATTACHMENTS,
    onSend,
    onError,
  } = config;

  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Clear error
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // Check if file type is supported
  const isSupported = useCallback((file: File): boolean => {
    return file.type in SUPPORTED_MEDIA_TYPES;
  }, []);

  // Add a file attachment
  const addFile = useCallback(
    async (file: File): Promise<void> => {
      // Validate file type
      if (!isSupported(file)) {
        const error = new Error(`Unsupported file type: ${file.type}`);
        setError(error.message);
        onError?.(error);
        return;
      }

      // Validate file size
      if (file.size > maxFileSize) {
        const error = new Error(
          `File too large: ${file.name} (max ${maxFileSize / 1024 / 1024}MB)`
        );
        setError(error.message);
        onError?.(error);
        return;
      }

      // Validate attachment count
      if (attachments.length >= maxAttachments) {
        const error = new Error(
          `Maximum attachments reached (${maxAttachments})`
        );
        setError(error.message);
        onError?.(error);
        return;
      }

      setIsProcessing(true);
      setError(null);

      try {
        const base64 = await fileToBase64(file);

        // Create preview for images
        let preview: string | undefined;
        if (file.type.startsWith("image/")) {
          preview = `data:${file.type};base64,${base64}`;
        }

        const attachment: Attachment = {
          id: generateId(),
          type: "binary",
          media_type: file.type,
          data: base64,
          filename: file.name,
          size: file.size,
          preview,
        };

        setAttachments((prev) => [...prev, attachment]);
      } catch (err) {
        const error =
          err instanceof Error ? err : new Error("Failed to process file");
        setError(error.message);
        onError?.(error);
      } finally {
        setIsProcessing(false);
      }
    },
    [attachments.length, isSupported, maxFileSize, maxAttachments, onError]
  );

  // Remove an attachment
  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id));
  }, []);

  // Clear all attachments
  const clearAttachments = useCallback(() => {
    setAttachments([]);
  }, []);

  // Build multimodal content array
  const buildMultimodalContent = useCallback(
    (text: string): MultimodalContentPart[] => {
      const content: MultimodalContentPart[] = [];

      // Add text content if present
      if (text.trim()) {
        content.push({ type: "text", content: text });
      }

      // Add binary attachments
      for (const attachment of attachments) {
        content.push({
          type: "binary",
          media_type: attachment.media_type,
          data: attachment.data,
          filename: attachment.filename,
        });
      }

      return content;
    },
    [attachments]
  );

  // Send multimodal message
  const sendMultimodalMessage = useCallback(
    async (text: string): Promise<void> => {
      if (!onSend) {
        setError("No onSend callback configured");
        return;
      }

      const content = buildMultimodalContent(text);

      if (content.length === 0) {
        setError("Message cannot be empty");
        return;
      }

      try {
        await onSend(content);
      } catch (err) {
        const error =
          err instanceof Error ? err : new Error("Failed to send message");
        setError(error.message);
        onError?.(error);
      }
    },
    [buildMultimodalContent, onSend, onError]
  );

  // Calculate attachment stats
  const stats = useMemo(() => {
    const totalSize = attachments.reduce((sum, a) => sum + a.size, 0);
    const hasImages = attachments.some((a) => a.media_type.startsWith("image/"));
    const hasAudio = attachments.some((a) => a.media_type.startsWith("audio/"));
    const hasDocuments = attachments.some(
      (a) =>
        a.media_type === "application/pdf" || a.media_type === "text/plain"
    );

    return {
      count: attachments.length,
      totalSize,
      hasImages,
      hasAudio,
      hasDocuments,
    };
  }, [attachments]);

  return useMemo(
    () => ({
      attachments,
      addFile,
      removeAttachment,
      clearAttachments,
      buildMultimodalContent,
      sendMultimodalMessage,
      isProcessing,
      error,
      clearError,
      isSupported,
      stats,
    }),
    [
      attachments,
      addFile,
      removeAttachment,
      clearAttachments,
      buildMultimodalContent,
      sendMultimodalMessage,
      isProcessing,
      error,
      clearError,
      isSupported,
      stats,
    ]
  );
}

export default useMultimodalInput;
