/**
 * Open-JSON-UI Image Component
 *
 * Renders images using Next.js Image component with URL validation.
 *
 * Story 22-C2: Implement Open-JSON-UI Renderer
 */

"use client";

import { memo, useState } from "react";
import Image from "next/image";
import { cn } from "@/lib/utils";
import { isValidUrl, sanitizeToPlainText } from "@/lib/open-json-ui/sanitize";
import { AlertCircle, ImageOff } from "lucide-react";
import type { ImageComponent as ImageComponentType } from "@/lib/open-json-ui/schema";

/**
 * Props for ImageComponent.
 */
export interface ImageComponentProps {
  /** Image component data */
  component: ImageComponentType;
}

/**
 * Default dimensions for images without explicit sizing.
 */
const DEFAULT_WIDTH = 400;
const DEFAULT_HEIGHT = 300;

/**
 * ImageComponent renders images with validation and fallback handling.
 *
 * Security: Only allows http/https URLs to prevent data: and javascript: attacks.
 *
 * @param props - Component props
 * @returns Rendered image or error state
 *
 * @example
 * ```tsx
 * <ImageComponent
 *   component={{
 *     type: "image",
 *     src: "https://example.com/image.png",
 *     alt: "Example image",
 *     width: 300,
 *     height: 200
 *   }}
 * />
 * ```
 */
export const ImageComponent = memo(function ImageComponent({
  component,
}: ImageComponentProps) {
  const [hasError, setHasError] = useState(false);
  const sanitizedAlt = sanitizeToPlainText(component.alt);
  const width = component.width ?? DEFAULT_WIDTH;
  const height = component.height ?? DEFAULT_HEIGHT;

  // Validate URL for security
  if (!isValidUrl(component.src)) {
    return (
      <div
        className={cn(
          "flex items-center gap-2 p-4 rounded-lg",
          "bg-red-50 border border-red-200 text-red-700"
        )}
        data-testid="open-json-ui-image-invalid"
        role="alert"
      >
        <AlertCircle className="h-5 w-5 flex-shrink-0" aria-hidden="true" />
        <span className="text-sm">Invalid image URL</span>
      </div>
    );
  }

  // Show fallback on load error
  if (hasError) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center gap-2 p-6 rounded-lg",
          "bg-slate-100 border border-slate-200 text-slate-500"
        )}
        style={{ width, height: Math.min(height, 150) }}
        data-testid="open-json-ui-image-error"
        role="img"
        aria-label={sanitizedAlt}
      >
        <ImageOff className="h-8 w-8" aria-hidden="true" />
        <span className="text-sm text-center">Failed to load image</span>
      </div>
    );
  }

  return (
    <div className="relative" data-testid="open-json-ui-image">
      <Image
        src={component.src}
        alt={sanitizedAlt}
        width={width}
        height={height}
        className="rounded-lg object-cover"
        onError={() => setHasError(true)}
        unoptimized // Allow external images
      />
    </div>
  );
});

export default ImageComponent;
