/**
 * Tests for utility functions.
 * Story 6-2: Chat Sidebar Interface
 */

import { cn } from "../../lib/utils";

describe("cn utility function", () => {
  it("merges class names correctly", () => {
    const result = cn("text-red-500", "bg-blue-500");
    expect(result).toBe("text-red-500 bg-blue-500");
  });

  it("handles conditional classes", () => {
    const isActive = true;
    const result = cn("base-class", isActive && "active-class");
    expect(result).toBe("base-class active-class");
  });

  it("filters out falsy values", () => {
    const result = cn("base-class", false && "hidden", null, undefined);
    expect(result).toBe("base-class");
  });

  it("handles empty inputs", () => {
    const result = cn();
    expect(result).toBe("");
  });

  it("merges conflicting Tailwind classes correctly", () => {
    // tailwind-merge should resolve conflicts by keeping the last value
    const result = cn("text-red-500", "text-blue-500");
    expect(result).toBe("text-blue-500");
  });

  it("handles array inputs via clsx", () => {
    const result = cn(["class1", "class2"]);
    expect(result).toBe("class1 class2");
  });

  it("handles object inputs via clsx", () => {
    const result = cn({
      "base-class": true,
      "active-class": true,
      "disabled-class": false,
    });
    expect(result).toBe("base-class active-class");
  });

  it("handles complex mixed inputs", () => {
    const isActive = true;
    const isDisabled = false;
    const result = cn(
      "p-4",
      "text-slate-600",
      isActive && "text-indigo-600",
      isDisabled && "opacity-50",
      { "font-bold": true }
    );
    // text-indigo-600 should override text-slate-600
    expect(result).toBe("p-4 text-indigo-600 font-bold");
  });
});
