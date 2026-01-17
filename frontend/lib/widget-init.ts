/**
 * Widget Initialization - AG-UI Enhancement
 *
 * Initializes the widget registry with all available widgets.
 * Should be called once at app startup.
 */

import { registerWidget } from "./widget-registry";

// Widget imports are done dynamically to avoid circular dependencies
let initialized = false;

/**
 * Initialize the widget registry with all core widgets.
 * Safe to call multiple times - only initializes once.
 */
export async function initializeWidgets(): Promise<void> {
  if (initialized) {
    return;
  }

  // Dynamically import widgets to avoid bundling issues
  const [
    { StepProgressWidget },
    { ApprovalDialogWidget },
    { DataTableWidget },
    { ActivityTrackerWidget },
    { StatusIndicatorWidget },
    { BarChartWidget, LineChartWidget, PieChartWidget },
  ] = await Promise.all([
    import("@/components/widgets/StepProgressWidget"),
    import("@/components/widgets/ApprovalDialogWidget"),
    import("@/components/widgets/DataTableWidget"),
    import("@/components/widgets/ActivityTrackerWidget"),
    import("@/components/widgets/StatusIndicatorWidget"),
    import("@/components/widgets/ChartWidget"),
  ]);

  // Register core widgets
  registerWidget("step_progress", StepProgressWidget);
  registerWidget("approval_dialog", ApprovalDialogWidget);
  registerWidget("data_table", DataTableWidget);
  registerWidget("activity_tracker", ActivityTrackerWidget);
  registerWidget("status_indicator", StatusIndicatorWidget);
  registerWidget("bar_chart", BarChartWidget);
  registerWidget("line_chart", LineChartWidget);
  registerWidget("pie_chart", PieChartWidget);

  initialized = true;
}

/**
 * Synchronous widget initialization for use in components.
 * Requires widgets to be pre-imported.
 */
export function initializeWidgetsSync(): void {
  if (initialized) {
    return;
  }

  // Import widgets synchronously (requires bundler support)
  const { StepProgressWidget } = require("@/components/widgets/StepProgressWidget");
  const { ApprovalDialogWidget } = require("@/components/widgets/ApprovalDialogWidget");
  const { DataTableWidget } = require("@/components/widgets/DataTableWidget");
  const { ActivityTrackerWidget } = require("@/components/widgets/ActivityTrackerWidget");
  const { StatusIndicatorWidget } = require("@/components/widgets/StatusIndicatorWidget");
  const { BarChartWidget, LineChartWidget, PieChartWidget } = require("@/components/widgets/ChartWidget");

  registerWidget("step_progress", StepProgressWidget);
  registerWidget("approval_dialog", ApprovalDialogWidget);
  registerWidget("data_table", DataTableWidget);
  registerWidget("activity_tracker", ActivityTrackerWidget);
  registerWidget("status_indicator", StatusIndicatorWidget);
  registerWidget("bar_chart", BarChartWidget);
  registerWidget("line_chart", LineChartWidget);
  registerWidget("pie_chart", PieChartWidget);

  initialized = true;
}

/**
 * Check if widgets have been initialized.
 */
export function areWidgetsInitialized(): boolean {
  return initialized;
}

export default initializeWidgets;
