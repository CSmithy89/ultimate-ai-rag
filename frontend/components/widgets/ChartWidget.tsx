"use client";

/**
 * ChartWidget - AG-UI Enhancement (Phase 5)
 *
 * Renders simple charts (bar, line, pie) using SVG.
 * Lightweight implementation without external dependencies.
 *
 * @example
 * ```tsx
 * <ChartWidget
 *   type="bar"
 *   data={[
 *     { label: "Jan", value: 100 },
 *     { label: "Feb", value: 150 },
 *     { label: "Mar", value: 120 },
 *   ]}
 *   title="Monthly Sales"
 * />
 * ```
 */

import { useMemo } from "react";
import { cn } from "@/lib/utils";

/**
 * Chart data point.
 */
export interface ChartDataPoint {
  label: string;
  value: number;
  color?: string;
}

/**
 * Chart type.
 */
export type ChartType = "bar" | "line" | "pie";

/**
 * Props for ChartWidget.
 */
export interface ChartWidgetProps {
  /** Chart type */
  type: ChartType;
  /** Chart data */
  data: ChartDataPoint[];
  /** Chart title */
  title?: string;
  /** Whether to show legend */
  showLegend?: boolean;
  /** Chart width */
  width?: number;
  /** Chart height */
  height?: number;
  /** Custom class name */
  className?: string;
}

/**
 * Default colors for chart segments.
 */
const DEFAULT_COLORS = [
  "#4F46E5", // Indigo-600
  "#10B981", // Emerald-500
  "#F59E0B", // Amber-500
  "#EF4444", // Red-500
  "#8B5CF6", // Violet-500
  "#06B6D4", // Cyan-500
  "#EC4899", // Pink-500
  "#84CC16", // Lime-500
];

/**
 * Get color for a data point.
 */
function getColor(index: number, customColor?: string): string {
  return customColor || DEFAULT_COLORS[index % DEFAULT_COLORS.length];
}

/**
 * Bar Chart component.
 */
function BarChart({
  data,
  width,
  height,
}: {
  data: ChartDataPoint[];
  width: number;
  height: number;
}) {
  const maxValue = Math.max(...data.map((d) => d.value), 1);
  const barWidth = Math.max((width - 40) / data.length - 8, 20);
  const chartHeight = height - 40;

  return (
    <svg width={width} height={height} className="overflow-visible">
      {/* Y-axis */}
      <line x1={35} y1={10} x2={35} y2={chartHeight + 10} stroke="#e2e8f0" strokeWidth={1} />

      {/* X-axis */}
      <line x1={35} y1={chartHeight + 10} x2={width - 5} y2={chartHeight + 10} stroke="#e2e8f0" strokeWidth={1} />

      {/* Bars */}
      {data.map((point, i) => {
        const barHeight = (point.value / maxValue) * chartHeight;
        const x = 40 + i * (barWidth + 8);
        const y = chartHeight + 10 - barHeight;

        return (
          <g key={i}>
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={barHeight}
              fill={getColor(i, point.color)}
              rx={2}
              className="transition-all duration-300 hover:opacity-80"
            />
            <text
              x={x + barWidth / 2}
              y={chartHeight + 25}
              textAnchor="middle"
              className="fill-slate-500 text-[10px]"
            >
              {point.label.slice(0, 6)}
            </text>
            <text
              x={x + barWidth / 2}
              y={y - 5}
              textAnchor="middle"
              className="fill-slate-600 text-[10px] font-medium"
            >
              {point.value}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/**
 * Line Chart component.
 */
function LineChart({
  data,
  width,
  height,
}: {
  data: ChartDataPoint[];
  width: number;
  height: number;
}) {
  const maxValue = Math.max(...data.map((d) => d.value), 1);
  const chartHeight = height - 40;
  const chartWidth = width - 45;

  const points = data.map((point, i) => {
    const x = 40 + (i / Math.max(data.length - 1, 1)) * chartWidth;
    const y = chartHeight + 10 - (point.value / maxValue) * chartHeight;
    return { x, y, ...point };
  });

  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");
  const areaPath = `${linePath} L ${points[points.length - 1]?.x || 40} ${chartHeight + 10} L 40 ${chartHeight + 10} Z`;

  return (
    <svg width={width} height={height} className="overflow-visible">
      {/* Grid lines */}
      {[0, 0.25, 0.5, 0.75, 1].map((ratio) => (
        <line
          key={ratio}
          x1={35}
          y1={10 + chartHeight * (1 - ratio)}
          x2={width - 5}
          y2={10 + chartHeight * (1 - ratio)}
          stroke="#e2e8f0"
          strokeWidth={1}
          strokeDasharray={ratio === 0 ? "0" : "4"}
        />
      ))}

      {/* Area fill */}
      <path d={areaPath} fill={DEFAULT_COLORS[0]} fillOpacity={0.1} />

      {/* Line */}
      <path d={linePath} fill="none" stroke={DEFAULT_COLORS[0]} strokeWidth={2} />

      {/* Points */}
      {points.map((point, i) => (
        <g key={i}>
          <circle
            cx={point.x}
            cy={point.y}
            r={4}
            fill="white"
            stroke={getColor(i, point.color)}
            strokeWidth={2}
            className="transition-all duration-300 hover:r-6"
          />
          <text
            x={point.x}
            y={chartHeight + 25}
            textAnchor="middle"
            className="fill-slate-500 text-[10px]"
          >
            {point.label.slice(0, 6)}
          </text>
        </g>
      ))}
    </svg>
  );
}

/**
 * Pie Chart component.
 */
function PieChart({
  data,
  width,
  height,
}: {
  data: ChartDataPoint[];
  width: number;
  height: number;
}): React.ReactElement {
  // Memoize derived values to prevent unnecessary recalculations
  const chartDimensions = useMemo(() => {
    const total = data.reduce((sum, d) => sum + d.value, 0) || 1;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) / 2 - 20;
    return { total, centerX, centerY, radius };
  }, [data, width, height]);

  const { total, centerX, centerY, radius } = chartDimensions;

  // Memoize slices with stable dependencies
  const slices = useMemo(() => {
    let currentAngle = -Math.PI / 2;
    return data.map((point, i) => {
      const angle = (point.value / total) * 2 * Math.PI;
      const startAngle = currentAngle;
      const endAngle = currentAngle + angle;
      currentAngle = endAngle;

      const x1 = centerX + radius * Math.cos(startAngle);
      const y1 = centerY + radius * Math.sin(startAngle);
      const x2 = centerX + radius * Math.cos(endAngle);
      const y2 = centerY + radius * Math.sin(endAngle);

      const largeArcFlag = angle > Math.PI ? 1 : 0;

      const path = `M ${centerX} ${centerY} L ${x1} ${y1} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${x2} ${y2} Z`;

      // Label position
      const labelAngle = startAngle + angle / 2;
      const labelRadius = radius * 0.65;
      const labelX = centerX + labelRadius * Math.cos(labelAngle);
      const labelY = centerY + labelRadius * Math.sin(labelAngle);

      return {
        path,
        color: getColor(i, point.color),
        label: point.label,
        value: point.value,
        percentage: Math.round((point.value / total) * 100),
        labelX,
        labelY,
      };
    });
  }, [data, chartDimensions]);

  return (
    <svg width={width} height={height} className="overflow-visible">
      {slices.map((slice, i) => (
        <g key={i}>
          <path
            d={slice.path}
            fill={slice.color}
            stroke="white"
            strokeWidth={2}
            className="transition-all duration-300 hover:opacity-80"
          />
          {slice.percentage >= 8 && (
            <text
              x={slice.labelX}
              y={slice.labelY}
              textAnchor="middle"
              dominantBaseline="middle"
              className="fill-white text-[11px] font-medium pointer-events-none"
            >
              {slice.percentage}%
            </text>
          )}
        </g>
      ))}
    </svg>
  );
}

/**
 * Legend component.
 */
function Legend({ data }: { data: ChartDataPoint[] }) {
  return (
    <div className="flex flex-wrap gap-3 mt-3 justify-center">
      {data.map((point, i) => (
        <div key={i} className="flex items-center gap-1.5">
          <div
            className="w-3 h-3 rounded-sm"
            style={{ backgroundColor: getColor(i, point.color) }}
          />
          <span className="text-xs text-slate-600">{point.label}</span>
        </div>
      ))}
    </div>
  );
}

/**
 * ChartWidget renders bar, line, or pie charts from data.
 */
export function ChartWidget({
  type,
  data,
  title,
  showLegend = true,
  width = 300,
  height = 200,
  className,
}: ChartWidgetProps) {
  if (!data || data.length === 0) {
    return (
      <div className={cn("flex items-center justify-center p-4 text-slate-400 text-sm", className)}>
        No data available
      </div>
    );
  }

  // Map chart types to components with type safety
  const chartComponents = {
    bar: BarChart,
    line: LineChart,
    pie: PieChart,
  } as const;

  const ChartComponent = chartComponents[type];

  // Handle invalid chart type gracefully
  if (!ChartComponent) {
    return (
      <div
        className={cn(
          "flex items-center justify-center p-4 text-red-500 text-sm",
          className
        )}
        role="alert"
      >
        Invalid chart type: {type}
      </div>
    );
  }

  return (
    <div className={cn("rounded-lg border bg-white p-4", className)}>
      {title && (
        <h3 className="text-sm font-medium text-slate-700 mb-3">{title}</h3>
      )}
      <div className="flex justify-center">
        <ChartComponent data={data} width={width} height={height} />
      </div>
      {showLegend && type !== "bar" && <Legend data={data} />}
    </div>
  );
}

/**
 * Specialized exports for each chart type.
 */
export function BarChartWidget(props: Omit<ChartWidgetProps, "type">) {
  return <ChartWidget {...props} type="bar" />;
}

export function LineChartWidget(props: Omit<ChartWidgetProps, "type">) {
  return <ChartWidget {...props} type="line" />;
}

export function PieChartWidget(props: Omit<ChartWidgetProps, "type">) {
  return <ChartWidget {...props} type="pie" />;
}

export default ChartWidget;
