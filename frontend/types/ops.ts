import { z } from "zod";

export const CostEventSchema = z.object({
  id: z.string(),
  trajectory_id: z.string().nullable().optional(),
  model_id: z.string(),
  prompt_tokens: z.number(),
  completion_tokens: z.number(),
  total_tokens: z.number(),
  input_cost_usd: z.number(),
  output_cost_usd: z.number(),
  total_cost_usd: z.number(),
  baseline_model_id: z.string().nullable().optional(),
  baseline_total_cost_usd: z.number().nullable().optional(),
  savings_usd: z.number().nullable().optional(),
  complexity: z.string().nullable().optional(),
  created_at: z.string(),
});

export type CostEvent = z.infer<typeof CostEventSchema>;

export const CostSummarySchema = z.object({
  total_cost_usd: z.number(),
  baseline_cost_usd: z.number(),
  total_savings_usd: z.number(),
  total_tokens: z.number(),
  total_requests: z.number(),
  by_model: z.array(
    z.object({
      model_id: z.string(),
      requests: z.number(),
      total_tokens: z.number(),
      total_cost_usd: z.number(),
    })
  ),
  trend: z.array(
    z.object({
      bucket: z.string(),
      total_cost_usd: z.number(),
      total_tokens: z.number(),
    })
  ),
  alerts: z.record(z.string(), z.unknown()),
});

export type CostSummary = z.infer<typeof CostSummarySchema>;

export const AlertConfigSchema = z
  .object({
    tenant_id: z.string(),
    daily_threshold_usd: z.number().nullable().optional(),
    monthly_threshold_usd: z.number().nullable().optional(),
    enabled: z.boolean(),
  })
  .nullable();

export type AlertConfig = z.infer<typeof AlertConfigSchema>;

export const TrajectorySummarySchema = z.object({
  id: z.string(),
  session_id: z.string().nullable().optional(),
  agent_type: z.string().nullable().optional(),
  created_at: z.string(),
  has_error: z.boolean(),
  event_count: z.number(),
  last_event_at: z.string().nullable().optional(),
});

export type TrajectorySummary = z.infer<typeof TrajectorySummarySchema>;

export const TrajectoryEventSchema = z.object({
  id: z.string(),
  event_type: z.string(),
  content: z.string(),
  created_at: z.string(),
});

export type TrajectoryEvent = z.infer<typeof TrajectoryEventSchema>;

export const TrajectoryDetailSchema = z.object({
  trajectory: z.object({
    id: z.string(),
    session_id: z.string().nullable().optional(),
    agent_type: z.string().nullable().optional(),
    created_at: z.string(),
  }),
  events: z.array(TrajectoryEventSchema),
  duration_ms: z.number().nullable().optional(),
});

export type TrajectoryDetail = z.infer<typeof TrajectoryDetailSchema>;
