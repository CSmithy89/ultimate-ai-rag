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
