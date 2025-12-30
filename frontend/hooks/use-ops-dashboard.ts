import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  AlertConfig,
  CostEvent,
  CostSummary,
  TrajectoryDetail,
  TrajectorySummary,
} from "@/types/ops";

export function useCostSummary(tenantId: string, window: string) {
  return useQuery<CostSummary>({
    queryKey: ["ops", "costs", "summary", tenantId, window],
    queryFn: () => api.ops.getCostSummary(tenantId, window),
    staleTime: 30000,
    refetchInterval: 30000,
  });
}

export function useCostEvents(tenantId: string) {
  return useQuery<CostEvent[]>({
    queryKey: ["ops", "costs", "events", tenantId],
    queryFn: () => api.ops.getCostEvents(tenantId),
    staleTime: 15000,
    refetchInterval: 15000,
  });
}

export function useCostAlerts(tenantId: string) {
  return useQuery<AlertConfig>({
    queryKey: ["ops", "costs", "alerts", tenantId],
    queryFn: () => api.ops.getCostAlerts(tenantId),
    staleTime: 60000,
  });
}

export function useUpdateCostAlerts() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: api.ops.updateCostAlerts,
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["ops", "costs", "alerts", variables.tenant_id],
      });
    },
  });
}

export function useTrajectories(
  tenantId: string,
  options?: { status?: string; agentType?: string; limit?: number; offset?: number }
) {
  return useQuery<TrajectorySummary[]>({
    queryKey: ["ops", "trajectories", tenantId, options],
    queryFn: () => api.ops.getTrajectories(tenantId, options),
    staleTime: 20000,
    refetchInterval: 20000,
  });
}

export function useTrajectoryDetail(tenantId: string, trajectoryId: string | null) {
  return useQuery<TrajectoryDetail>({
    queryKey: ["ops", "trajectory", tenantId, trajectoryId],
    queryFn: () => api.ops.getTrajectoryDetail(tenantId, trajectoryId ?? ""),
    enabled: Boolean(trajectoryId),
    staleTime: 20000,
  });
}
