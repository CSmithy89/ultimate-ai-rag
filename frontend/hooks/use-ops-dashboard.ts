import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { AlertConfig, CostEvent, CostSummary } from "@/types/ops";

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
