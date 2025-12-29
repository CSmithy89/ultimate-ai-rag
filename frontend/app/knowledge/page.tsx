/**
 * Knowledge Graph visualization page.
 * Story 4.4: Knowledge Graph Visualization
 */

'use client';

import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  KnowledgeGraph,
  GraphFilterControls,
} from '../../components/graphs';
import {
  useKnowledgeGraph,
  useKnowledgeStats,
} from '../../hooks/use-knowledge-graph';
import type { GraphFilterState, GraphNode, GraphEdge } from '../../types/graphs';
import { entityColors } from '../../types/graphs';

// Demo tenant ID for development - in production this would come from auth context
const DEMO_TENANT_ID = '00000000-0000-0000-0000-000000000001';

/**
 * Stats panel component displaying graph statistics.
 */
function StatsPanel({
  nodeCount,
  edgeCount,
  orphanCount,
  entityTypeCounts,
}: {
  nodeCount: number;
  edgeCount: number;
  orphanCount: number;
  entityTypeCounts: Record<string, number>;
}) {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">Graph Statistics</h3>
      
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">{nodeCount}</div>
          <div className="text-xs text-gray-500">Entities</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-indigo-600">{edgeCount}</div>
          <div className="text-xs text-gray-500">Relationships</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-orange-500">{orphanCount}</div>
          <div className="text-xs text-gray-500">Orphans</div>
        </div>
      </div>
      
      {Object.keys(entityTypeCounts).length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-gray-600 mb-2">By Type</h4>
          <div className="space-y-1">
            {Object.entries(entityTypeCounts).map(([type, count]) => (
              <div key={type} className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span
                    className="w-3 h-3 rounded-full"
                    style={{
                      backgroundColor:
                        entityColors[type as keyof typeof entityColors] || '#6B7280',
                    }}
                  />
                  <span className="text-gray-600">{type}</span>
                </div>
                <span className="font-medium text-gray-800">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Loading skeleton for the graph.
 */
function GraphSkeleton() {
  return (
    <div className="w-full h-full bg-gray-50 rounded-lg flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p className="text-gray-500">Loading knowledge graph...</p>
      </div>
    </div>
  );
}

/**
 * Error display component.
 */
function ErrorDisplay({ error }: { error: Error }) {
  return (
    <div className="w-full h-full bg-red-50 rounded-lg flex items-center justify-center p-8">
      <div className="text-center max-w-md">
        <div className="text-red-500 text-4xl mb-4">!</div>
        <h3 className="text-lg font-semibold text-red-800 mb-2">
          Failed to Load Graph
        </h3>
        <p className="text-sm text-red-600">{error.message}</p>
      </div>
    </div>
  );
}

/**
 * Empty state component.
 */
function EmptyState() {
  return (
    <div className="w-full h-full bg-gray-50 rounded-lg flex items-center justify-center">
      <div className="text-center">
        <div className="text-gray-400 text-5xl mb-4">No Data</div>
        <p className="text-gray-500">No entities found in the knowledge graph.</p>
        <p className="text-gray-400 text-sm mt-2">
          Start by ingesting documents to populate the graph.
        </p>
      </div>
    </div>
  );
}

/**
 * Selected node detail panel.
 */
function NodeDetailPanel({
  node,
  onClose,
}: {
  node: GraphNode;
  onClose: () => void;
}) {
  const description = node.properties?.description;
  
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-700">Entity Details</h3>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-gray-600"
        >
          x
        </button>
      </div>
      
      <div className="space-y-2">
        <div>
          <span className="text-xs text-gray-500">Name</span>
          <p className="text-sm font-medium text-gray-800">{node.label}</p>
        </div>
        <div>
          <span className="text-xs text-gray-500">Type</span>
          <p className="text-sm font-medium text-gray-800">{node.type}</p>
        </div>
        {node.is_orphan && (
          <div className="bg-orange-50 text-orange-700 text-xs px-2 py-1 rounded">
            This entity has no relationships
          </div>
        )}
        { typeof description === "string" && description && (
          <div>
            <span className="text-xs text-gray-500">Description</span>
            <p className="text-sm text-gray-600">
              {String(description)}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Main content component with data fetching.
 */
function KnowledgePageContent() {
  const [filters, setFilters] = useState<GraphFilterState>({
    entityTypes: [],
    relationshipTypes: [],
    showOrphansOnly: false,
    searchQuery: '',
  });
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  // Fetch graph data
  const {
    data: graphData,
    isLoading: graphLoading,
    error: graphError,
  } = useKnowledgeGraph({
    tenantId: DEMO_TENANT_ID,
    limit: 100,
  });

  // Fetch stats
  const { data: statsData } = useKnowledgeStats(DEMO_TENANT_ID);

  // Handle node click
  const handleNodeClick = (node: GraphNode) => {
    setSelectedNode(node);
  };

  // Handle edge click
  const handleEdgeClick = (edge: GraphEdge) => {
    console.log('Edge clicked:', edge);
  };

  // Determine what to render
  const renderContent = () => {
    if (graphLoading) {
      return <GraphSkeleton />;
    }

    if (graphError) {
      return <ErrorDisplay error={graphError as Error} />;
    }

    if (!graphData || graphData.nodes.length === 0) {
      return <EmptyState />;
    }

    return (
      <KnowledgeGraph
        data={graphData}
        filters={filters}
        onNodeClick={handleNodeClick}
        onEdgeClick={handleEdgeClick}
      />
    );
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-xl font-bold text-gray-900">Knowledge Graph</h1>
          <p className="text-sm text-gray-500 mt-1">
            Visualize entities and their relationships
          </p>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-12 gap-6">
          {/* Left sidebar - Filters */}
          <div className="col-span-12 lg:col-span-3 space-y-4">
            <GraphFilterControls
              filters={filters}
              onFiltersChange={setFilters}
            />
            
            {statsData && (
              <StatsPanel
                nodeCount={statsData.node_count}
                edgeCount={statsData.edge_count}
                orphanCount={statsData.orphan_count}
                entityTypeCounts={statsData.entity_type_counts}
              />
            )}
            
            {selectedNode && (
              <NodeDetailPanel
                node={selectedNode}
                onClose={() => setSelectedNode(null)}
              />
            )}
          </div>

          {/* Main graph area */}
          <div className="col-span-12 lg:col-span-9">
            <div
              className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden"
              style={{ height: 'calc(100vh - 200px)', minHeight: 500 }}
            >
              {renderContent()}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

/**
 * Knowledge Graph page with query client provider.
 * QueryClient is created inside the component using useState to avoid SSR issues.
 */
export default function KnowledgePage() {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 30000,
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <KnowledgePageContent />
    </QueryClientProvider>
  );
}
