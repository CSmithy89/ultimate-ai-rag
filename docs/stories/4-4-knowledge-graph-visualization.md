# Story 4.4: Knowledge Graph Visualization

Status: drafted

## Story

As a data engineer,
I want to visualize the current state of the knowledge graph,
so that I can identify gaps, orphan nodes, and data quality issues.

## Acceptance Criteria

1. Given entities and relationships exist in Neo4j, when the user requests graph data via `GET /api/v1/knowledge/graph`, then the API returns nodes with id, label, type, and properties along with edges containing source, target, type, and properties.

2. Given the API has returned graph data, when the React Flow component renders, then nodes are displayed with their labels and type-based colors following the design system (Person: Blue, Organization: Emerald, Technology: Indigo, Concept: Violet, Location: Amber).

3. Given the graph is rendered, when edges are displayed, then each edge shows its relationship type label (MENTIONS, AUTHORED_BY, PART_OF, USES, RELATED_TO).

4. Given the graph visualization is displayed, when the user interacts with the canvas, then they can zoom in/out, pan across the graph, and click to select individual nodes.

5. Given nodes exist in the knowledge graph, when the orphan detection logic runs, then nodes with zero relationships are identified and highlighted in the warning color (Amber-400/Orange).

6. Given the graph contains multiple entity types, when the user applies entity type or relationship filters, then only nodes/edges matching the selected criteria are displayed.

7. Given the user requests graph statistics via `GET /api/v1/knowledge/stats`, then the API returns nodeCount, edgeCount, and orphanCount with proper tenant_id filtering.

## Tasks / Subtasks

- [ ] Create Knowledge Graph API endpoints (AC: 1, 5, 7)
  - [ ] Add `backend/src/agentic_rag_backend/api/routes/knowledge.py` router
  - [ ] Implement `GET /api/v1/knowledge/graph` endpoint with pagination and filtering
  - [ ] Implement `GET /api/v1/knowledge/stats` endpoint for node/edge counts
  - [ ] Implement `GET /api/v1/knowledge/orphans` endpoint for orphan node listing
  - [ ] Add tenant_id filtering to all queries
  - [ ] Register router in main.py

- [ ] Create React Flow graph component (AC: 2, 3, 4)
  - [ ] Add `frontend/src/components/graphs/KnowledgeGraph.tsx` main component
  - [ ] Configure force-directed layout using React Flow
  - [ ] Implement zoom, pan, and node selection controls
  - [ ] Add minimap and controls panel

- [ ] Create custom node component (AC: 2, 5)
  - [ ] Add `frontend/src/components/graphs/EntityNode.tsx` custom node
  - [ ] Implement entity type color coding per design system
  - [ ] Display node label and type badge
  - [ ] Add orphan node highlighting (Amber-400 border/glow)
  - [ ] Implement node selection state styling

- [ ] Create custom edge component (AC: 3)
  - [ ] Add `frontend/src/components/graphs/RelationshipEdge.tsx` custom edge
  - [ ] Display relationship type labels on edges
  - [ ] Style edges by relationship type (optional color coding)
  - [ ] Handle edge selection state

- [ ] Create filter controls (AC: 6)
  - [ ] Add `frontend/src/components/graphs/GraphFilterControls.tsx`
  - [ ] Implement entity type multi-select filter (Person, Organization, Technology, Concept, Location)
  - [ ] Implement relationship type filter
  - [ ] Add orphan-only toggle filter
  - [ ] Implement search/filter by entity name

- [ ] Create data fetching hook with TanStack Query (AC: 1, 7)
  - [ ] Add `frontend/src/hooks/use-knowledge-graph.ts`
  - [ ] Implement `useKnowledgeGraph` hook for graph data fetching
  - [ ] Implement `useKnowledgeStats` hook for statistics
  - [ ] Implement `useKnowledgeOrphans` hook for orphan nodes
  - [ ] Add query caching and invalidation

- [ ] Create Knowledge page (AC: 1-7)
  - [ ] Add `frontend/src/app/(features)/knowledge/page.tsx`
  - [ ] Integrate KnowledgeGraph component
  - [ ] Add stats summary panel
  - [ ] Include filter controls
  - [ ] Add loading and error states

- [ ] Update API client (AC: 1, 7)
  - [ ] Add knowledge graph API methods to `frontend/src/lib/api.ts`
  - [ ] Define TypeScript types in `frontend/src/types/graphs.ts`

- [ ] Create Pydantic models for API responses (AC: 1, 7)
  - [ ] Add graph response models to `backend/src/agentic_rag_backend/models/graphs.py`
  - [ ] Define GraphNode, GraphEdge, GraphData, GraphStats models
  - [ ] Ensure models follow standard API response format

- [ ] Write backend unit tests (AC: 1, 5, 7)
  - [ ] Add `backend/tests/api/test_knowledge.py` for API endpoints
  - [ ] Test graph data retrieval with mock Neo4j
  - [ ] Test stats endpoint
  - [ ] Test orphan detection
  - [ ] Test tenant isolation

- [ ] Write frontend tests (AC: 2-6)
  - [ ] Add `frontend/__tests__/components/knowledge-graph.test.tsx`
  - [ ] Test graph rendering with mock data
  - [ ] Test node/edge component rendering
  - [ ] Test filter functionality
  - [ ] Test zoom/pan interactions

## Dev Notes

### React Flow Integration

**Install dependencies:**
```bash
cd frontend && pnpm add reactflow
```

**Basic setup:**
```typescript
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
} from 'reactflow';
import 'reactflow/dist/style.css';

const KnowledgeGraph = ({ data }: { data: GraphData }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState(data.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(data.edges);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      fitView
    >
      <Controls />
      <MiniMap />
      <Background />
    </ReactFlow>
  );
};
```

### Entity Type Color Scheme

Follow the design system colors from the tech spec:
```typescript
const entityColors: Record<string, string> = {
  Person: '#3B82F6',        // Blue-500
  Organization: '#10B981',   // Emerald-500
  Technology: '#6366F1',     // Indigo-500
  Concept: '#8B5CF6',        // Violet-500
  Location: '#F59E0B',       // Amber-500
  orphan: '#F97316',         // Orange-500 (warning)
};
```

### API Endpoint Design

**GET /api/v1/knowledge/graph:**
```
Query params:
  - tenant_id (required): UUID
  - limit (optional): int, default 100
  - offset (optional): int, default 0
  - entity_type (optional): string filter
  - relationship_type (optional): string filter
  - date_from (optional): ISO date string

Response:
{
  "data": {
    "nodes": [
      {"id": "uuid", "label": "Entity Name", "type": "Technology", "properties": {...}, "isOrphan": false}
    ],
    "edges": [
      {"id": "uuid", "source": "uuid1", "target": "uuid2", "type": "USES", "label": "USES", "properties": {...}}
    ]
  },
  "meta": {"requestId": "uuid", "timestamp": "ISO8601"}
}
```

**GET /api/v1/knowledge/stats:**
```
Response:
{
  "data": {
    "nodeCount": 1500,
    "edgeCount": 3200,
    "orphanCount": 12,
    "entityTypeCounts": {"Person": 200, "Technology": 500, ...},
    "relationshipTypeCounts": {"USES": 800, "MENTIONS": 500, ...}
  },
  "meta": {...}
}
```

### Neo4j Queries

**Fetch graph data:**
```cypher
MATCH (n:Entity)
WHERE n.tenant_id = $tenant_id
OPTIONAL MATCH (n)-[r]->(m:Entity)
WHERE m.tenant_id = $tenant_id
RETURN n, r, m
SKIP $offset
LIMIT $limit
```

**Detect orphan nodes:**
```cypher
MATCH (n:Entity)
WHERE n.tenant_id = $tenant_id
AND NOT (n)-[]-()
RETURN n
```

**Graph statistics:**
```cypher
MATCH (n:Entity) WHERE n.tenant_id = $tenant_id
WITH count(n) as nodeCount
MATCH ()-[r]->() WHERE startNode(r).tenant_id = $tenant_id
WITH nodeCount, count(r) as edgeCount
MATCH (orphan:Entity) WHERE orphan.tenant_id = $tenant_id AND NOT (orphan)-[]-()
RETURN nodeCount, edgeCount, count(orphan) as orphanCount
```

### Component Structure

```
frontend/src/
├── app/(features)/knowledge/
│   └── page.tsx                    # Main knowledge page
├── components/graphs/
│   ├── KnowledgeGraph.tsx          # Main React Flow container
│   ├── EntityNode.tsx              # Custom node component
│   ├── RelationshipEdge.tsx        # Custom edge component
│   └── GraphFilterControls.tsx     # Filter UI controls
├── hooks/
│   └── use-knowledge-graph.ts      # TanStack Query hooks
├── lib/
│   └── api.ts                      # API client methods
└── types/
    └── graphs.ts                   # TypeScript types
```

### TanStack Query Data Fetching

As per project conventions, always use TanStack Query for data fetching:

```typescript
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';

export function useKnowledgeGraph(options?: GraphQueryOptions) {
  return useQuery({
    queryKey: ['knowledge', 'graph', options],
    queryFn: () => api.knowledge.getGraph(options),
    staleTime: 30000, // 30 seconds
  });
}

export function useKnowledgeStats() {
  return useQuery({
    queryKey: ['knowledge', 'stats'],
    queryFn: () => api.knowledge.getStats(),
    staleTime: 60000, // 1 minute
  });
}
```

### Multi-Tenancy Requirements

Every API endpoint and database query MUST include `tenant_id` filtering:
- API routes extract tenant_id from request context/headers
- All Neo4j queries filter by tenant_id property
- No cross-tenant data exposure

### Error Handling

Use RFC 7807 Problem Details format:
```json
{
  "type": "https://api.example.com/errors/graph-fetch-failed",
  "title": "Graph Fetch Failed",
  "status": 500,
  "detail": "Unable to retrieve graph data from Neo4j",
  "instance": "/api/v1/knowledge/graph"
}
```

### Performance Considerations

- Default limit of 100 nodes to prevent UI overload
- Use pagination for large graphs (offset/limit)
- Consider implementing graph subgraph expansion (click node to load neighbors)
- React Flow handles large graphs well with virtualization

## References

- Tech Spec: `docs/epics/epic-4-tech-spec.md#34-story-44-knowledge-graph-visualization`
- Architecture: `_bmad-output/architecture.md#frontend-architecture`
- Epic Definition: `_bmad-output/project-planning-artifacts/epics.md#story-44-knowledge-graph-visualization`
- UX Design: `_bmad-output/project-planning-artifacts/ux-design-specification.md` (Generative Graph Visualizer)
- Story 4.3 Reference: `docs/stories/4-3-agentic-entity-extraction.md`
- React Flow Documentation: https://reactflow.dev/
- TanStack Query: https://tanstack.com/query/latest

## Dev Agent Record

<!-- This section is filled in by the dev agent during implementation -->

### Agent Model Used

<!-- Model ID will be recorded here -->

### Debug Log References

<!-- Any debug notes from implementation -->

### Completion Notes List

<!-- Numbered list of implementation notes -->

### File List

<!-- List of files created/modified -->

## Senior Developer Review

<!-- This section is filled in by the code review workflow -->
