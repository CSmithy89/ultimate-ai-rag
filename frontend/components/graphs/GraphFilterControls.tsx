/**
 * Filter controls for knowledge graph visualization.
 * Story 4.4: Knowledge Graph Visualization
 */

'use client';

import { useCallback } from 'react';
import type { EntityType, GraphFilterState, RelationshipType } from '../../types/graphs';
import { entityColors } from '../../types/graphs';

interface GraphFilterControlsProps {
  filters: GraphFilterState;
  onFiltersChange: (filters: GraphFilterState) => void;
}

const ENTITY_TYPES: EntityType[] = [
  'Person',
  'Organization',
  'Technology',
  'Concept',
  'Location',
];

const RELATIONSHIP_TYPES: RelationshipType[] = [
  'MENTIONS',
  'AUTHORED_BY',
  'PART_OF',
  'USES',
  'RELATED_TO',
];

/**
 * Filter controls component for the knowledge graph.
 * Allows filtering by entity type, relationship type, and orphan status.
 */
export function GraphFilterControls({
  filters,
  onFiltersChange,
}: GraphFilterControlsProps) {
  const handleEntityTypeToggle = useCallback(
    (entityType: EntityType) => {
      const newTypes = filters.entityTypes.includes(entityType)
        ? filters.entityTypes.filter((t) => t !== entityType)
        : [...filters.entityTypes, entityType];
      onFiltersChange({ ...filters, entityTypes: newTypes });
    },
    [filters, onFiltersChange]
  );

  const handleRelationshipTypeToggle = useCallback(
    (relType: RelationshipType) => {
      const newTypes = filters.relationshipTypes.includes(relType)
        ? filters.relationshipTypes.filter((t) => t !== relType)
        : [...filters.relationshipTypes, relType];
      onFiltersChange({ ...filters, relationshipTypes: newTypes });
    },
    [filters, onFiltersChange]
  );

  const handleOrphansToggle = useCallback(() => {
    onFiltersChange({ ...filters, showOrphansOnly: !filters.showOrphansOnly });
  }, [filters, onFiltersChange]);

  const handleSearchChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      onFiltersChange({ ...filters, searchQuery: event.target.value });
    },
    [filters, onFiltersChange]
  );

  const handleClearFilters = useCallback(() => {
    onFiltersChange({
      entityTypes: [],
      relationshipTypes: [],
      showOrphansOnly: false,
      searchQuery: '',
    });
  }, [onFiltersChange]);

  const hasActiveFilters =
    filters.entityTypes.length > 0 ||
    filters.relationshipTypes.length > 0 ||
    filters.showOrphansOnly ||
    filters.searchQuery.length > 0;

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 space-y-4">
      {/* Search input */}
      <div>
        <label htmlFor="search" className="block text-sm font-medium text-gray-700 mb-1">
          Search Entities
        </label>
        <input
          id="search"
          type="text"
          value={filters.searchQuery}
          onChange={handleSearchChange}
          placeholder="Search by name..."
          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />
      </div>

      {/* Entity type filters */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Entity Types
        </label>
        <div className="flex flex-wrap gap-2">
          {ENTITY_TYPES.map((type) => (
            <button
              key={type}
              onClick={() => handleEntityTypeToggle(type)}
              className="px-3 py-1.5 rounded-full text-xs font-medium transition-all"
              style={{
                backgroundColor: filters.entityTypes.includes(type)
                  ? entityColors[type]
                  : '#F3F4F6',
                color: filters.entityTypes.includes(type) ? 'white' : '#374151',
              }}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* Relationship type filters */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Relationship Types
        </label>
        <div className="flex flex-wrap gap-2">
          {RELATIONSHIP_TYPES.map((type) => (
            <button
              key={type}
              onClick={() => handleRelationshipTypeToggle(type)}
              className="px-3 py-1.5 rounded-full text-xs font-medium transition-all"
              style={{
                backgroundColor: filters.relationshipTypes.includes(type)
                  ? '#6366F1'
                  : '#F3F4F6',
                color: filters.relationshipTypes.includes(type) ? 'white' : '#374151',
              }}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* Orphan filter toggle */}
      <div className="flex items-center justify-between">
        <label htmlFor="orphans-toggle" className="text-sm font-medium text-gray-700">
          Show Orphans Only
        </label>
        <button
          id="orphans-toggle"
          onClick={handleOrphansToggle}
          className="relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-orange-500 focus:ring-offset-2"
          style={{
            backgroundColor: filters.showOrphansOnly ? '#F97316' : '#D1D5DB',
          }}
          role="switch"
          aria-checked={filters.showOrphansOnly}
        >
          <span
            className="pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out"
            style={{
              transform: filters.showOrphansOnly
                ? 'translateX(1.25rem)'
                : 'translateX(0)',
            }}
          />
        </button>
      </div>

      {/* Clear filters button */}
      {hasActiveFilters && (
        <button
          onClick={handleClearFilters}
          className="w-full px-3 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
        >
          Clear All Filters
        </button>
      )}
    </div>
  );
}

export default GraphFilterControls;
