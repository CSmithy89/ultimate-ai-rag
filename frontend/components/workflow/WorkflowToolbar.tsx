/**
 * Workflow toolbar with action buttons.
 * Story 20-H6: Implement Visual Workflow Editor
 */

'use client';

import { memo, useState } from 'react';
import type { Workflow } from '../../types/workflow';

interface WorkflowToolbarProps {
  workflowName: string;
  onNameChange: (name: string) => void;
  onSave: () => boolean;
  onLoad: (id: string) => boolean;
  onClear: () => void;
  onRun: () => Promise<void>;
  getSavedWorkflows: () => Workflow[];
  isRunning?: boolean;
}

/**
 * Toolbar component with workflow actions.
 */
function WorkflowToolbarComponent({
  workflowName,
  onNameChange,
  onSave,
  onLoad,
  onClear,
  onRun,
  getSavedWorkflows,
  isRunning = false,
}: WorkflowToolbarProps) {
  const [showLoadMenu, setShowLoadMenu] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saved' | 'error'>('idle');

  const handleSave = () => {
    const success = onSave();
    setSaveStatus(success ? 'saved' : 'error');
    setTimeout(() => setSaveStatus('idle'), 2000);
  };

  const handleLoad = (id: string) => {
    onLoad(id);
    setShowLoadMenu(false);
  };

  const savedWorkflows = getSavedWorkflows();

  return (
    <div className="h-14 bg-white border-b border-gray-200 flex items-center justify-between px-4">
      {/* Left section - workflow name */}
      <div className="flex items-center gap-4">
        <input
          type="text"
          value={workflowName}
          onChange={(e) => onNameChange(e.target.value)}
          className="
            text-lg font-semibold text-gray-800
            border-0 border-b-2 border-transparent
            focus:border-blue-500 focus:outline-none
            bg-transparent px-1 py-0.5
          "
          placeholder="Workflow Name"
        />
      </div>

      {/* Right section - action buttons */}
      <div className="flex items-center gap-2">
        {/* Clear button */}
        <button
          onClick={onClear}
          className="
            px-3 py-1.5 text-sm font-medium
            text-gray-600 hover:text-gray-800
            hover:bg-gray-100 rounded-md
            transition-colors
          "
          title="Clear workflow"
        >
          Clear
        </button>

        {/* Load button with dropdown */}
        <div className="relative">
          <button
            onClick={() => setShowLoadMenu(!showLoadMenu)}
            className="
              px-3 py-1.5 text-sm font-medium
              text-gray-600 hover:text-gray-800
              hover:bg-gray-100 rounded-md
              transition-colors flex items-center gap-1
            "
          >
            Load
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </button>

          {showLoadMenu && (
            <div className="
              absolute right-0 top-full mt-1 w-64
              bg-white border border-gray-200 rounded-lg shadow-lg
              z-50 max-h-64 overflow-y-auto
            ">
              {savedWorkflows.length === 0 ? (
                <div className="px-4 py-3 text-sm text-gray-500">
                  No saved workflows
                </div>
              ) : (
                savedWorkflows.map((workflow) => (
                  <button
                    key={workflow.id}
                    onClick={() => handleLoad(workflow.id)}
                    className="
                      w-full px-4 py-2 text-left text-sm
                      hover:bg-gray-50 transition-colors
                      border-b border-gray-100 last:border-0
                    "
                  >
                    <div className="font-medium text-gray-800">
                      {workflow.name}
                    </div>
                    <div className="text-xs text-gray-500">
                      {new Date(workflow.updatedAt).toLocaleDateString()}
                    </div>
                  </button>
                ))
              )}
            </div>
          )}
        </div>

        {/* Save button */}
        <button
          onClick={handleSave}
          className={`
            px-3 py-1.5 text-sm font-medium rounded-md
            transition-colors flex items-center gap-1
            ${saveStatus === 'saved'
              ? 'bg-green-100 text-green-700'
              : saveStatus === 'error'
              ? 'bg-red-100 text-red-700'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }
          `}
        >
          {saveStatus === 'saved' ? (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              Saved
            </>
          ) : saveStatus === 'error' ? (
            'Error'
          ) : (
            'Save'
          )}
        </button>

        {/* Run button */}
        <button
          onClick={onRun}
          disabled={isRunning}
          className={`
            px-4 py-1.5 text-sm font-medium rounded-md
            transition-colors flex items-center gap-2
            ${isRunning
              ? 'bg-blue-400 text-white cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700'
            }
          `}
        >
          {isRunning ? (
            <>
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Running...
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Run
            </>
          )}
        </button>
      </div>

      {/* Click outside to close menu */}
      {showLoadMenu && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setShowLoadMenu(false)}
        />
      )}
    </div>
  );
}

export const WorkflowToolbar = memo(WorkflowToolbarComponent);
export default WorkflowToolbar;
