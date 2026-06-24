import React, { useState } from 'react'

// Renders the pipeline progress.
// steps: [{ node, label, detail }] in arrival order.
// - live mode (completed=false): the last step is in progress (spinner); earlier
//   steps are done. Used while waiting for the answer.
// - completed mode (completed=true): every step is done (✓), no spinner, and the
//   block is collapsible. Used to keep the flow visible above a finished answer.
export default function ProgressSteps({ steps, completed = false }) {
  const [open, setOpen] = useState(true)

  if (!steps || steps.length === 0) {
    if (completed) return null
    return (
      <div className="progress-steps">
        <div className="progress-title">진행 단계</div>
        <div className="loading-row">
          <span className="spinner" /> 답변을 생성하는 중입니다… (수 초 소요)
        </div>
      </div>
    )
  }

  const lastIndex = steps.length - 1
  return (
    <div className={`progress-steps ${completed ? 'completed' : ''}`}>
      {completed ? (
        <button
          type="button"
          className="progress-title progress-toggle"
          onClick={() => setOpen((o) => !o)}
        >
          진행 단계 {steps.length}단계 {open ? '▼' : '▶'}
        </button>
      ) : (
        <div className="progress-title">진행 단계</div>
      )}
      {open && (
        <ul className="step-list">
          {steps.map((s, i) => {
            const active = !completed && i === lastIndex
            return (
              <li
                key={`${s.node}-${i}`}
                className={`step-item ${active ? 'active' : 'done'}`}
              >
                <span className="step-icon">
                  {active ? <span className="spinner small-spinner" /> : '✓'}
                </span>
                <span className="step-body">
                  <span className="step-label">{s.label || s.node}</span>
                  {s.detail && <span className="step-detail">{s.detail}</span>}
                  {active && <span className="step-status muted small">진행 중</span>}
                </span>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
