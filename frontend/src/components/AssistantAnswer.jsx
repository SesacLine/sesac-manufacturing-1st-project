import React, { useState } from 'react'

// ---- failure_history(DB) 코드 → 한글 라벨 ----
const FT_KO = {
  TWF: '공구 마모', HDF: '열/냉각', OSF: '과부하', PWF: '전원/구동',
  RNF: '무고장', SAFETY_INTERLOCK: '안전 인터록',
}
const COMP_KO = {
  tooling: '공구', spindle_bearing: '스핀들 베어링', spindle_drive: '스핀들 드라이브',
  drive_system: '구동 시스템', coolant_system: '쿨런트 시스템',
  guard_interlock: '가드 인터록', drive_fan: '드라이브 팬',
}
const FIELD_KO = {
  event_date: '일자', failure_type: '고장유형', severity: '심각도', component: '부품/영역',
  symptom: '증상', root_cause: '근본원인', corrective_action: '조치', preventive_action: '예방조치',
  downtime_min: '다운타임(분)', case_count: '건수', total_downtime_min: '총 다운타임(분)', notes: '비고',
}
// 카드 본문에 노출할 필드(헤더에 쓰는 3개는 제외). 그 외 컬럼(id, *_json)은 숨긴다.
const BODY_FIELDS = [
  'severity', 'symptom', 'root_cause', 'corrective_action', 'preventive_action',
  'downtime_min', 'case_count', 'total_downtime_min', 'notes',
]

function display(key, val) {
  if (val === null || val === undefined || val === '') return null
  if (key === 'failure_type') return FT_KO[val] ? `${FT_KO[val]} (${val})` : String(val)
  if (key === 'component') return COMP_KO[val] || String(val)
  return String(val)
}

// failure_history 한 행을 카드로 렌더. detail 행과 aggregate 행을 모두 처리한다.
function DbCard({ row }) {
  const headFt = display('failure_type', row.failure_type) || '조회 결과'
  const headComp = display('component', row.component)
  return (
    <div className="db-card">
      <div className="db-card-head">
        <span className="db-card-ft">{headFt}</span>
        {headComp && <span className="db-card-comp">{headComp}</span>}
        {row.event_date && <span className="db-card-date">{row.event_date}</span>}
      </div>
      <dl className="db-card-body">
        {BODY_FIELDS.map((k) => {
          const v = display(k, row[k])
          return v ? (
            <div className="db-card-field" key={k}>
              <dt>{FIELD_KO[k] || k}</dt>
              <dd>{v}</dd>
            </div>
          ) : null
        })}
      </dl>
    </div>
  )
}

// Renders an assistant response: answer text, SQL(DB) result cards, clickable
// citations (with snippet preview), warnings, and an optional debug trace box.
export default function AssistantAnswer({ data }) {
  const [warnOpen, setWarnOpen] = useState(false)
  const [sqlOpen, setSqlOpen] = useState(false)
  const [openCite, setOpenCite] = useState(null)
  const [openRef, setOpenRef] = useState(null)
  const {
    answer,
    citations = [],
    data_refs = [],
    warnings = [],
    missing_inputs = [],
    blocked,
    sql,
    evidence,
    trace,
  } = data || {}

  const sqlRows = Array.isArray(sql?.rows) ? sql.rows : []
  // evidence가 OK가 아니면(문서 없음/관련성 낮음/실패) 안내 문구를 보여준다.
  const EV_NOTE = { EMPTY: '관련 문서 근거를 찾지 못했습니다.', LOW_RELEVANCE: '검색된 문서의 관련성이 낮습니다.', FAIL: '문서 근거 조회에 실패했습니다.' }
  const evNote = evidence && evidence.status && evidence.status !== 'OK' ? EV_NOTE[evidence.status] : null

  return (
    <div className={`assistant-answer ${blocked ? 'blocked' : ''}`}>
      {blocked && <div className="notice-tag">⚠ 차단된 응답 (blocked)</div>}

      <div className="answer-text">{answer}</div>

      {missing_inputs.length > 0 && (
        <div className="missing-inputs">
          <span className="section-label">추가 입력이 필요합니다:</span>{' '}
          {missing_inputs.join(', ')}
        </div>
      )}

      {sqlRows.length > 0 && (
        <div className="db-section">
          <button
            type="button"
            className="db-toggle"
            onClick={() => setSqlOpen((o) => !o)}
          >
            🗂 고장 이력 DB 조회 결과 {sql.row_count ?? sqlRows.length}건 {sqlOpen ? '▼' : '▶'}
          </button>
          {sqlOpen && (
            <div className="db-cards">
              {sqlRows.map((row, i) => (
                <DbCard key={row.id ?? i} row={row} />
              ))}
            </div>
          )}
        </div>
      )}

      {evNote && citations.length === 0 && (
        <div className="evidence-note muted small">📄 {evNote}</div>
      )}

      {citations.length > 0 && (
        <div className="citations">
          <div className="section-label">
            문서 근거 {citations.length}건 (클릭하면 원문 일부){evNote ? ` · ${evNote}` : ''}
          </div>
          <ul className="cite-list">
            {citations.map((c, i) => {
              const id = c.citation_id ?? `C${i + 1}`
              const open = openCite === id
              return (
                <li key={id}>
                  <button
                    type="button"
                    className={`cite-chip ${open ? 'open' : ''}`}
                    onClick={() => setOpenCite(open ? null : id)}
                  >
                    <span className="cite-id">[{id}]</span>{' '}
                    {c.title || c.citation_id}
                    {c.chunk_index != null && (
                      <span className="cite-chunk"> · chunk {c.chunk_index}</span>
                    )}
                    <span className="cite-caret">{open ? ' ▼' : ' ▶'}</span>
                  </button>
                  {open && (
                    <div className="cite-detail">
                      {c.source && <div className="cite-source">원본: {c.source}</div>}
                      {c.snippet ? (
                        <div className="cite-snippet">{c.snippet}</div>
                      ) : (
                        <div className="muted small">미리볼 본문이 없습니다.</div>
                      )}
                    </div>
                  )}
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {data_refs.length > 0 && (
        <div className="data-refs">
          <div className="section-label">데이터 출처 {data_refs.length}건 (클릭하면 조회 내역)</div>
          <ul className="cite-list">
            {data_refs.map((d, i) => {
              const id = d.ref_id ?? `D${i + 1}`
              const open = openRef === id
              const more = d.truncated ? '+' : ''
              return (
                <li key={id}>
                  <button
                    type="button"
                    className={`cite-chip ${open ? 'open' : ''}`}
                    onClick={() => setOpenRef(open ? null : id)}
                  >
                    <span className="cite-id">[{id}]</span>{' '}
                    {d.label || '데이터'}
                    <span className="cite-chunk"> · {d.row_count ?? 0}{more}행</span>
                    {d.time_window && <span className="cite-chunk"> · {d.time_window}</span>}
                    <span className="cite-caret">{open ? ' ▼' : ' ▶'}</span>
                  </button>
                  {open && (
                    <div className="cite-detail">
                      {d.sql && <div className="cite-snippet">{d.sql}</div>}
                    </div>
                  )}
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {warnings.length > 0 && (
        <div className="warnings">
          <button
            type="button"
            className="collapse-toggle muted"
            onClick={() => setWarnOpen((o) => !o)}
          >
            {warnOpen ? '▼' : '▶'} 경고 {warnings.length}건
          </button>
          {warnOpen && (
            <ul className="muted small">
              {warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {trace && (
        <div className="trace-box">
          <div className="section-label">debug trace</div>
          {Array.isArray(trace.gates) && trace.gates.length > 0 && (
            <pre>
              gates:
              {'\n'}
              {trace.gates
                .map(([name, status]) => `  ${name}: ${status}`)
                .join('\n')}
            </pre>
          )}
          {Array.isArray(trace.tasks) && trace.tasks.length > 0 && (
            <pre>
              tasks:
              {'\n'}
              {trace.tasks.map((t) => `  - ${t}`).join('\n')}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}
