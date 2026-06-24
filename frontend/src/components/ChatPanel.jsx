import React, { useEffect, useRef, useState } from 'react'
import MessageList from './MessageList.jsx'
import ProgressSteps from './ProgressSteps.jsx'
import FeatureInputs, { buildInputFeatures, EMPTY_FEATURES } from './FeatureInputs.jsx'
import { api, ApiError, chatStream } from '../api.js'

function errorMessage(err) {
  if (err instanceof ApiError) {
    if (err.status === 404) return '사용자/대화를 먼저 만들어 주세요.'
    if (err.status === 503 && err.detail === 'llm_quota_exhausted')
      return 'LLM 사용량(쿼터)이 소진되어 답변을 생성할 수 없습니다. 크레딧 충전 후 다시 시도하세요.'
    if (err.status === 0) return '서버에 연결할 수 없습니다.'
  }
  return '오류가 발생했습니다. 잠시 후 다시 시도하세요.'
}

// Map a streaming error event ({status, code, message}) to a Korean message.
function streamErrorMessage(e) {
  if (!e) return '오류가 발생했습니다. 잠시 후 다시 시도하세요.'
  if (e.status === 404) return '사용자/대화를 먼저 만들어 주세요.'
  if (e.code === 'llm_quota_exhausted')
    return 'LLM 사용량(쿼터)이 소진되어 답변을 생성할 수 없습니다. 크레딧 충전 후 다시 시도하세요.'
  if (e.status === 0) return '서버에 연결할 수 없습니다.'
  return '오류가 발생했습니다. 잠시 후 다시 시도하세요.'
}

export default function ChatPanel({ userId, threadId }) {
  const [messages, setMessages] = useState([])
  const [message, setMessage] = useState('')
  const [features, setFeatures] = useState(EMPTY_FEATURES)
  const [debug, setDebug] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [steps, setSteps] = useState([])
  const abortRef = useRef(null)

  // Abort any in-flight stream when the thread changes or on unmount.
  useEffect(() => {
    return () => {
      abortRef.current?.abort()
    }
  }, [userId, threadId])

  // Load history whenever the selected thread changes.
  useEffect(() => {
    let cancelled = false
    setMessages([])
    setError('')
    if (!userId || !threadId) return
    api
      .getHistory(userId, threadId)
      .then((res) => {
        if (cancelled) return
        const turns = (res?.turns || []).map((t) => ({
          role: t.role === 'assistant' ? 'assistant' : 'user',
          content: t.content,
        }))
        setMessages(turns)
      })
      .catch((err) => {
        if (!cancelled) setError(errorMessage(err))
      })
    return () => {
      cancelled = true
    }
  }, [userId, threadId])

  const send = async () => {
    const text = message.trim()
    if (!text || loading) return
    setError('')
    const inputFeatures = buildInputFeatures(features)

    setMessages((m) => [...m, { role: 'user', content: text }])
    setMessage('')
    setSteps([])
    setLoading(true)

    const controller = new AbortController()
    abortRef.current = controller

    const body = {
      user_id: userId,
      thread_id: threadId,
      message: text,
      ...(inputFeatures ? { input_features: inputFeatures } : {}),
    }

    // 스트리밍 동안 도착한 진행 단계를 따로 모아둔다(완료 후에도 답변 위에 남기기 위해).
    const runSteps = []

    await chatStream(body, {
      debug,
      signal: controller.signal,
      onStart: () => {
        runSteps.length = 0
        setSteps([])
      },
      onStep: (evt) => {
        const step = { node: evt.node, label: evt.label, detail: evt.detail }
        runSteps.push(step)
        setSteps((s) => [...s, step])
      },
      onDone: (evt) => {
        const data = {
          user_id: evt.user_id,
          thread_id: evt.thread_id,
          answer: evt.answer,
          citations: evt.citations || [],
          data_refs: evt.data_refs || [],
          warnings: evt.warnings || [],
          missing_inputs: evt.missing_inputs || [],
          blocked: evt.blocked,
          sql: evt.sql || null,
          evidence: evt.evidence || null,
          trace: evt.trace || null,
        }
        // 진행 단계를 메시지에 함께 보존한다(답변 위에 접이식으로 표시).
        setMessages((m) => [...m, { role: 'assistant', data, steps: runSteps.slice() }])
        setSteps([])
        setLoading(false)
        abortRef.current = null
      },
      onError: (e) => {
        setError(streamErrorMessage(e))
        setSteps([])
        setLoading(false)
        abortRef.current = null
      },
    })

    // Safety net: if the stream ended without done/error (rare), clear loading.
    if (abortRef.current === controller) {
      setLoading(false)
      setSteps([])
      abortRef.current = null
    }
  }

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      send()
    }
  }

  if (!threadId) {
    return (
      <main className="chat-panel empty">
        <div className="empty-chat muted">
          왼쪽에서 대화를 선택하거나 새 대화를 생성하세요.
        </div>
      </main>
    )
  }

  return (
    <main className="chat-panel">
      <MessageList messages={messages} loading={false} />

      {loading && (
        <div className="message assistant">
          <div className="msg-role">어시스턴트</div>
          <ProgressSteps steps={steps} />
        </div>
      )}

      {error && <div className="error-line">{error}</div>}

      <div className="composer">
        <div className="composer-row">
          <label className="section-label" htmlFor="query-input">
            질의입력란
          </label>
          <label className="debug-toggle">
            <input
              type="checkbox"
              checked={debug}
              onChange={(e) => setDebug(e.target.checked)}
            />
            debug
          </label>
        </div>
        <textarea
          id="query-input"
          className="query-input"
          placeholder="설비 상태나 증상을 자연어로 입력하세요. (Ctrl+Enter 전송)"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={loading}
          rows={3}
        />

        <FeatureInputs values={features} onChange={setFeatures} disabled={loading} />

        <div className="composer-actions">
          <button
            className="btn btn-primary"
            onClick={send}
            disabled={loading || !message.trim()}
          >
            {loading ? '전송 중…' : '전송'}
          </button>
        </div>
      </div>
    </main>
  )
}
