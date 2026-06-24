import React, { useEffect, useRef } from 'react'
import AssistantAnswer from './AssistantAnswer.jsx'
import ProgressSteps from './ProgressSteps.jsx'

// messages: [{ role: 'user'|'assistant', content?: string, data?: chatResponse }]
export default function MessageList({ messages, loading }) {
  const endRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  return (
    <div className="message-list">
      {messages.length === 0 && !loading && (
        <div className="empty-chat muted">
          질문을 입력해 진단을 시작하세요.
        </div>
      )}
      {messages.map((m, i) => (
        <div key={i} className={`message ${m.role}`}>
          <div className="msg-role">{m.role === 'user' ? '사용자' : '어시스턴트'}</div>
          {m.role === 'assistant' && m.data ? (
            <>
              {m.steps && m.steps.length > 0 && (
                <ProgressSteps steps={m.steps} completed />
              )}
              <AssistantAnswer data={m.data} />
            </>
          ) : (
            <div className="answer-text">{m.content}</div>
          )}
        </div>
      ))}
      {loading && (
        <div className="message assistant">
          <div className="msg-role">어시스턴트</div>
          <div className="loading-row">
            <span className="spinner" /> 답변을 생성하는 중입니다… (수 초 소요)
          </div>
        </div>
      )}
      <div ref={endRef} />
    </div>
  )
}
