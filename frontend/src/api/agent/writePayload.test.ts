import assert from 'node:assert/strict'
import test from 'node:test'

import { agentWritePayload } from './writePayload.ts'

test('strips identity fields the UpdateAgentRequest contract forbids', () => {
  const payload = agentWritePayload({
    name: '快速问答',
    description: 'RAG',
    avatar: '',
    config: { model_id: 'm1' },
    id: 'builtin-quick-answer',
    is_builtin: true,
    tenant_id: 1,
    created_at: '2026-01-01',
  } as never)

  assert.deepEqual(payload, {
    name: '快速问答',
    description: 'RAG',
    avatar: '',
    config: { model_id: 'm1' },
  })
})

test('drops a null name so builtin config-only saves stay name-free', () => {
  const payload = agentWritePayload({
    name: null,
    config: { model_id: 'm1' },
  })

  assert.deepEqual(payload, {
    config: { model_id: 'm1' },
  })
})
