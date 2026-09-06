import assert from 'node:assert/strict'
import test from 'node:test'

import {
  dropEmptyIncompleteAssistants,
  isEmptyIncompleteAssistant,
  resolveStreamAssistant,
  shouldKeepAssistantRow,
  type ChatMessage,
} from './chatStreamAssistant.ts'

test('resolveStreamAssistant prefers request id then the trailing incomplete row', () => {
  const incomplete: ChatMessage = {
    id: 'assistant-1',
    role: 'assistant',
    is_completed: false,
    content: '',
  }
  const messages: ChatMessage[] = [{ id: 'user-1', role: 'user' }, incomplete]
  assert.equal(resolveStreamAssistant(messages, 'req-1'), incomplete)
  incomplete.request_id = 'req-1'
  assert.equal(resolveStreamAssistant(messages, 'req-1'), incomplete)
})

test('dropEmptyIncompleteAssistants removes the leftover thinking shell', () => {
  const ghost: ChatMessage = {
    id: 'ghost',
    role: 'assistant',
    is_completed: false,
    content: '',
    isAgentMode: true,
  }
  const answer: ChatMessage = {
    id: 'answer',
    role: 'assistant',
    is_completed: true,
    content: 'done',
    isAgentMode: true,
    agentEventStream: [{ type: 'answer', content: 'done' }],
  }
  const messages: ChatMessage[] = [{ role: 'user', content: 'q' }, ghost, answer]
  dropEmptyIncompleteAssistants(messages, answer)
  assert.equal(messages.length, 2)
  assert.equal(messages[1], answer)
})

test('shouldKeepAssistantRow hides an empty incomplete sibling of a completed answer', () => {
  const ghost: ChatMessage = {
    role: 'assistant',
    is_completed: false,
    isAgentMode: true,
    content: '',
  }
  const answer: ChatMessage = {
    role: 'assistant',
    is_completed: true,
    isAgentMode: true,
    agentEventStream: [{ type: 'answer' }],
  }
  assert.equal(shouldKeepAssistantRow(ghost, [ghost, answer]), false)
  assert.equal(shouldKeepAssistantRow(answer, [ghost, answer]), true)
  assert.equal(isEmptyIncompleteAssistant(ghost), true)
})
