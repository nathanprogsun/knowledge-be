export type ChatMessage = Record<string, unknown>

export function resolveStreamAssistant(
  messages: ChatMessage[],
  dataId: string | undefined,
): ChatMessage | undefined {
  if (dataId) {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const item = messages[i]
      if (item.request_id === dataId || item.id === dataId) return item
    }
  }
  const last = messages[messages.length - 1]
  if (last?.role === 'assistant' && !last.is_completed) return last
  return undefined
}

export function isEmptyIncompleteAssistant(message: ChatMessage): boolean {
  if (message.role !== 'assistant' || message.is_completed) return false
  const stream = message.agentEventStream
  const refs = message.knowledge_references
  const hasStream = Array.isArray(stream) && stream.length > 0
  const hasRefs = Array.isArray(refs) && refs.length > 0
  return !message.content && !hasStream && !hasRefs
}

export function dropEmptyIncompleteAssistants(
  messages: ChatMessage[],
  keep: ChatMessage,
): void {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const item = messages[i]
    if (item === keep) continue
    if (isEmptyIncompleteAssistant(item)) messages.splice(i, 1)
  }
}

export function shouldKeepAssistantRow(
  session: ChatMessage,
  siblings: ChatMessage[],
): boolean {
  if (!session?.isAgentMode) return true
  if (!session.is_completed) {
    if (
      isEmptyIncompleteAssistant(session) &&
      siblings.some((row) => row !== session && row.role === 'assistant' && row.is_completed)
    ) {
      return false
    }
    return true
  }
  const stream = session.agentEventStream
  if (Array.isArray(stream) && stream.length > 0) return true
  if (Array.isArray(session.knowledge_references) && session.knowledge_references.length > 0) {
    return true
  }
  return false
}
