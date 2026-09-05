export type AgentWriteInput = {
  name?: string | null
  description?: string | null
  avatar?: string | null
  config?: Record<string, unknown> | null
}

export type AgentWritePayload = {
  name?: string
  description?: string | null
  avatar?: string | null
  config?: Record<string, unknown>
}

export function agentWritePayload(data: AgentWriteInput): AgentWritePayload {
  const payload: AgentWritePayload = {}
  if (data.name != null) payload.name = data.name
  if (data.description !== undefined) payload.description = data.description ?? null
  if (data.avatar !== undefined) payload.avatar = data.avatar ?? null
  if (data.config != null) payload.config = data.config
  return payload
}
