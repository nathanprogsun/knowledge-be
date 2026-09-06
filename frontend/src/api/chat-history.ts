import { get, put, post } from '@/utils/request'
import type { components } from '@/api/__generated__/schema'

type Schema = components['schemas']

// Tenant KV payload is not a named OpenAPI schema. Keep the fields the
// settings page writes; knowledge_base_id is server-managed.
export type ChatHistoryConfig = {
  enabled: boolean
  embedding_model_id: string
  knowledge_base_id?: string
}

export type ChatHistoryKBStats = Schema['ChatHistoryStats']
export type MessageSearchGroupItem = Schema['MessageSearchHit']
export type MessageSearchResult = Schema['MessageSearchResponse']
export type MessageSearchRequest = Partial<Omit<Schema['SearchMessagesRequest'], 'query'>> & {
  query: string
  mode?: 'keyword' | 'vector' | 'hybrid'
}

export function getTenantChatHistoryConfig() {
  return get<{ data?: ChatHistoryConfig }>('/api/v1/tenants/kv/chat-history-config')
}

export function updateTenantChatHistoryConfig(config: ChatHistoryConfig) {
  return put('/api/v1/tenants/kv/chat-history-config', config)
}

export function getChatHistoryKBStats() {
  return get<Schema['ChatHistoryStatsEnvelope']>('/api/v1/messages/chat-history-stats')
}

export function searchMessages(data: MessageSearchRequest) {
  return post<Schema['SearchMessagesEnvelope']>('/api/v1/messages/search', data)
}
