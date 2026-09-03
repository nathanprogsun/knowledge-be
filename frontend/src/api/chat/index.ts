import { get, post, put, del, postChat } from "../../utils/request";
import type { components } from '@/api/__generated__/schema'

type Schema = components['schemas']

export type CreateSessionRequest = Schema['CreateSessionRequest']
export type UpdateSessionRequest = Schema['UpdateSessionRequest']
export type Session = Schema['Session']
export type SessionEnvelope = Schema['SessionEnvelope']
export type SessionListEnvelope = Schema['SessionListEnvelope']
export type Message = Schema['Message']
export type MessageLoadEnvelope = Schema['MessageLoadEnvelope']
export type BatchDeleteSessionsRequest = Schema['BatchDeleteSessionsRequest']
export type CreateKnowledgeQARequest = Schema['CreateKnowledgeQARequest']

export async function createSessions(data: CreateSessionRequest = {}) {
  return post<SessionEnvelope>("/api/v1/sessions", data);
}

export async function getSessionsList(page: number, page_size: number, source?: string) {
  const params = new URLSearchParams({ page: String(page), page_size: String(page_size) });
  if (source) {
    params.set("source", source);
  }
  return get<SessionListEnvelope>(`/api/v1/sessions?${params.toString()}`);
}

export async function pinSession(session_id: string) {
  return post<SessionEnvelope>(`/api/v1/sessions/${session_id}/pin`, {});
}

export async function unpinSession(session_id: string) {
  return del<SessionEnvelope>(`/api/v1/sessions/${session_id}/pin`);
}

export async function generateSessionsTitle(session_id: string, data: { query?: string } = {}) {
  return post(`/api/v1/sessions/${session_id}/generate_title`, data);
}

export async function updateSession(session_id: string, data: UpdateSessionRequest) {
  return put<SessionEnvelope>(`/api/v1/sessions/${session_id}`, data);
}

export async function knowledgeChat(data: { session_id: string; query: string }) {
  return postChat(`/api/v1/knowledge-chat/${data.session_id}`, { query: data.query, channel: "web" });
}

export async function agentChat(data: {
  session_id: string;
  query: string;
  knowledge_base_ids?: string[];
  agent_enabled: boolean;
}) {
  const body: Pick<CreateKnowledgeQARequest, 'query' | 'knowledge_base_ids' | 'agent_enabled'> & { channel: string } = {
    query: data.query,
    knowledge_base_ids: data.knowledge_base_ids,
    agent_enabled: data.agent_enabled,
    channel: "web",
  };
  return postChat(`/api/v1/agent-chat/${data.session_id}`, body);
}

export async function getMessageList(data: { session_id: string; limit: number; created_at: string }) {
  if (data.created_at) {
    return get<MessageLoadEnvelope>(`/api/v1/messages/${data.session_id}/load?before_time=${encodeURIComponent(data.created_at)}&limit=${data.limit}`);
  }
  return get<MessageLoadEnvelope>(`/api/v1/messages/${data.session_id}/load?limit=${data.limit}`);
}

export async function delSession(session_id: string) {
  return del(`/api/v1/sessions/${session_id}`);
}

export async function batchDelSessions(ids: string[]) {
  const body: BatchDeleteSessionsRequest = { ids, delete_all: false };
  return del(`/api/v1/sessions/batch`, body);
}

export async function deleteAllSessions() {
  const body: BatchDeleteSessionsRequest = { delete_all: true };
  return del(`/api/v1/sessions/batch`, body);
}

export async function getSession(session_id: string) {
  return get<SessionEnvelope>(`/api/v1/sessions/${session_id}`);
}

export async function stopSession(session_id: string, message_id: string) {
  return post(`/api/v1/sessions/${session_id}/stop`, { message_id });
}

export async function clearSessionMessages(session_id: string) {
  return del(`/api/v1/sessions/${session_id}/messages`);
}
