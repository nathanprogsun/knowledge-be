import { get, post } from '@/utils/request'
import type { components } from '@/api/__generated__/schema'

type Schema = components['schemas']

export type MessageSuggestionItem = Schema['SuggestionQuestion'] & {
  source: 'model' | 'faq' | 'document' | 'wiki' | string
  knowledge_base_ids?: string[]
}

export type MessageSuggestionSet = Partial<Omit<Schema['SuggestionSet'], 'questions'>> & {
  id: string
  session_id: string
  assistant_message_id: string
  status: 'generating' | 'ready' | 'suppressed' | 'failed'
  allow_regenerate: boolean
  suppression_reason?: string
  questions: MessageSuggestionItem[]
  generated_at?: string
}

export function ensureMessageSuggestions(
  sessionId: string,
  messageId: string,
  regenerate = false,
  context?: { query?: string; answer?: string },
) {
  return post<Schema['SuggestionEnvelope']>(
    `/api/v1/sessions/${sessionId}/messages/${messageId}/suggestions`,
    {
      regenerate,
      query: context?.query || undefined,
      answer: context?.answer || undefined,
    },
  )
}

export function getMessageSuggestions(sessionId: string, messageId: string) {
  return get<Schema['SuggestionEnvelope']>(
    `/api/v1/sessions/${sessionId}/messages/${messageId}/suggestions`,
  )
}

export function recordMessageSuggestionEvent(
  sessionId: string,
  suggestionSetId: string,
  eventType: 'impression' | 'click' | 'dismiss',
  questionId = '',
) {
  const body: Schema['SuggestionEventRequest'] = {
    suggestion_set_id: suggestionSetId,
    question_id: questionId,
    event_type: eventType,
  }
  return post(
    `/api/v1/sessions/${sessionId}/suggestion-events`,
    body,
  )
}
