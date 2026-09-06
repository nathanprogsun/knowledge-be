import { get, post, put, del } from "../../utils/request";
import type { components } from '@/api/__generated__/schema';

type Schema = components['schemas'];

type NullToOptional<T> = {
  [K in keyof T]: null extends T[K] ? Exclude<T[K], null> | undefined : T[K];
};

export type AgentType = 'rag-qa' | 'wiki-qa' | 'hybrid-rag-wiki' | 'data-analysis' | 'custom';

export type QuestionSuggestionConfig = Schema['AgentQuestionSuggestions'] & {
  starters: Schema['AgentQuestionSuggestionsStarters'] & {
    enabled: boolean;
    mode: 'curated' | 'knowledge' | 'hybrid';
    items: string[];
    count: number;
  };
  follow_ups: Schema['AgentQuestionSuggestionsFollowUps'] & {
    enabled: boolean;
    mode: 'generated' | 'knowledge' | 'hybrid';
    categories: Array<'clarify' | 'deepen' | 'action'>;
  };
};

export type CustomAgentConfig = Partial<NullToOptional<Omit<
  Schema['AgentConfig'],
  'question_suggestions' | 'agent_mode' | 'agent_type' | 'kb_selection_mode' | 'mcp_selection_mode' | 'skills_selection_mode' | 'fallback_strategy'
>>> & {
  agent_mode?: 'quick-answer' | 'smart-reasoning';
  agent_type?: AgentType;
  kb_selection_mode?: 'all' | 'selected' | 'none';
  mcp_selection_mode?: 'all' | 'selected' | 'none';
  skills_selection_mode?: 'all' | 'selected' | 'none';
  fallback_strategy?: 'fixed' | 'model';
  reflection_enabled?: boolean;
  welcome_message?: string;
  question_suggestions?: QuestionSuggestionConfig;
};

export type CustomAgent = Partial<NullToOptional<Omit<Schema['Agent'], 'id' | 'name' | 'is_builtin' | 'config' | 'deleted_at'>>> & {
  id: string;
  name: string;
  is_builtin: boolean;
  config: CustomAgentConfig;
  creator_name?: string;
};

export type CreateAgentRequest = Omit<Schema['CreateAgentRequest'], 'config'> & {
  config?: CustomAgentConfig;
};

export type UpdateAgentRequest = Omit<Schema['UpdateAgentRequest'], 'config'> & {
  config?: CustomAgentConfig;
};

export const BUILTIN_QUICK_ANSWER_ID = 'builtin-quick-answer';
export const BUILTIN_SMART_REASONING_ID = 'builtin-smart-reasoning';

export const AGENT_MODE_QUICK_ANSWER = 'quick-answer';
export const AGENT_MODE_SMART_REASONING = 'smart-reasoning';

export const BUILTIN_AGENT_NORMAL_ID = BUILTIN_QUICK_ANSWER_ID;
export const BUILTIN_AGENT_AGENT_ID = BUILTIN_SMART_REASONING_ID;

export function listAgents(params?: {
  creator?: 'all' | 'mine' | 'others';
}) {
  const qs = params?.creator && params.creator !== 'all' ? `?creator=${params.creator}` : '';
  return get<{ data: CustomAgent[]; disabled_own_agent_ids?: string[] }>(`/api/v1/agents${qs}`);
}

export function getAgentById(id: string) {
  return get<{ data: CustomAgent }>(`/api/v1/agents/${id}`);
}

export function createAgent(data: CreateAgentRequest) {
  return post<{ data: CustomAgent }>('/api/v1/agents', data);
}

export function updateAgent(id: string, data: UpdateAgentRequest) {
  return put<{ data: CustomAgent }>(`/api/v1/agents/${id}`, data);
}

export function deleteAgent(id: string) {
  return del<{ success: boolean }>(`/api/v1/agents/${id}`);
}

export function copyAgent(id: string) {
  return post<{ data: CustomAgent }>(`/api/v1/agents/${id}/copy`);
}

export function isBuiltinAgent(agentId: string): boolean {
  return agentId.startsWith('builtin-');
}

export type PlaceholderDefinition = {
  name: string;
  label: string;
  description: string;
};

export type PlaceholdersResponse = {
  [K in keyof Schema['AgentPlaceholderGroup']]: PlaceholderDefinition[];
};

export function getPlaceholders() {
  return get<{ data: PlaceholdersResponse }>('/api/v1/agents/placeholders');
}

export type AgentTypeKBFilter = {
  any_of?: string[];
  all_of?: string[];
  none_of?: string[];
};

export type KBCapabilities = {
  vector: boolean;
  keyword: boolean;
  wiki: boolean;
  graph: boolean;
  faq: boolean;
};

export type AgentTypePresetConfig = Partial<NullToOptional<Pick<
  Schema['AgentConfig'],
  | 'system_prompt_id'
  | 'temperature'
  | 'max_iterations'
  | 'allowed_tools'
  | 'retain_retrieval_history'
  | 'faq_priority_enabled'
  | 'web_search_enabled'
  | 'supported_file_types'
>>> & {
  kb_selection_mode?: 'all' | 'selected' | 'none';
};

export type AgentTypePresetI18n = {
  label: string;
  description: string;
};

export type AgentTypePreset = {
  id: AgentType;
  i18n: Record<string, AgentTypePresetI18n>;
  config?: AgentTypePresetConfig;
  kb_filter?: AgentTypeKBFilter;
};

export function getAgentTypePresets() {
  return get<{ data: AgentTypePreset[] }>('/api/v1/agents/type-presets');
}

export type IMChannel = Partial<Omit<Schema['IMChannelRecord'], 'id' | 'agent_id' | 'name' | 'enabled' | 'platform' | 'mode' | 'output_mode'>> & {
  id: string;
  agent_id: string;
  platform: 'wecom' | 'feishu' | 'lark' | 'slack' | 'telegram' | 'dingtalk' | 'mattermost' | 'wechat' | 'qqbot' | 'yunzhijia';
  name: string;
  enabled: boolean;
  mode: 'webhook' | 'websocket' | 'longpoll';
  output_mode: 'stream' | 'full';
  session_mode?: 'user' | 'thread';
  credentials: Record<string, unknown>;
};

export function listIMChannels(agentId: string) {
  return get<{ data: IMChannel[] }>(`/api/v1/agents/${agentId}/im-channels`);
}

export type IMChannelOverview = Schema['IMChannelRecord'] & {
  agent_name: string;
  platform: IMChannel['platform'];
  mode: IMChannel['mode'];
  output_mode: IMChannel['output_mode'];
  session_mode?: IMChannel['session_mode'];
};

export function listAllIMChannels() {
  return get<{ data: IMChannelOverview[] }>('/api/v1/im-channels');
}

export function createIMChannel(agentId: string, data: Partial<IMChannel>) {
  return post<{ data: IMChannel }>(`/api/v1/agents/${agentId}/im-channels`, data);
}

export function updateIMChannel(id: string, data: Partial<IMChannel>) {
  return put<{ data: IMChannel }>(`/api/v1/im-channels/${id}`, data);
}

export function deleteIMChannel(id: string) {
  return del<{ success: boolean }>(`/api/v1/im-channels/${id}`);
}

export function toggleIMChannel(id: string) {
  return post<{ data: IMChannel }>(`/api/v1/im-channels/${id}/toggle`);
}

export type SuggestedQuestion = Schema['SuggestedQuestion'];

export function getSuggestedQuestions(
  agentId: string,
  params?: {
    knowledge_base_ids?: string[];
    knowledge_ids?: string[];
    tag_scopes?: Array<{ knowledge_base_id: string; tag_ids: string[] }>;
    limit?: number;
  }
) {
  const query = new URLSearchParams();
  if (params?.knowledge_base_ids?.length) query.set('knowledge_base_ids', params.knowledge_base_ids.join(','));
  if (params?.knowledge_ids?.length) query.set('knowledge_ids', params.knowledge_ids.join(','));
  if (params?.tag_scopes?.length) query.set('tag_scopes', JSON.stringify(params.tag_scopes));
  if (params?.limit) query.set('limit', String(params.limit));
  const qs = query.toString();
  return get<Schema['SuggestedQuestionsEnvelope']>(`/api/v1/agents/${agentId}/suggested-questions${qs ? '?' + qs : ''}`);
}

export type WeChatQRCodeResult = {
  qrcode_url: string;
  qrcode: string;
};

export type WeChatQRCodeStatus = {
  status: 'wait' | 'scaned' | 'confirmed' | 'expired';
  credentials?: {
    bot_token: string;
    ilink_bot_id: string;
    ilink_user_id: string;
  };
  baseurl?: string;
};

export function getWeChatQRCode() {
  return post<{ data: WeChatQRCodeResult }>('/api/v1/wechat/qrcode');
}

export function pollWeChatQRCodeStatus(qrcode: string) {
  return post<{ data: WeChatQRCodeStatus }>('/api/v1/wechat/qrcode/status', { qrcode });
}
