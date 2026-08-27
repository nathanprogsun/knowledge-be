import { post, get, put } from '@/utils/request'
import i18n from '@/i18n'
import type { components } from '@/api/__generated__/schema'

const t = (key: string) => i18n.global.t(key)

// Wire types are generated from the FastAPI OpenAPI schema (make openapi).
// Never hand-write backend response shapes here — regenerate instead.
type Schema = components['schemas']

export type LoginRequest = Schema['LoginRequest']
export type LoginResponse = Schema['LoginResponse']
export type OIDCAuthURLResponse = Schema['OIDCAuthorizeURLResponse']
export type OIDCConfigResponse = Schema['OIDCMetaConfig']
export type RegisterRequest = Schema['RegisterRequest']
export type RegisterResponse = Schema['RegisterResponse']
export type AuthConfigResponse = Schema['AuthConfigResponse']
export type UserPreferences = Schema['UserPreferences']
export type RegisterByInviteRequest = Schema['RegisterByInviteRequest']

// TenantInfo is a frontend view-model: the store normalizes id to string
// (tenant ids travel as strings in X-Tenant-ID headers and localStorage)
// and fabricates owner_id (user.id fallback) for ownership checks, since
// the wire Tenant no longer carries it. Derived from the generated shape
// so new wire fields flow through automatically.
export type TenantInfo = Partial<Omit<Schema['Tenant'], 'id'>> & {
  id: string
  name: string
  owner_id?: string
}

// tenantOwnerId reads the legacy owner_id field off a wire Tenant if the
// backend still emits it, falling back to the current user id. The field
// is absent from the Python wire contract, hence the loose read.
export function tenantOwnerId(tenant: unknown, fallback: string): string {
  const ownerId = (tenant as { owner_id?: string | null })?.owner_id
  return ownerId || fallback
}
export type ModelInfo = Schema['Model']
export type MembershipInfo = Schema['Membership']
export type InviteLookup = Schema['InviteLookup']
export type InviteLookupResponse = Schema['InvitationLookupResponse']

// KnowledgeBaseInfo extends the generated wire shape with legacy read
// aliases still emitted by some tool-result paths (document_count).
export type KnowledgeBaseInfo = Schema['KnowledgeBase'] & {
  document_count?: number
}

// AuthCapabilities is the /auth/me capabilities sub-object.
export type AuthCapabilities = Schema['MeCapabilities']

// UserInfo is a frontend view-model, not a wire type: tenant_id is
// normalized to string and missing timestamps are backfilled by
// userInfoFromApi below. Do not replace with the generated AuthUser.
export interface UserInfo {
  id: string
  username: string
  email: string
  avatar?: string
  tenant_id: string
  can_access_all_tenants?: boolean
  preferences?: UserPreferences
  is_system_admin?: boolean
  created_at: string
  updated_at: string
}

/**
 * 把后端返回的 user JSON 规范化成前端 UserInfo。
 *
 * 历史上有 4 处独立的 setUser 调用（Login、autoSetup、token rehydrate、
 * /auth/me 主动 refresh）各自手写字段白名单，每加一个 user 字段都要在
 * 4 处同步——否则该字段就被悄悄过滤掉。is_system_admin 上线时就因为
 * 漏拷一处而看不到「系统管理」入口；这个工厂存在的目的就是杜绝同类
 * 漏拷再发生。**新增 user 字段请只改这里**。
 *
 * fallbackTenantId 是 tenant_id 缺失时的兜底来源——
 *   - autoSetup 响应顶层有 tenant.id，但 user 对象上没有 tenant_id
 *   - /auth/me 偶发只返回 user 不带 tenant 时也走兜底
 * 调用方按需传入；不传则保持空字符串（与历史行为一致）。
 *
 * 字段读取统一走 `=== true` 而不是 `|| false`，对偶发非 boolean
 * 类型（后端某天传 1/0 或字符串）做严格收敛，避免把 truthy 字符串
 * 误判为权限通过。
 */
export function userInfoFromApi(
  u: any,
  fallbackTenantId?: string | number | null,
): UserInfo {
  const rawTenantId =
    u?.tenant_id !== undefined && u?.tenant_id !== null && u.tenant_id !== ''
      ? u.tenant_id
      : fallbackTenantId ?? ''
  const tid = Number(rawTenantId) > 0 ? rawTenantId : ''
  return {
    id: u?.id || '',
    username: u?.username || '',
    email: u?.email || '',
    avatar: u?.avatar,
    tenant_id: String(tid) || '',
    can_access_all_tenants: u?.can_access_all_tenants === true,
    is_system_admin: u?.is_system_admin === true,
    preferences: u?.preferences,
    created_at: u?.created_at || new Date().toISOString(),
    updated_at: u?.updated_at || new Date().toISOString(),
  }
}

// Error-return helper: api wrappers resolve a minimal failure object on
// transport errors instead of throwing, so callers always get a shape
// with success=false. The cast is contained here and never escapes.
function failure<T extends { success: boolean }>(message: string): T {
  return { success: false, message } as unknown as T
}

// Success envelopes do not declare `message` in the OpenAPI schema, but
// consumers surface `response.message` on failure paths. WithMessage
// adds the optional field without widening the generated shape.
export type WithMessage<T> = T & { message?: string }

/**
 * 用户登录
 */
export async function login(data: LoginRequest): Promise<LoginResponse> {
  try {
    const response = await post('/api/v1/auth/login', data)
    return response as unknown as LoginResponse
  } catch (error: any) {
    return failure<LoginResponse>(error.message || t('error.auth.loginFailed'))
  }
}

/**
 * 获取 OIDC 登录跳转地址
 */
export async function getOIDCAuthorizationURL(redirectURI: string): Promise<WithMessage<OIDCAuthURLResponse>> {
  try {
    const response = await get(`/api/v1/auth/oidc/url?redirect_uri=${encodeURIComponent(redirectURI)}`)
    return response as unknown as WithMessage<OIDCAuthURLResponse>
  } catch (error: any) {
    return failure<WithMessage<OIDCAuthURLResponse>>(error.message || t('error.auth.loginFailed'))
  }
}

/**
 * 获取 OIDC 登录配置
 */
export async function getOIDCConfig(): Promise<OIDCConfigResponse> {
  try {
    const response = await get('/api/v1/auth/oidc/config')
    return response as unknown as OIDCConfigResponse
  } catch (error: any) {
    return failure<OIDCConfigResponse>(error.message || t('error.auth.loginFailed'))
  }
}

/**
 * 获取认证配置（仅返回前端渲染需要的公开字段，例如注册模式）。
 *
 * 后端通过 `auth.registration_mode` 控制是否允许自助注册：
 *   - "self_serve"  保留现有自助注册入口（默认）
 *   - "invite_only" 关闭注册，要求管理员邀请
 *
 * 失败时回落到 self_serve，避免接口异常导致注册入口直接消失。
 */
export async function getAuthConfig(): Promise<AuthConfigResponse> {
  try {
    const response = await get('/api/v1/auth/config')
    return response as unknown as AuthConfigResponse
  } catch {
    return { success: false, registration_mode: 'self_serve' }
  }
}

/**
 * 用户注册
 */
export async function register(data: RegisterRequest): Promise<RegisterResponse> {
  try {
    const response = await post('/api/v1/auth/register', data)
    return response as unknown as RegisterResponse
  } catch (error: any) {
    return failure<RegisterResponse>(error.message || t('error.auth.registerFailed'))
  }
}

/**
 * Lite 版自动初始化（创建默认用户/空间 + 签发令牌）
 */
export async function autoSetup(): Promise<LoginResponse> {
  try {
    const response = await post('/api/v1/auth/auto-setup', {})
    return response as unknown as LoginResponse
  } catch (error: any) {
    return failure<LoginResponse>(error.message || 'Auto-setup unavailable')
  }
}

/**
 * 获取当前用户信息
 */
export async function getCurrentUser(): Promise<WithMessage<Schema['MeResponse']>> {
  try {
    const response = await get('/api/v1/auth/me')
    return response as unknown as WithMessage<Schema['MeResponse']>
  } catch (error: any) {
    return failure<WithMessage<Schema['MeResponse']>>(error.message || t('error.auth.getUserFailed'))
  }
}

/**
 * 更新当前用户的偏好设置（PATCH 语义：只发要改的字段，后端只覆盖发了的 key，
 * 其它 key 保持不变）。后端会返回更新后的完整 preferences 对象。
 */
export async function updateMyPreferences(
  patch: Partial<UserPreferences>,
): Promise<WithMessage<Schema['UpdatePreferencesResponse']>> {
  try {
    const response = await put('/api/v1/auth/me/preferences', patch)
    return response as unknown as WithMessage<Schema['UpdatePreferencesResponse']>
  } catch (error: any) {
    return failure<WithMessage<Schema['UpdatePreferencesResponse']>>(
      error.message || t('error.auth.updatePreferencesFailed'),
    )
  }
}

/**
 * 获取当前空间信息
 */
export async function getCurrentTenant(): Promise<WithMessage<Schema['TenantEnvelope']>> {
  try {
    const response = await get('/api/v1/auth/tenant')
    return response as unknown as WithMessage<Schema['TenantEnvelope']>
  } catch (error: any) {
    return failure<WithMessage<Schema['TenantEnvelope']>>(error.message || t('error.auth.getTenantFailed'))
  }
}

/**
 * 刷新Token
 */
export async function refreshToken(refreshToken: string): Promise<{ success: boolean; data?: { token: string; refreshToken: string }; message?: string }> {
  try {
    const response = (await post('/api/v1/auth/refresh', { refreshToken })) as unknown as Schema['RefreshTokenResponse']
    if (response && response.success) {
      if (response.access_token || response.refresh_token) {
        return {
          success: true,
          data: {
            token: response.access_token,
            refreshToken: response.refresh_token,
          }
        }
      }
    }

    // 其他情况直接返回原始消息
    return {
      success: false,
      message: response?.message || t('error.auth.refreshTokenFailed')
    }
  } catch (error: any) {
    return {
      success: false,
      message: error.message || t('error.auth.refreshTokenFailed')
    }
  }
}

/**
 * 用户登出
 */
export async function logout(): Promise<{ success: boolean; message?: string }> {
  try {
    await post('/api/v1/auth/logout', {})
    return {
      success: true
    }
  } catch (error: any) {
    return {
      success: false,
      message: error.message || t('error.auth.logoutFailed')
    }
  }
}

/**
 * 验证Token有效性
 */
export async function validateToken(): Promise<{ success: boolean; valid?: boolean; message?: string }> {
  try {
    const response = await get('/api/v1/auth/validate')
    return response as unknown as { success: boolean; valid?: boolean; message?: string }
  } catch (error: any) {
    return {
      success: false,
      valid: false,
      message: error.message || t('error.auth.validateTokenFailed')
    }
  }
}




/**
 * Resolve a share-link token (no auth) into the context the
 * registration page needs (tenant name, role, expiry). Returns 410
 * when the link is invalid / revoked / expired.
 *
 * Uses POST + body (rather than GET + path) so the plaintext token
 * never appears in access logs, browser history, or tracing spans.
 */
export async function getInvitationByToken(token: string): Promise<InviteLookupResponse> {
  try {
    const response = await post(`/api/v1/auth/invitations/lookup`, { token })
    return response as unknown as InviteLookupResponse
  } catch (error: any) {
    return failure<InviteLookupResponse>(error.message || '')
  }
}

/**
 * Complete registration via a share-link token. The invitee supplies
 * their own email — the token is the authorisation, not an identity
 * lock.
 */
export async function registerByInvite(data: RegisterByInviteRequest): Promise<LoginResponse> {
  try {
    const response = await post('/api/v1/auth/register-by-invite', data)
    return response as unknown as LoginResponse
  } catch (error: any) {
    return failure<LoginResponse>(error.message || t('error.auth.registerFailed'))
  }
}
