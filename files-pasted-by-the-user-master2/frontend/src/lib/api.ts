export const API = 'http://127.0.0.1:8000/api'

let onUnauthorized: (() => void) | null = null

export function setUnauthorizedHandler(handler: (() => void) | null) {
  onUnauthorized = handler
}

export async function apiFetch(path: string, init: RequestInit = {}) {
  try {
    const response = await globalThis.fetch(path.startsWith('http') ? path : `${API}${path}`, {
      ...init,
      credentials: 'include',
    })
    if (response.status === 401) onUnauthorized?.()
    return response
  } catch {
    throw new Error('Unable to connect to server')
  }
}

export async function responseError(response: Response, fallback: string) {
  try {
    const body = await response.json()
    const message = typeof body?.detail === 'string' ? body.detail : ''
    if (message === 'Invalid email or password.') return 'Invalid email or password'
    if (message === 'Email already registered.') return 'Email already registered'
    if (message === 'Authentication required.') return 'Please log in to continue'
    if (message.includes('Password must be at least 6 characters')) return 'Password must be at least 6 characters'
    return fallback
  } catch {
    return fallback
  }
}
