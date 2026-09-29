import React, { createContext, useContext, useEffect, useMemo, useState } from 'react'
import { apiFetch, responseError, setUnauthorizedHandler } from '../lib/api'

export type User = { id: number; name: string; email: string }
type AuthContextValue = {
  user: User | null
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (name: string, email: string, password: string) => Promise<void>
  logout: () => Promise<void>
}
const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    setUnauthorizedHandler(() => {
      setUser(null)
      if (window.location.pathname !== '/login' && window.location.pathname !== '/register') {
        window.location.replace('/login')
      }
    })
    apiFetch('/auth/me').then(async response => {
      if (response.ok) setUser(await response.json())
    }).catch(() => setUser(null)).finally(() => setIsLoading(false))
    return () => setUnauthorizedHandler(null)
  }, [])

  const value = useMemo<AuthContextValue>(() => ({
    user,
    isLoading,
    async login(email, password) {
      const response = await apiFetch('/auth/login', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      })
      if (!response.ok) throw new Error(await responseError(response, 'Invalid email or password'))
      setUser(await response.json())
    },
    async register(name, email, password) {
      const response = await apiFetch('/auth/register', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, password }),
      })
      if (!response.ok) throw new Error(await responseError(response, 'Unable to create account'))
    },
    async logout() {
      try { await apiFetch('/auth/logout', { method: 'POST' }) } catch { /* Local state still exits the protected UI. */ }
      setUser(null)
    },
  }), [user, isLoading])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const value = useContext(AuthContext)
  if (!value) throw new Error('useAuth must be used inside AuthProvider')
  return value
}
