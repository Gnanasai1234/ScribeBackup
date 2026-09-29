import { useState, type FormEvent } from 'react'
import { Eye, EyeOff, Sparkles } from 'lucide-react'
import { useAuth } from './AuthContext'

function AuthFrame({ children }: { children: React.ReactNode }) {
  return <main className="auth-screen"><section className="auth-card"><a className="auth-brand" href="/"><span className="brand-mark"><Sparkles size={18}/></span><span>scribe<span className="brand-dot">.</span></span></a>{children}<p className="auth-foot">Source-grounded document intelligence</p></section></main>
}

export function LoginPage() {
  const { login } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [visible, setVisible] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  async function submit(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError('')
    try { await login(email, password); window.location.href = '/' }
    catch (err) { setError(err instanceof Error ? err.message : 'Unable to connect to server') }
    finally { setBusy(false) }
  }
  return <AuthFrame><div className="auth-heading"><span>WELCOME BACK</span><h1>Sign in to Scribe</h1><p>Your documents are waiting in your private workspace.</p></div><form className="auth-form" onSubmit={submit}>
    <label>Email<input type="email" autoComplete="email" required value={email} onChange={e=>setEmail(e.target.value)} placeholder="you@example.com"/></label>
    <label>Password<span className="auth-password"><input type={visible?'text':'password'} autoComplete="current-password" required value={password} onChange={e=>setPassword(e.target.value)} placeholder="Your password"/><button type="button" aria-label={visible?'Hide password':'Show password'} onClick={()=>setVisible(v=>!v)}>{visible?<EyeOff size={17}/>:<Eye size={17}/>}</button></span></label>
    {error&&<p className="auth-error" role="alert">{error}</p>}<button className="primary-btn auth-submit" disabled={busy}>{busy?<><span className="spinner"/> Signing in…</>:'Sign in'}</button>
  </form><p className="auth-switch">New to Scribe? <a href="/register">Create an account</a></p></AuthFrame>
}

export function RegisterPage() {
  const { register } = useAuth()
  const [name, setName] = useState(''); const [email, setEmail] = useState('')
  const [password, setPassword] = useState(''); const [confirm, setConfirm] = useState('')
  const [visible, setVisible] = useState(false); const [busy, setBusy] = useState(false); const [error, setError] = useState('')
  async function submit(event: FormEvent) {
    event.preventDefault(); setError('')
    if (password !== confirm) { setError('Passwords do not match'); return }
    if (password.length < 6) { setError('Password must be at least 6 characters'); return }
    setBusy(true)
    try { await register(name, email, password); window.location.href = '/login?registered=1' }
    catch (err) { setError(err instanceof Error ? err.message : 'Unable to create account') }
    finally { setBusy(false) }
  }
  return <AuthFrame><div className="auth-heading"><span>GET STARTED</span><h1>Create your account</h1><p>Keep your documents and analysis private to you.</p></div><form className="auth-form" onSubmit={submit}>
    <label>Name<input autoComplete="name" required maxLength={160} value={name} onChange={e=>setName(e.target.value)} placeholder="Your name"/></label>
    <label>Email<input type="email" autoComplete="email" required value={email} onChange={e=>setEmail(e.target.value)} placeholder="you@example.com"/></label>
    <label>Password<span className="auth-password"><input type={visible?'text':'password'} autoComplete="new-password" required minLength={6} value={password} onChange={e=>setPassword(e.target.value)} placeholder="At least 6 characters"/><button type="button" aria-label={visible?'Hide password':'Show password'} onClick={()=>setVisible(v=>!v)}>{visible?<EyeOff size={17}/>:<Eye size={17}/>}</button></span></label>
    <label>Confirm password<input type={visible?'text':'password'} autoComplete="new-password" required value={confirm} onChange={e=>setConfirm(e.target.value)} placeholder="Enter your password again"/></label>
    {error&&<p className="auth-error" role="alert">{error}</p>}<button className="primary-btn auth-submit" disabled={busy}>{busy?<><span className="spinner"/> Creating account…</>:'Create account'}</button>
  </form><p className="auth-switch">Already have an account? <a href="/login">Sign in</a></p></AuthFrame>
}
