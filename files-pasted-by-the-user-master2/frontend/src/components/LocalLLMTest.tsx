import { useState, type FormEvent } from 'react'
import { AudioLines, Cpu, Send, Sparkles } from 'lucide-react'
import { VoiceInput, type VoiceState } from './VoiceInput'
import { apiFetch } from '../lib/api'

type Stage = 'idle' | 'voice' | 'speech-to-text' | 'transcribed' | 'llm' | 'complete' | 'failed'
type ProviderTrace = { provider: string; status: 'success' | 'failed'; error_type?: string }
type ProviderResult = { provider: string; model: string; fallback_used: boolean; fallback_reason?: string }
const providers = [
  { id: 'openai-gateway', title: 'OpenAI Gateway' },
  { id: 'gemini', title: 'Gemini' },
  { id: 'ollama-local', title: 'Ollama Local' },
]

const stageNames: Record<Stage, string> = {
  idle: 'Ready for a prompt',
  voice: 'Voice - recording',
  'speech-to-text': 'Speech-to-Text - transcribing locally',
  transcribed: 'Transcription ready - review before sending',
  llm: 'Testing AI provider...',
  complete: 'Complete',
  failed: 'Provider test failed',
}

export function LocalLLMTest() {
  const [prompt, setPrompt] = useState('')
  const [language, setLanguage] = useState('en')
  const [voiceState, setVoiceState] = useState<VoiceState>('idle')
  const [stage, setStage] = useState<Stage>('idle')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [response, setResponse] = useState('')
  const [model, setModel] = useState('')
  const [providerResult, setProviderResult] = useState<ProviderResult | null>(null)
  const [providerTrace, setProviderTrace] = useState<ProviderTrace[]>([])

  async function sendPrompt(event: FormEvent) {
    event.preventDefault()
    if (!prompt.trim() || busy) return
    setBusy(true)
    setError('')
    setResponse('')
    setProviderResult(null)
    setProviderTrace([])
    if (stage === 'transcribed') console.info('[voice] transcription submitted to the configured LLM provider chain prompt_characters=%d', prompt.trim().length)
    setStage('llm')
    try {
      const result = await apiFetch('/llm/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt.trim() }),
      })
      const body = await result.json().catch(() => ({}))
      if (!result.ok) {
        setProviderTrace(Array.isArray(body?.provider_trace) ? body.provider_trace : [])
        setError(typeof body?.error === 'string' ? body.error : 'All configured AI providers are currently unavailable.')
        setStage('failed')
        return
      }
      const text = typeof body?.response === 'string' ? body.response.trim() : ''
      if (!text || typeof body?.provider !== 'string' || typeof body?.model !== 'string') throw new Error('The provider chain returned an incomplete response.')
      const provider: ProviderResult = {
        provider: body.provider,
        model: body.model,
        fallback_used: Boolean(body.fallback_used),
        fallback_reason: typeof body.fallback_reason === 'string' ? body.fallback_reason : undefined,
      }
      setProviderTrace(Array.isArray(body?.provider_trace) ? body.provider_trace : [])
      setProviderResult(provider)
      setResponse(text)
      const title = providers.find((item) => item.id === provider.provider)?.title || provider.provider
      console.info('[llm] response received characters=%d provider=%s model=%s', text.length, provider.provider, provider.model)
      setModel(`${title} - ${provider.model}`)
      setStage('complete')
    } catch (cause) {
      setStage('failed')
      setError(cause instanceof Error ? cause.message : 'LLM stage failed: unable to reach the backend.')
    } finally {
      setBusy(false)
    }
  }

  function handleVoiceState(next: VoiceState) {
    setVoiceState(next)
    if (next === 'listening' || next === 'processing_audio') setStage('voice')
    else if (next === 'transcribing') setStage('speech-to-text')
    else if (next === 'transcription_ready') setStage('transcribed')
  }

  return (
    <div className="local-llm-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow"><span /> LLM PROVIDER WORKSPACE</div>
          <h1>LLM Provider Test</h1>
          <p>Test text or voice prompts using the same AI provider fallback chain used throughout Scribe.</p>
        </div>
      </div>

      <form className="local-llm-card" onSubmit={sendPrompt}>
        <div className="local-llm-title-row">
          <div className="local-llm-icon"><Cpu size={18} /></div>
          <div><h2>Prompt</h2><p>Text and voice use the same centralized provider router.</p></div>
          <span className="local-llm-model"><span /> {model || 'OpenAI Gateway -> Gemini -> Ollama Local'}</span>
        </div>

        <label className="local-llm-label" htmlFor="local-llm-prompt">Type a prompt or review your transcription</label>
        <textarea
          id="local-llm-prompt"
          className="local-llm-textarea"
          rows={5}
          maxLength={4000}
          value={prompt}
          onChange={(event) => { setPrompt(event.target.value); setError(''); if (!response) setStage('idle') }}
          placeholder="For example: Summarize why the library extended its weekday hours."
        />
        <div className="local-llm-meta"><span>{prompt.length}/4000 characters</span><span>Voice transcription stays on this device.</span></div>

        <div className="local-llm-voice-row">
          <div className="local-llm-voice-copy">
            <div><AudioLines size={15} /><b>Voice input</b><span className={`local-llm-listening ${voiceState === 'listening' || voiceState === 'processing_audio' ? 'is-active' : ''}`}>
              {voiceState === 'listening' ? 'Listening... Recording...' : voiceState === 'processing_audio' ? 'Processing audio...' : voiceState === 'transcribing' ? 'Transcribing locally...' : voiceState === 'transcription_ready' ? 'Transcription ready' : voiceState === 'error' ? 'Voice input error' : 'Offline speech-to-text'}
            </span></div>
            <p>Record, review the recognized words, then use the transcription above.</p>
          </div>
          <div className="local-llm-voice-control">
            <label className="local-llm-language">Language
              <select value={language} onChange={(event) => setLanguage(event.target.value)} disabled={voiceState === 'listening' || voiceState === 'processing_audio' || voiceState === 'transcribing'}>
                <option value="en">English</option><option value="hi">Hindi</option><option value="te">Telugu</option><option value="ta">Tamil</option><option value="kn">Kannada</option><option value="auto">Auto detect</option>
              </select>
            </label>
            <VoiceInput
              language={language}
              transcriptionMode="local"
              className="local-llm-voice-input"
              disabled={busy}
              onStateChange={handleVoiceState}
              onError={(failedStage, message) => { setStage(failedStage); setError(message) }}
              onTranscript={(text) => { setPrompt(text); setError(''); setResponse(''); setStage('transcribed') }}
            />
          </div>
        </div>

        <div className="local-llm-actions">
          <span className="local-llm-stage" role="status" aria-live="polite">{stageNames[stage]}</span>
          <button className="primary-btn" type="submit" disabled={busy || !prompt.trim()}>
            {busy ? <><span className="spinner" /> Testing AI provider...</> : <><Send size={14} /> Send prompt</>}
          </button>
        </div>
        {error && <div className="local-llm-error" role="alert"><b>{stage === 'voice' ? 'Voice' : stage === 'speech-to-text' ? 'Speech-to-Text' : 'LLM'} stage failed</b><span>{error}</span></div>}
        {(response || providerTrace.length > 0) && <section className="local-llm-response" aria-live="polite">
          {providerResult && <div className="local-llm-provider-meta">
            <span><b>Provider:</b> {providers.find((item) => item.id === providerResult.provider)?.title || providerResult.provider}</span>
            <span><b>Model:</b> {providerResult.model}</span>
            <span><b>Fallback used:</b> {providerResult.fallback_used ? 'Yes' : 'No'}</span>
            {providerResult.fallback_reason && <span><b>Fallback reason:</b> {providerResult.fallback_reason.replace(/_/g, ' ')}</span>}
          </div>}
          <div className="local-llm-provider-trace"><b>Provider chain</b>
            {providers.map((item) => {
              const attempt = providerTrace.find((entry) => entry.provider === item.id)
              const status = attempt?.status || 'not-attempted'
              const mark = status === 'success' ? 'OK' : status === 'failed' ? 'X' : '-'
              const label = status === 'not-attempted' ? 'Not attempted' : status[0].toUpperCase() + status.slice(1)
              return <div className={`local-llm-trace-row is-${status}`} key={item.id}><span aria-hidden="true">{mark}</span><span>{item.title}</span><small>{label}</small></div>
            })}
          </div>
          {response && <><div><span className="hero-kicker"><Sparkles size={13} /> PROVIDER RESPONSE</span><span className="local-llm-response-model">{model}</span></div><p>{response}</p></>}
        </section>}
      </form>
    </div>
  )
}
