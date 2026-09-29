import { useState, type FormEvent } from 'react'
import { AudioLines, Copy, Cpu, Headphones, Languages, Send, Sparkles } from 'lucide-react'
import { VoiceInput, type VoiceState } from './VoiceInput'
import { apiFetch } from '../lib/api'

type Stage = 'idle' | 'voice' | 'speech-to-text' | 'transcribed' | 'llm' | 'complete' | 'failed'
type ProviderTrace = { provider: string; status: 'success' | 'failed' | 'not_configured'; error_type?: string; http_status?: string }
type ProviderResult = { provider: string; model: string; base_url?: string; elapsed_seconds?: number; fallback_used: boolean; fallback_reason?: string }
type GenerationSettings = { purpose: string; audience: string; depth: string }
type LocalLLMTestProps = {
  documentReady?: boolean
  documentId?: string
  onPlayAudio?: (text: string, language?: string) => void
  onOpenAudio?: (text: string, language?: string) => void
}
const providers = [
  { id: 'openrouter', title: 'OpenRouter' },
  { id: 'openai-direct', title: 'Direct OpenAI' },
  { id: 'ollama-local', title: 'Ollama Local' },
]
const purposeOptions = [
  ['general', 'General Summary'], ['study', 'Study / Learning'], ['business', 'Business'],
  ['technical', 'Technical'], ['research', 'Research'], ['presentation', 'Presentation'],
  ['key_findings', 'Key Findings'], ['custom', 'Custom'],
]
const audienceOptions = [
  ['general', 'General Audience'], ['student', 'Student'], ['developer_technical', 'Developer / Technical'],
  ['business_professional', 'Business Professional'], ['executive', 'Executive'], ['researcher', 'Researcher'], ['expert', 'Expert'],
]
const depthOptions = [
  ['quick', 'Quick', 'Short overview with only the most important points.'],
  ['standard', 'Standard', 'Balanced summary covering the main information.'],
  ['deep', 'Deep', 'Detailed summary covering important context, evidence, relationships, and supporting details.'],
]
const translationLanguages = [['en','English'],['hi','Hindi'],['te','Telugu'],['ta','Tamil'],['kn','Kannada'],['ml','Malayalam'],['bn','Bengali'],['mr','Marathi'],['gu','Gujarati'],['pa','Punjabi'],['ur','Urdu']]

const stageNames: Record<Stage, string> = {
  idle: 'Ready for a prompt',
  voice: 'Voice - recording',
  'speech-to-text': 'Speech-to-Text - transcribing locally',
  transcribed: 'Transcription ready - review before sending',
  llm: 'Testing AI provider...',
  complete: 'Complete',
  failed: 'Provider test failed',
}

export function LocalLLMTest({ documentReady = false, documentId, onPlayAudio, onOpenAudio }: LocalLLMTestProps) {
  const [prompt, setPrompt] = useState('')
  const [purpose, setPurpose] = useState('general')
  const [audience, setAudience] = useState('general')
  const [depth, setDepth] = useState('standard')
  const [targetLanguage, setTargetLanguage] = useState('te')
  const [language, setLanguage] = useState('en')
  const [voiceState, setVoiceState] = useState<VoiceState>('idle')
  const [stage, setStage] = useState<Stage>('idle')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [response, setResponse] = useState('')
  const [model, setModel] = useState('')
  const [providerResult, setProviderResult] = useState<ProviderResult | null>(null)
  const [providerTrace, setProviderTrace] = useState<ProviderTrace[]>([])
  const [generationSettings, setGenerationSettings] = useState<GenerationSettings | null>(null)
  const [historyPersisted, setHistoryPersisted] = useState(false)
  const [responseLanguage, setResponseLanguage] = useState<string | null>(null)

  async function sendPrompt(event: FormEvent) {
    event.preventDefault()
    if (!prompt.trim() || busy) return
    setBusy(true)
    setError('')
    setResponse('')
    setProviderResult(null)
    setProviderTrace([])
    setGenerationSettings(null)
    setHistoryPersisted(false)
    setResponseLanguage(null)
    if (stage === 'transcribed') console.info('[voice] transcription submitted to the configured LLM provider chain prompt_characters=%d', prompt.trim().length)
    setStage('llm')
    try {
      const result = await apiFetch('/llm/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt.trim(), purpose, audience, depth, document_id: documentId || undefined }),
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
        base_url: typeof body?.provider_metadata?.base_url === 'string' ? body.provider_metadata.base_url : undefined,
        elapsed_seconds: typeof body?.provider_metadata?.elapsed_seconds === 'number' ? body.provider_metadata.elapsed_seconds : undefined,
        fallback_used: Boolean(body.fallback_used),
        fallback_reason: typeof body.fallback_reason === 'string' ? body.fallback_reason : undefined,
      }
      setProviderTrace(Array.isArray(body?.provider_trace) ? body.provider_trace : [])
      setProviderResult(provider)
      setHistoryPersisted(body?.history_persisted === true)
      setResponseLanguage(typeof body?.translation_language === 'string' ? body.translation_language : null)
      setGenerationSettings({
        purpose: body?.generation_settings?.purpose || purpose,
        audience: body?.generation_settings?.audience || audience,
        depth: body?.generation_settings?.depth || depth,
      })
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

  async function translateResponse() {
    if (!response.trim() || busy) return
    setBusy(true)
    setError('')
    setProviderTrace([])
    setStage('llm')
    try {
      const result = await apiFetch('/llm/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: response, purpose, audience, depth, document_id: documentId || undefined, translate_to: targetLanguage }),
      })
      const body = await result.json().catch(() => ({}))
      if (!result.ok) {
        setProviderTrace(Array.isArray(body?.provider_trace) ? body.provider_trace : [])
        setError(typeof body?.error === 'string' ? body.error : 'Translation is unavailable from the configured providers.')
        setStage('failed')
        return
      }
      const translated = typeof body?.response === 'string' ? body.response.trim() : ''
      if (!translated || typeof body?.provider !== 'string' || typeof body?.model !== 'string') throw new Error('The provider chain returned an incomplete translation.')
      setResponse(translated)
      setResponseLanguage(targetLanguage)
      setProviderResult({ provider: body.provider, model: body.model, base_url: body.provider_metadata?.base_url, elapsed_seconds: body.provider_metadata?.elapsed_seconds, fallback_used: Boolean(body.fallback_used), fallback_reason: typeof body.fallback_reason === 'string' ? body.fallback_reason : undefined })
      setProviderTrace(Array.isArray(body?.provider_trace) ? body.provider_trace : [])
      setGenerationSettings(body?.generation_settings || { purpose, audience, depth })
      setHistoryPersisted(body?.history_persisted === true)
      const title = providers.find((item) => item.id === body.provider)?.title || body.provider
      setModel(`${title} - ${body.model}`)
      setStage('complete')
    } catch (cause) {
      setStage('failed')
      setError(cause instanceof Error ? cause.message : 'Translation failed: unable to reach the backend.')
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
        <section className="local-llm-settings" aria-labelledby="local-llm-settings-heading">
          <div className="local-llm-settings-heading"><h2 id="local-llm-settings-heading">Generation Settings</h2><p>These settings guide the backend prompt for every provider in the shared fallback chain.</p></div>
          <div className="local-llm-settings-grid">
            <label>Purpose<select value={purpose} onChange={(event) => setPurpose(event.target.value)} disabled={busy}>{purposeOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
            <label>Audience<select value={audience} onChange={(event) => setAudience(event.target.value)} disabled={busy}>{audienceOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
            <label>Summary Depth<select value={depth} onChange={(event) => setDepth(event.target.value)} disabled={busy}>{depthOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
          </div>
          <div className="local-llm-depth-hint">{depthOptions.find(([value]) => value === depth)?.[2]}</div>
        </section>

        <div className="local-llm-title-row">
          <div className="local-llm-icon"><Cpu size={18} /></div>
          <div><h2>Prompt</h2><p>Text and voice use the same centralized provider router.</p></div>
          <span className="local-llm-model"><span /> {model || 'OpenRouter -> OpenAI -> Ollama Local'}</span>
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
            {providerResult.provider === 'openai-direct' && <span><b>Base URL:</b> https://api.openai.com/v1</span>}
            {providerResult.elapsed_seconds !== undefined && <span><b>Timing:</b> {providerResult.elapsed_seconds.toFixed(2)}s</span>}
            {responseLanguage && <span><b>Response language:</b> {translationLanguages.find(([code]) => code === responseLanguage)?.[1] || responseLanguage}</span>}
            {generationSettings && <><span><b>Purpose:</b> {purposeOptions.find(([value]) => value === generationSettings.purpose)?.[1] || generationSettings.purpose}</span><span><b>Audience:</b> {audienceOptions.find(([value]) => value === generationSettings.audience)?.[1] || generationSettings.audience}</span><span><b>Depth:</b> {generationSettings.depth}</span></>}
            <span><b>Fallback used:</b> {providerResult.fallback_used ? 'Yes' : 'No'}</span>
            <span><b>History:</b> {historyPersisted ? 'Saved' : 'Unavailable'}</span>
            {providerResult.fallback_reason && <span><b>Fallback reason:</b> {providerResult.fallback_reason.replace(/_/g, ' ')}</span>}
          </div>}
          <div className="local-llm-provider-trace"><b>Provider chain</b>
            {providers.map((item) => {
              const attempt = providerTrace.find((entry) => entry.provider === item.id)
              const status = attempt?.status || 'not-attempted'
              const mark = status === 'success' ? 'OK' : status === 'failed' ? 'X' : '-'
              const label = status === 'not-attempted' ? 'Not attempted' : status === 'not_configured' ? 'Not configured' : status[0].toUpperCase() + status.slice(1)
              const diagnostic = [attempt?.http_status && attempt.http_status !== 'unknown' ? `HTTP ${attempt.http_status}` : '', attempt?.error_type || ''].filter(Boolean).join(' · ')
              return <div className={`local-llm-trace-row is-${status}`} key={item.id}><span aria-hidden="true">{mark}</span><span>{item.title}</span><small>{label}{diagnostic ? ` · ${diagnostic}` : ''}</small></div>
            })}
          </div>
          {response && <><div><span className="hero-kicker"><Sparkles size={13} /> GENERATED RESPONSE</span><span className="local-llm-response-model">{model}</span></div><p>{response}</p><div className="local-llm-result-actions">
            <button className="secondary-btn" type="button" onClick={() => onPlayAudio?.(response, responseLanguage || undefined)} disabled={!documentReady || !onPlayAudio || busy}><Headphones size={15}/> Play Audio</button>
            <button className="secondary-btn" type="button" onClick={() => onOpenAudio?.(response, responseLanguage || undefined)} disabled={!documentReady || !onOpenAudio || busy}><AudioLines size={15}/> Open Audio Briefing</button>
            <label className="local-llm-translate-target">Translate to<select value={targetLanguage} onChange={(event) => setTargetLanguage(event.target.value)} disabled={busy}>{translationLanguages.map(([code, name]) => <option key={code} value={code}>{name}</option>)}</select></label>
            <button className="secondary-btn" type="button" onClick={() => void translateResponse()} disabled={busy}><Languages size={15}/>{busy && stage === 'llm' ? ' Translating...' : ' Translate Summary'}</button>
            <button className="secondary-btn" type="button" onClick={() => void navigator.clipboard?.writeText(response)}><Copy size={15}/> Copy</button>
          </div><small className="local-llm-action-note">{documentReady ? 'Audio uses the selected document’s existing briefing controls. Translation uses Scribe’s existing translation helper and provider chain. Each result is saved to History.' : 'Translation and History use Scribe’s existing services. Audio requires a selected, processed document; standalone prompt audio is not supported.'}</small></>}
        </section>}
      </form>
    </div>
  )
}
