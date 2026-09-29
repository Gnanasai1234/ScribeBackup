import { useState, type FormEvent } from 'react'
import { AudioLines, Cpu, Send, Sparkles } from 'lucide-react'
import { VoiceInput, type VoiceState } from './VoiceInput'
import { apiFetch } from '../lib/api'

type Stage = 'idle' | 'voice' | 'speech-to-text' | 'transcribed' | 'llm' | 'complete'

const stageNames: Record<Stage, string> = {
  idle: 'Ready for a prompt',
  voice: 'Voice · recording',
  'speech-to-text': 'Speech-to-Text · transcribing locally',
  transcribed: 'Transcription ready · review before sending',
  llm: 'LLM · waiting for Ollama',
  complete: 'Complete',
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

  async function sendPrompt(event: FormEvent) {
    event.preventDefault()
    if (!prompt.trim() || busy) return
    setBusy(true)
    setError('')
    setResponse('')
    if (stage === 'transcribed') console.info('[voice] transcription submitted to Ollama prompt_characters=%d', prompt.trim().length)
    setStage('llm')
    try {
      const result = await apiFetch('/local-llm/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt.trim() }),
      })
      const body = await result.json().catch(() => ({}))
      if (!result.ok) throw new Error(typeof body?.detail === 'string' ? body.detail : 'LLM stage failed: Ollama did not return a response.')
      const text = typeof body?.response === 'string' ? body.response.trim() : ''
      if (!text) throw new Error('LLM stage failed: Ollama returned an empty response.')
      setResponse(text)
      console.info('[llm] response received characters=%d model=%s', text.length, body?.model || 'unknown')
      setModel(typeof body?.model === 'string' ? body.model : '')
      setStage('complete')
    } catch (cause) {
      setStage('llm')
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
          <div className="eyebrow"><span /> LOCAL MODEL WORKSPACE</div>
          <h1>Local LLM test</h1>
          <p>Try text or dictate a prompt. Speech is transcribed locally, then sent to your configured Ollama model.</p>
        </div>
      </div>

      <form className="local-llm-card" onSubmit={sendPrompt}>
        <div className="local-llm-title-row">
          <div className="local-llm-icon"><Cpu size={18} /></div>
          <div><h2>Prompt</h2><p>Text and voice use the same local Ollama endpoint.</p></div>
          <span className="local-llm-model"><span /> Ollama · {model || 'configured model'}</span>
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
            {busy ? <><span className="spinner" /> Running locally…</> : <><Send size={14} /> Send to Ollama</>}
          </button>
        </div>
        {error && <div className="local-llm-error" role="alert"><b>{stage === 'voice' ? 'Voice' : stage === 'speech-to-text' ? 'Speech-to-Text' : 'LLM'} stage failed</b><span>{error}</span></div>}
        {response && <section className="local-llm-response" aria-live="polite"><div><span className="hero-kicker"><Sparkles size={13} /> LOCAL MODEL RESPONSE</span><span className="local-llm-response-model">{model}</span></div><p>{response}</p></section>}
      </form>
    </div>
  )
}
