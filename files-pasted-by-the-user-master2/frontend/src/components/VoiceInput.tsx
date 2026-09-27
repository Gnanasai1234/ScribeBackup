import * as React from 'react'
import { useEffect, useRef, useState, useCallback } from 'react'

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export type VoiceState =
  | 'idle'
  | 'listening'
  | 'processing_audio'
  | 'transcribing'
  | 'transcription_ready'
  | 'error'

/** BCP-47 locale tags for each supported language code. */
const LOCALE_MAP: Record<string, string> = {
  en: 'en-US',
  hi: 'hi-IN',
  te: 'te-IN',
  ta: 'ta-IN',
}

const STATE_LABELS: Record<VoiceState, string> = {
  idle: 'Click microphone to speak',
  listening: 'Listening\u2026',
  processing_audio: 'Processing voice\u2026',
  transcribing: 'Converting speech to text\u2026',
  transcription_ready: 'Review your transcription',
  error: 'Voice input failed. Try again.',
}

// ─────────────────────────────────────────────────────────────────────────────
// Waveform – reads real microphone amplitude via Web Audio API
// ─────────────────────────────────────────────────────────────────────────────

function MicWaveform({ stream }: { stream: MediaStream | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number>(0)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const dataRef = useRef<Uint8Array>(new Uint8Array(0))

  useEffect(() => {
    if (!stream) return
    let ctx: AudioContext | null = null
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext
      ctx = new AudioCtx()
      const source = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 128
      source.connect(analyser)
      analyserRef.current = analyser
      dataRef.current = new Uint8Array(analyser.frequencyBinCount)
    } catch {
      // Fallback: canvas stays silent
    }

    const canvas = canvasRef.current
    if (!canvas) return
    const draw = () => {
      const a = analyserRef.current
      if (a) a.getByteFrequencyData(dataRef.current)
      const cctx = canvas.getContext('2d')
      if (!cctx) return
      const W = canvas.width
      const H = canvas.height
      cctx.clearRect(0, 0, W, H)
      const bars = dataRef.current.length
      const gap = 2
      const barW = Math.max(2, (W - gap * (bars - 1)) / bars)
      for (let i = 0; i < bars; i++) {
        const amp = a ? (dataRef.current[i] ?? 0) / 255 : 0.1 + 0.1 * Math.sin(Date.now() / 200 + i)
        const barH = Math.max(3, amp * H)
        const alpha = 0.55 + amp * 0.45
        cctx.fillStyle = `rgba(113, 227, 165, ${alpha})`
        cctx.beginPath()
        cctx.roundRect(
          i * (barW + gap),
          (H - barH) / 2,
          barW,
          barH,
          barW / 2,
        )
        cctx.fill()
      }
      rafRef.current = requestAnimationFrame(draw)
    }
    draw()

    return () => {
      cancelAnimationFrame(rafRef.current)
      analyserRef.current = null
      dataRef.current = new Uint8Array(0)
      try { ctx?.close() } catch { /* ignore */ }
    }
  }, [stream])

  return (
    <canvas
      ref={canvasRef}
      width={220}
      height={36}
      className="voice-waveform"
      aria-hidden="true"
    />
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main VoiceInput Component
// ─────────────────────────────────────────────────────────────────────────────

export interface VoiceInputProps {
  /** Language code (en / hi / te / ta). Defaults to 'en'. */
  language?: string
  /** Called with the final transcription text when transcription is ready. */
  onTranscript: (text: string) => void
  /** Called when recording starts. */
  onStart?: () => void
  /** Called when recording stops (before transcription). */
  onStop?: () => void
  /** Called whenever the voice state changes. */
  onStateChange?: (state: VoiceState) => void
  /** Disables the button while true. */
  disabled?: boolean
  /** Extra class for the root element. */
  className?: string
}

export function VoiceInput({
  language = 'en',
  onTranscript,
  onStart,
  onStop,
  onStateChange,
  disabled = false,
  className = '',
}: VoiceInputProps) {
  const [voiceState, setVoiceStateRaw] = useState<VoiceState>('idle')
  const [transcript, setTranscript] = useState('')
  const [errorMsg, setErrorMsg] = useState('')
  const [stream, setStream] = useState<MediaStream | null>(null)

  const recognitionRef = useRef<any>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const setVoiceState = useCallback(
    (s: VoiceState) => {
      setVoiceStateRaw(s)
      onStateChange?.(s)
    },
    [onStateChange],
  )

  useEffect(() => {
    return () => {
      recognitionRef.current?.abort()
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const SpeechRecognition =
    typeof window !== 'undefined'
      ? (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
      : null

  const isSupported = !!SpeechRecognition

  const startRecording = useCallback(async () => {
    if (!isSupported) {
      setErrorMsg('Voice input is not supported in this browser. Try Chrome, Edge, or Safari.')
      setVoiceState('error')
      return
    }

    setTranscript('')
    setErrorMsg('')

    let micStream: MediaStream | null = null
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = micStream
      setStream(micStream)
    } catch (err: any) {
      const name = err?.name || ''
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setErrorMsg('Microphone access is required for voice input.')
      } else if (name === 'NotFoundError') {
        setErrorMsg('No microphone was found on this device.')
      } else {
        setErrorMsg('Could not access the microphone. Please check your device settings.')
      }
      setVoiceState('error')
      return
    }

    setVoiceState('listening')
    onStart?.()

    const recognition = new SpeechRecognition()
    recognition.lang = LOCALE_MAP[language] ?? 'en-US'
    recognition.continuous = false
    recognition.interimResults = false
    recognition.maxAlternatives = 1
    recognitionRef.current = recognition

    recognition.onresult = (event: any) => {
      const text: string = event.results[0]?.[0]?.transcript?.trim() ?? ''
      if (!text) {
        setErrorMsg("Didn't hear anything. Please try again.")
        setVoiceState('error')
      } else {
        setTranscript(text)
        setVoiceState('transcription_ready')
      }
    }

    recognition.onerror = (event: any) => {
      const code: string = event.error ?? ''
      switch (code) {
        case 'not-allowed':
          setErrorMsg('Microphone access was denied.')
          break
        case 'no-speech':
          setErrorMsg("Didn't hear anything. Please try again.")
          break
        case 'network':
          setErrorMsg('A network error occurred during voice recognition.')
          break
        case 'aborted':
          return
        default:
          setErrorMsg(`Voice recognition failed (${code || 'unknown'}). Please try again.`)
      }
      setVoiceState('error')
    }

    recognition.onend = () => {
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      setStream(null)
      onStop?.()
      setVoiceStateRaw((prev) => {
        if (prev === 'listening' || prev === 'processing_audio') {
          onStateChange?.('transcribing')
          return 'transcribing'
        }
        return prev
      })
    }

    try {
      recognition.start()
    } catch {
      setErrorMsg('Could not start voice recognition. Please try again.')
      setVoiceState('error')
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      setStream(null)
    }
  }, [isSupported, language, onStart, onStop, setVoiceState, onStateChange, SpeechRecognition])

  const stopRecording = useCallback(() => {
    setVoiceState('processing_audio')
    recognitionRef.current?.stop()
  }, [setVoiceState])

  const submit = useCallback(() => {
    const text = transcript.trim()
    if (!text) return
    onTranscript(text)
    setVoiceState('idle')
    setTranscript('')
  }, [transcript, onTranscript, setVoiceState])

  const discard = useCallback(() => {
    recognitionRef.current?.abort()
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    setStream(null)
    setVoiceState('idle')
    setTranscript('')
    setErrorMsg('')
  }, [setVoiceState])

  const isRecording = voiceState === 'listening' || voiceState === 'processing_audio'
  const isReady = voiceState === 'transcription_ready'
  const isError = voiceState === 'error'

  return (
    <div className={`voice-input-root ${className}`} data-state={voiceState}>
      {/* Microphone trigger button */}
      {(voiceState === 'idle' || isError) && (
        <>
          <button
            type="button"
            className={`voice-mic-btn ${isError ? 'voice-mic-btn--error' : ''}`}
            onClick={startRecording}
            disabled={disabled || !isSupported}
            aria-label={isSupported ? 'Start voice input' : 'Voice input not supported'}
            title={isSupported ? `Voice input (${language.toUpperCase()})` : 'Voice input not available in this browser'}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
              <line x1="12" x2="12" y1="19" y2="22"/>
            </svg>
          </button>
          {isError && errorMsg && (
            <p className="voice-error" role="alert">{errorMsg}</p>
          )}
          {!isSupported && (
            <p className="voice-unsupported" role="note">
              Voice input requires Chrome, Edge, or Safari.
            </p>
          )}
        </>
      )}

      {/* Recording panel */}
      {isRecording && (
        <div className="voice-recording-panel" role="status" aria-live="polite">
          <div className="voice-recording-header">
            <span className="voice-dot" aria-hidden="true" />
            <span className="voice-status-text">{STATE_LABELS[voiceState]}</span>
          </div>
          <MicWaveform stream={stream} />
          <button
            type="button"
            className="voice-stop-btn"
            onClick={stopRecording}
            aria-label="Stop voice input"
          >
            &#9632; Stop Recording
          </button>
        </div>
      )}

      {/* Transcribing spinner */}
      {voiceState === 'transcribing' && (
        <div className="voice-transcribing" role="status" aria-live="polite">
          <span className="spinner" aria-hidden="true" />
          <span>{STATE_LABELS.transcribing}</span>
        </div>
      )}

      {/* Transcription ready */}
      {isReady && (
        <div className="voice-transcript-panel" role="region" aria-label="Voice transcription ready">
          <div className="voice-transcript-header">
            <span className="voice-transcript-icon" aria-hidden="true">&#127897;</span>
            <span>Voice input complete &mdash; review and edit before submitting</span>
          </div>
          <textarea
            className="voice-transcript-edit"
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
            rows={2}
            aria-label="Edit transcription before submitting"
            placeholder="Your transcription will appear here&hellip;"
          />
          <div className="voice-transcript-actions">
            <button
              type="button"
              className="secondary-btn"
              onClick={discard}
              aria-label="Discard voice transcription"
            >
              Discard
            </button>
            <button
              type="button"
              className="primary-btn"
              onClick={submit}
              disabled={!transcript.trim()}
              aria-label="Use this transcription"
            >
              Use this &#8593;
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
