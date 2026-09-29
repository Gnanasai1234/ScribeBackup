import * as React from 'react'
import { useEffect, useRef, useState, useCallback } from 'react'
import { apiFetch } from '../lib/api'

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
  listening: 'Listening... Recording...',
  processing_audio: 'Processing audio...',
  transcribing: 'Transcribing locally...',
  transcription_ready: 'Transcription ready',
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
  /** Use backend-hosted offline transcription instead of browser speech services. */
  transcriptionMode?: 'browser' | 'local'
  /** Reports which voice pipeline stage failed. */
  onError?: (stage: 'voice' | 'speech-to-text', message: string) => void
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
  transcriptionMode = 'browser',
  onError,
  className = '',
}: VoiceInputProps) {
  const [voiceState, setVoiceStateRaw] = useState<VoiceState>('idle')
  const [transcript, setTranscript] = useState('')
  const [errorMsg, setErrorMsg] = useState('')
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [audioInputs, setAudioInputs] = useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')

  const recognitionRef = useRef<any>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const audioPartsRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const recordingStartedAtRef = useRef(0)
  const activeRef = useRef(false)
  const recognitionFinalRef = useRef('')
  const recognitionInterimRef = useRef('')
  const recognitionFailedRef = useRef(false)

  useEffect(() => {
    if (transcriptionMode !== 'local' || !navigator.mediaDevices?.enumerateDevices) return
    const refreshDevices = () => navigator.mediaDevices.enumerateDevices()
      .then((devices) => setAudioInputs(devices.filter((device) => device.kind === 'audioinput')))
      .catch(() => setAudioInputs([]))
    void refreshDevices()
    navigator.mediaDevices.addEventListener?.('devicechange', refreshDevices)
    return () => navigator.mediaDevices.removeEventListener?.('devicechange', refreshDevices)
  }, [transcriptionMode])

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
      activeRef.current = false
      if (recorderRef.current) {
        // Unmount cleanup must never trigger an upload.
        recorderRef.current.onstop = null
        recorderRef.current.ondataavailable = null
        if (recorderRef.current.state !== 'inactive') recorderRef.current.stop()
      }
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const SpeechRecognition =
    typeof window !== 'undefined'
      ? (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
      : null

  const isSupported = transcriptionMode === 'local'
    ? typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== 'undefined'
    : !!SpeechRecognition

  const startRecording = useCallback(async () => {
    if (activeRef.current || voiceState === 'listening' || voiceState === 'processing_audio' || voiceState === 'transcribing') return
    if (!isSupported) {
      const message = transcriptionMode === 'local'
        ? 'Voice stage failed: local microphone recording is not supported in this browser.'
        : 'Voice input is not supported in this browser. Try Chrome, Edge, or Safari.'
      setErrorMsg(message)
      onError?.('voice', message)
      setVoiceState('error')
      return
    }

    setTranscript('')
    setErrorMsg('')
    activeRef.current = true

    let micStream: MediaStream | null = null
    try {
      const audio: MediaTrackConstraints | true = selectedDeviceId
        ? { deviceId: { exact: selectedDeviceId } }
        : true
      micStream = await navigator.mediaDevices.getUserMedia({ audio })
      const tracks = micStream.getAudioTracks()
      const track = tracks[0]
      console.info('[voice] microphone permission=granted tracks=%d enabled=%s muted=%s readyState=%s selectedDevice=%s', tracks.length, track?.enabled, track?.muted, track?.readyState, Boolean(track?.getSettings().deviceId))
      if (!track || !track.enabled || track.muted || track.readyState !== 'live') {
        micStream.getTracks().forEach((item) => item.stop())
        const missingTrack = new Error('Microphone device was not found or has no active audio track.')
        missingTrack.name = 'NotFoundError'
        throw missingTrack
      }
      streamRef.current = micStream
      setStream(micStream)
      if (transcriptionMode === 'local') {
        navigator.mediaDevices.enumerateDevices().then((devices) => setAudioInputs(devices.filter((device) => device.kind === 'audioinput'))).catch(() => {})
      }
    } catch (err: any) {
      activeRef.current = false
      const name = err?.name || ''
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setErrorMsg('No microphone permission.')
      } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
        setErrorMsg('Microphone device was not found.')
      } else {
        setErrorMsg(err instanceof Error ? err.message : 'Could not access the microphone. Check device settings.')
      }
      onError?.('voice', name === 'NotFoundError' || name === 'DevicesNotFoundError' ? 'Microphone device was not found.' : name === 'NotAllowedError' || name === 'PermissionDeniedError' ? 'No microphone permission.' : err instanceof Error ? err.message : 'Could not access the microphone.')
      setVoiceState('error')
      return
    }

    setVoiceState('listening')
    onStart?.()

    if (transcriptionMode === 'local') {
      try {
        activeRef.current = true
        audioPartsRef.current = []
        const supportedTypes = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4', 'audio/webm', 'audio/ogg']
        const preferredType = supportedTypes.find((type) => MediaRecorder.isTypeSupported(type))
        const recorder = new MediaRecorder(micStream, preferredType ? { mimeType: preferredType } : undefined)
        recorderRef.current = recorder
        recorder.ondataavailable = (event) => { if (event.data.size) audioPartsRef.current.push(event.data) }
        recorder.onerror = () => {
          activeRef.current = false
          recorder.onstop = null
          audioPartsRef.current = []
          const message = 'Audio recording failed before transcription.'
          setErrorMsg(message)
          onError?.('speech-to-text', message)
          setVoiceState('error')
          micStream.getTracks().forEach((track) => track.stop())
          streamRef.current = null
          setStream(null)
        }
        recorder.onstop = async () => {
          activeRef.current = false
          const durationMs = Math.max(0, Date.now() - recordingStartedAtRef.current)
          micStream.getTracks().forEach((track) => track.stop())
          streamRef.current = null
          setStream(null)
          onStop?.()
          setVoiceState('transcribing')
          const blobType = recorder.mimeType || audioPartsRef.current.find((part) => part.type)?.type || ''
          const blob = new Blob(audioPartsRef.current, blobType ? { type: blobType } : undefined)
          audioPartsRef.current = []
          console.info('[voice] recording stopped duration_ms=%d blob_bytes=%d mime=%s', durationMs, blob.size, blob.type || 'unknown')
          if (!blob.size || durationMs < 50) {
            const message = 'Audio recording was empty.'
            setErrorMsg(message)
            onError?.('speech-to-text', message)
            setVoiceState('error')
            return
          }
          if (blob.size > 15 * 1024 * 1024) {
            const message = 'Audio recording was invalid: recording exceeds 15 MiB.'
            setErrorMsg(message)
            onError?.('speech-to-text', message)
            setVoiceState('error')
            return
          }
          if (!blob.type.startsWith('audio/')) {
            const message = 'Audio recording was invalid: the browser returned an unsupported audio format.'
            setErrorMsg(message)
            onError?.('speech-to-text', message)
            setVoiceState('error')
            return
          }
          console.info('[voice] STT request start engine=faster-whisper language=%s bytes=%d', language, blob.size)
          const extension = blob.type.includes('ogg') ? 'ogg' : blob.type.includes('mp4') ? 'mp4' : blob.type.includes('wav') ? 'wav' : 'webm'
          const form = new FormData()
          form.append('file', blob, `voice-prompt.${extension}`)
          form.append('language', language)
          try {
            const response = await apiFetch('/speech/transcribe', { method: 'POST', body: form })
            const body = await response.json().catch(() => ({}))
            if (!response.ok) throw new Error(typeof body?.detail === 'string' ? body.detail : 'Speech-to-Text stage failed: local transcription did not complete.')
            const text = typeof body?.transcript === 'string' ? body.transcript.trim() : ''
            if (!text) throw new Error('Speech-to-text returned no transcription.')
            console.info('[voice] STT result returned characters=%d language=%s', text.length, body?.language || 'unknown')
            setTranscript(text)
            setVoiceState('transcription_ready')
            onTranscript(text)
          } catch (error) {
            const message = error instanceof Error ? error.message : 'Local speech-to-text service is unavailable.'
            setErrorMsg(message)
            onError?.('speech-to-text', message)
            setVoiceState('error')
          }
        }
        recordingStartedAtRef.current = Date.now()
        recorder.start(500)
        console.info('[voice] recording started mime=%s', recorder.mimeType || 'browser default')
      } catch (error) {
        activeRef.current = false
        micStream.getTracks().forEach((track) => track.stop())
        streamRef.current = null
        setStream(null)
        const message = `Voice stage failed: ${error instanceof Error ? error.message : 'could not start recording.'}`
        setErrorMsg(message)
        onError?.('voice', message)
        setVoiceState('error')
      }
      return
    }

    const recognition = new SpeechRecognition()
    recognition.lang = LOCALE_MAP[language] ?? 'en-US'
    recognition.continuous = true
    recognition.interimResults = true
    recognition.maxAlternatives = 1
    recognitionFinalRef.current = ''
    recognitionInterimRef.current = ''
    recognitionFailedRef.current = false
    recognitionRef.current = recognition

    recognition.onresult = (event: any) => {
      let interim = ''
      for (let index = event.resultIndex ?? 0; index < event.results.length; index += 1) {
        const result = event.results[index]
        const text = result?.[0]?.transcript?.trim() ?? ''
        if (result?.isFinal && text) recognitionFinalRef.current = `${recognitionFinalRef.current} ${text}`.trim()
        else if (text) interim = `${interim} ${text}`.trim()
      }
      recognitionInterimRef.current = interim
      const visibleText = `${recognitionFinalRef.current} ${interim}`.trim()
      if (visibleText) setTranscript(visibleText)
    }

    recognition.onerror = (event: any) => {
      const code: string = event.error ?? ''
      switch (code) {
        case 'not-allowed':
          recognitionFailedRef.current = true
          setErrorMsg('Microphone access was denied.')
          break
        case 'no-speech':
          // Browsers can emit this during a short pause; wait for final results/onend.
          return
        case 'network':
          recognitionFailedRef.current = true
          setErrorMsg('A network error occurred during voice recognition.')
          break
        case 'aborted':
          return
        default:
          recognitionFailedRef.current = true
          setErrorMsg(`Voice recognition failed (${code || 'unknown'}). Please try again.`)
      }
      setVoiceState('error')
    }

    recognition.onend = () => {
      activeRef.current = false
      const text = recognitionFinalRef.current.trim() || recognitionInterimRef.current.trim()
      if (text) {
        setTranscript(text)
        setVoiceState('transcription_ready')
        onTranscript(text)
      } else if (!recognitionFailedRef.current) {
        setErrorMsg('Speech recognition stopped before receiving a result.')
        setVoiceState('error')
      }
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      setStream(null)
      onStop?.()
    }

    try {
      recognition.start()
    } catch {
      activeRef.current = false
      setErrorMsg('Could not start voice recognition. Please try again.')
      setVoiceState('error')
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      setStream(null)
    }
  }, [isSupported, language, transcriptionMode, onStart, onStop, onError, onTranscript, setVoiceState, onStateChange, SpeechRecognition, selectedDeviceId, voiceState])

  const stopRecording = useCallback(() => {
    setVoiceState('processing_audio')
    if (transcriptionMode === 'local') {
      const recorder = recorderRef.current
      if (recorder?.state === 'recording') {
        try { recorder.requestData() } catch { /* stop still dispatches its final dataavailable event */ }
        recorder.stop()
      }
      return
    }
    recognitionRef.current?.stop()
  }, [transcriptionMode, setVoiceState])

  const submit = useCallback(() => {
    const text = transcript.trim()
    if (!text) return
    onTranscript(text)
    setVoiceState('idle')
    setTranscript('')
  }, [transcript, onTranscript, setVoiceState])

  const discard = useCallback(() => {
    activeRef.current = false
    recognitionRef.current?.abort()
    activeRef.current = false
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
          {transcriptionMode === 'local' && audioInputs.length > 1 && (
            <label className="voice-device-select">Microphone
              <select value={selectedDeviceId} onChange={(event) => setSelectedDeviceId(event.target.value)} disabled={disabled}>
                <option value="">Default microphone</option>
                {audioInputs.map((device, index) => <option key={device.deviceId || index} value={device.deviceId}>{device.label || `Microphone ${index + 1}`}</option>)}
              </select>
            </label>
          )}
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
              {transcriptionMode === 'local' ? 'Local voice recording is unavailable in this browser.' : 'Voice input requires Chrome, Edge, or Safari.'}
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
          {transcriptionMode === 'browser' && transcript && <p className="voice-interim-text">{transcript}</p>}
          <MicWaveform stream={stream} />
          <button
            type="button"
            className="voice-stop-btn"
            onClick={stopRecording}
            disabled={voiceState === 'processing_audio'}
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
            onChange={(e) => {
              setTranscript(e.target.value)
              if (transcriptionMode === 'local') onTranscript(e.target.value)
            }}
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
