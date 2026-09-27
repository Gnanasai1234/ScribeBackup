import * as React from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { Stage, colorForState, usePrefersReducedMotion } from './shared'

const BAR_COUNT = 32

/** Reads real frequency data from the given audio element via the Web Audio API. */
function useAnalyser(audioEl: HTMLAudioElement | null) {
  const analyserRef = React.useRef<AnalyserNode | null>(null)
  const dataRef = React.useRef<Uint8Array>(new Uint8Array(BAR_COUNT))
  const connectedTo = React.useRef<HTMLAudioElement | null>(null)

  React.useEffect(() => {
    if (!audioEl || connectedTo.current === audioEl) return
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext
      const ctx = new AudioCtx()
      const source = ctx.createMediaElementSource(audioEl)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 64
      source.connect(analyser)
      analyser.connect(ctx.destination)
      analyserRef.current = analyser
      dataRef.current = new Uint8Array(analyser.frequencyBinCount)
      connectedTo.current = audioEl
    } catch (e) {
      // Web Audio can fail (autoplay policy, CORS, already-connected element) — visualization
      // falls back to the idle pulse below; playback itself is unaffected.
      console.warn('Audio analyser unavailable; using fallback pulse.', e)
    }
  }, [audioEl])

  return { analyser: analyserRef.current, data: dataRef.current }
}

function Bars({ playing, analyser, data, reduced }: { playing: boolean; analyser: AnalyserNode | null; data: Uint8Array; reduced: boolean }) {
  const ref = React.useRef<THREE.InstancedMesh>(null!)
  const dummy = React.useMemo(() => new THREE.Object3D(), [])
  const phase = React.useRef(0)

  useFrame(() => {
    if (!ref.current) return
    if (analyser) analyser.getByteFrequencyData(data)
    phase.current += reduced ? 0 : 0.12
    for (let i = 0; i < BAR_COUNT; i++) {
      let amp: number
      if (analyser) {
        amp = playing ? (data[i] ?? 0) / 255 : 0.04
      } else {
        // No analyser (browser speech synthesis): honest, simple state-driven pulse, not fake audio data.
        amp = playing && !reduced ? 0.15 + Math.abs(Math.sin(phase.current + i * 0.4)) * 0.35 : 0.04
      }
      const height = 0.06 + amp * 1.1
      dummy.position.set((i - BAR_COUNT / 2) * 0.09, height / 2 - 0.6, 0)
      dummy.scale.set(1, height, 1)
      dummy.updateMatrix()
      ref.current.setMatrixAt(i, dummy.matrix)
    }
    ref.current.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh ref={ref} args={[undefined, undefined, BAR_COUNT]}>
      <boxGeometry args={[0.05, 1, 0.05]} />
      <meshStandardMaterial color={colorForState('completed')} emissive={colorForState('completed')} emissiveIntensity={playing ? 0.5 : 0.1} />
    </instancedMesh>
  )
}

export function AudioVisualization({ audioEl, playing }: { audioEl: HTMLAudioElement | null; playing: boolean }) {
  const reduced = usePrefersReducedMotion()
  const { analyser, data } = useAnalyser(audioEl)
  const fallback = <div className="canvas-fallback">{playing ? 'Playing…' : 'Audio idle'}</div>
  return (
    <div className="audio-visual">
      <Stage height={120} fallback={fallback} camera={{ position: [0, 0, 3], fov: 42 }}>
        <ambientLight intensity={1.2} />
        <Bars playing={playing} analyser={analyser} data={data} reduced={reduced} />
      </Stage>
    </div>
  )
}
