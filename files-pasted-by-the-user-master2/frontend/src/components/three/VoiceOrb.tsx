import * as React from 'react'
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { Stage, usePrefersReducedMotion, THEME } from './shared'
import type { VoiceState } from '../VoiceInput'

// ─────────────────────────────────────────────────────────────────────────────
// Orb – pulsing icosahedron that reflects voice state
// ─────────────────────────────────────────────────────────────────────────────

function Orb({ state, audioLevel }: { state: VoiceState; audioLevel: number }) {
  const meshRef = useRef<THREE.Mesh>(null!)
  const innerRef = useRef<THREE.Mesh>(null!)
  const reduced = usePrefersReducedMotion()

  const phase = useRef(0)

  useFrame(({ clock }) => {
    phase.current = clock.elapsedTime

    if (!meshRef.current || !innerRef.current) return

    const t = phase.current
    const isListening = state === 'listening'
    const isTranscribing = state === 'transcribing' || state === 'processing_audio'
    const isReady = state === 'transcription_ready'

    // Rotation
    if (!reduced) {
      meshRef.current.rotation.y = t * 0.3
      meshRef.current.rotation.x = Math.sin(t * 0.17) * 0.18
      innerRef.current.rotation.y = -t * 0.22
    }

    // Scale pulse based on state + audio level
    let pulse = 1
    if (isListening) {
      const base = 0.06
      const reactivity = audioLevel * 0.25
      pulse = 1 + base + reactivity + (reduced ? 0 : Math.sin(t * 4) * 0.03)
    } else if (isTranscribing) {
      pulse = 1 + (reduced ? 0 : Math.sin(t * 6) * 0.045)
    } else if (isReady) {
      pulse = 1 + (reduced ? 0 : Math.sin(t * 2) * 0.02)
    }

    meshRef.current.scale.setScalar(pulse)
    innerRef.current.scale.setScalar(pulse * 0.68)

    // Color / emissive based on state
    const mat = meshRef.current.material as THREE.MeshStandardMaterial
    const innerMat = innerRef.current.material as THREE.MeshBasicMaterial

    if (state === 'error') {
      mat.emissive.set(THEME.failed)
      mat.emissiveIntensity = 0.9
      innerMat.color.set(THEME.failed)
      innerMat.opacity = 0.2
    } else if (isListening) {
      mat.emissive.set('#ff6b2b')
      mat.emissiveIntensity = 0.55 + audioLevel * 0.7
      innerMat.color.set('#ff8c4d')
      innerMat.opacity = 0.14 + audioLevel * 0.15
    } else if (isTranscribing) {
      mat.emissive.set(THEME.processing)
      mat.emissiveIntensity = 0.8
      innerMat.color.set(THEME.processing)
      innerMat.opacity = 0.18
    } else if (isReady) {
      mat.emissive.set(THEME.completed)
      mat.emissiveIntensity = 0.7
      innerMat.color.set(THEME.completed)
      innerMat.opacity = 0.18
    } else {
      // idle
      mat.emissive.set(THEME.green2)
      mat.emissiveIntensity = 0.5
      innerMat.color.set(THEME.green)
      innerMat.opacity = 0.12
    }
  })

  return (
    <>
      {/* Outer wireframe orb */}
      <mesh ref={meshRef}>
        <icosahedronGeometry args={[1.1, 2]} />
        <meshStandardMaterial
          color="#19342b"
          wireframe
          emissive={THEME.green2}
          emissiveIntensity={0.5}
        />
      </mesh>

      {/* Inner translucent orb */}
      <mesh ref={innerRef} scale={0.72}>
        <icosahedronGeometry args={[1.1, 2]} />
        <meshBasicMaterial
          color={THEME.green}
          wireframe
          transparent
          opacity={0.12}
        />
      </mesh>

      {/* Point light that matches the emissive color */}
      <pointLight color={THEME.green} intensity={state === 'listening' ? 14 : 7} distance={6} />
    </>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Particles that drift toward the orb when listening
// ─────────────────────────────────────────────────────────────────────────────

const PARTICLE_COUNT = 22

function Particles({ active, audioLevel }: { active: boolean; audioLevel: number }) {
  const ref = useRef<THREE.Points>(null!)
  const reduced = usePrefersReducedMotion()

  const positions = React.useMemo(() => {
    const arr = new Float32Array(PARTICLE_COUNT * 3)
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 5
      arr[i * 3 + 1] = (Math.random() - 0.5) * 5
      arr[i * 3 + 2] = (Math.random() - 0.5) * 5
    }
    return arr
  }, [])

  useFrame(({ clock }) => {
    if (!ref.current || reduced) return
    const t = clock.elapsedTime
    const pos = ref.current.geometry.attributes.position
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const x = pos.getX(i)
      const y = pos.getY(i)
      const z = pos.getZ(i)
      // Drift toward origin when active
      const speed = active ? (0.012 + audioLevel * 0.018) : 0.003
      pos.setXYZ(
        i,
        x * (1 - speed) + Math.sin(t * 0.3 + i) * speed * 0.4,
        y * (1 - speed) + Math.cos(t * 0.25 + i) * speed * 0.4,
        z * (1 - speed),
      )
      // Respawn when too close to center
      if (Math.abs(x) < 0.12 && Math.abs(y) < 0.12) {
        pos.setXYZ(
          i,
          (Math.random() - 0.5) * 5,
          (Math.random() - 0.5) * 5,
          (Math.random() - 0.5) * 5,
        )
      }
    }
    pos.needsUpdate = true
  })

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        color={active ? '#ff8c4d' : THEME.green}
        size={0.055}
        transparent
        opacity={active ? 0.75 : 0.35}
        sizeAttenuation
      />
    </points>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Exported component
// ─────────────────────────────────────────────────────────────────────────────

interface VoiceOrbProps {
  state: VoiceState
  /** Normalised [0,1] audio level from the microphone. */
  audioLevel?: number
  height?: number | string
}

const STATE_LABEL: Record<VoiceState, string> = {
  idle: 'AI CORE · IDLE',
  listening: 'AI CORE · LISTENING',
  processing_audio: 'AI CORE · PROCESSING',
  transcribing: 'AI CORE · TRANSCRIBING',
  transcription_ready: 'AI CORE · READY',
  error: 'AI CORE · ERROR',
}

export function VoiceOrb({ state, audioLevel = 0, height = 180 }: VoiceOrbProps) {
  const fallback = (
    <div className="canvas-fallback">{STATE_LABEL[state]}</div>
  )

  return (
    <div className="voice-orb-wrapper" aria-label={STATE_LABEL[state]} aria-live="polite">
      <Stage height={height} fallback={fallback} camera={{ position: [0, 0, 4.2] }}>
        <ambientLight intensity={1.1} />
        <Orb state={state} audioLevel={audioLevel} />
        <Particles active={state === 'listening'} audioLevel={audioLevel} />
      </Stage>
      <div className="voice-orb-label">{STATE_LABEL[state]}</div>
    </div>
  )
}
