import * as React from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { Stage, usePrefersReducedMotion } from './shared'

const MAX_RENDERED_CHUNKS = 48

// ─── Animated connector with travelling energy pulse ─────────────────────────
function Connector({
  from,
  to,
  reduced,
}: {
  from: [number, number, number]
  to: [number, number, number]
  reduced: boolean
}) {
  const pulseRef = React.useRef<THREE.Mesh>(null!)
  const glowRef = React.useRef<THREE.Mesh>(null!)

  const start = React.useMemo(() => new THREE.Vector3(...from), [from[0], from[1], from[2]])
  const end   = React.useMemo(() => new THREE.Vector3(...to),   [to[0],   to[1],   to[2]])
  const dir   = end.clone().sub(start)
  const length = dir.length()
  const mid   = start.clone().lerp(end, 0.5)
  const quat  = new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    dir.clone().normalize(),
  )

  useFrame(({ clock }) => {
    if (reduced) return
    const t = clock.elapsedTime
    const progress = (t % 1.8) / 1.8
    if (pulseRef.current) {
      const pulseY = start.y + (end.y - start.y) * progress
      pulseRef.current.position.y = pulseY
      pulseRef.current.scale.setScalar(0.8 + Math.sin(t * 4) * 0.2)
      ;(pulseRef.current.material as THREE.MeshBasicMaterial).opacity = 0.9 - progress * 0.5
    }
    if (glowRef.current) {
      ;(glowRef.current.material as THREE.MeshBasicMaterial).opacity =
        0.08 + Math.sin(t * 2.2) * 0.04
    }
  })

  return (
    <group>
      {/* Glow tube */}
      <mesh ref={glowRef} position={mid.toArray() as [number, number, number]} quaternion={quat}>
        <cylinderGeometry args={[0.025, 0.025, length, 6]} />
        <meshBasicMaterial color="#71e3a5" transparent opacity={0.1} />
      </mesh>
      {/* Core line */}
      <mesh position={mid.toArray() as [number, number, number]} quaternion={quat}>
        <cylinderGeometry args={[0.008, 0.008, length, 6]} />
        <meshBasicMaterial color="#39c681" />
      </mesh>
      {/* Travelling pulse */}
      {!reduced && (
        <mesh ref={pulseRef} position={[start.x, start.y, start.z]}>
          <sphereGeometry args={[0.04, 8, 8]} />
          <meshBasicMaterial color="#a8ffdb" transparent opacity={0.9} />
        </mesh>
      )}
    </group>
  )
}

// ─── Pipeline node with orbiting ring and breathing glow ─────────────────────
function AnimatedNode({
  position,
  radius = 0.16,
  reduced,
  delay = 0,
}: {
  position: [number, number, number]
  radius?: number
  reduced: boolean
  delay?: number
}) {
  const coreRef = React.useRef<THREE.Mesh>(null!)
  const ringRef = React.useRef<THREE.Mesh>(null!)
  const glowRef = React.useRef<THREE.Mesh>(null!)

  useFrame(({ clock }) => {
    if (reduced) return
    const t = clock.elapsedTime + delay
    if (coreRef.current) {
      coreRef.current.rotation.y = t * 0.5
      coreRef.current.scale.setScalar(1 + Math.sin(t * 2.1) * 0.06)
      ;(coreRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity =
        0.55 + Math.sin(t * 2.1) * 0.25
    }
    if (ringRef.current) {
      ringRef.current.rotation.z = t * 0.7
      ringRef.current.rotation.x = Math.sin(t * 0.4) * 0.3
    }
    if (glowRef.current) {
      glowRef.current.scale.setScalar(1 + Math.sin(t * 1.6) * 0.15)
      ;(glowRef.current.material as THREE.MeshBasicMaterial).opacity =
        0.06 + Math.sin(t * 1.6) * 0.04
    }
  })

  return (
    <group position={position}>
      {/* Outer glow */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[radius * 2.2, 12, 10]} />
        <meshBasicMaterial color="#71e3a5" transparent opacity={0.07} />
      </mesh>
      {/* Orbit ring */}
      <mesh ref={ringRef}>
        <torusGeometry args={[radius * 1.6, 0.012, 6, 32]} />
        <meshBasicMaterial color="#39c681" transparent opacity={0.55} />
      </mesh>
      {/* Core sphere */}
      <mesh ref={coreRef}>
        <sphereGeometry args={[radius, 20, 16]} />
        <meshStandardMaterial color="#19342b" emissive="#58d68d" emissiveIntensity={0.55} />
      </mesh>
    </group>
  )
}

// ─── Chunk cloud ──────────────────────────────────────────────────────────────
function ChunkCloud({ count, reduced }: { count: number; reduced: boolean }) {
  const instanceRef = React.useRef<THREE.InstancedMesh>(null!)
  const dummy = React.useMemo(() => new THREE.Object3D(), [])

  const layout = React.useMemo(() => {
    const cols = Math.ceil(Math.sqrt(count))
    return Array.from({ length: count }, (_, i) => {
      const col = i % cols
      const row = Math.floor(i / cols)
      const jitter = (Math.random() - 0.5) * 0.06
      return {
        base: new THREE.Vector3(
          (col - cols / 2 + 0.5) * 0.27 + jitter,
          0.48 + (row - cols / 2 + 0.5) * -0.27 + jitter,
          (Math.random() - 0.5) * 0.2,
        ),
        phase: Math.random() * Math.PI * 2,
        speed: 0.8 + Math.random() * 0.8,
      }
    })
  }, [count])

  useFrame(({ clock }) => {
    if (!instanceRef.current) return
    const t = reduced ? 0 : clock.elapsedTime
    layout.forEach(({ base, phase, speed }, i) => {
      dummy.position.set(
        base.x + (reduced ? 0 : Math.sin(t * speed * 0.5 + phase) * 0.018),
        base.y + (reduced ? 0 : Math.sin(t * speed + phase) * 0.022),
        base.z,
      )
      dummy.scale.setScalar(reduced ? 1 : 0.85 + Math.sin(t * speed + phase) * 0.15)
      dummy.updateMatrix()
      instanceRef.current.setMatrixAt(i, dummy.matrix)
    })
    instanceRef.current.instanceMatrix.needsUpdate = true
    if (!reduced) {
      ;(instanceRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity =
        0.4 + Math.sin(clock.elapsedTime * 1.3) * 0.2
    }
  })

  return (
    <instancedMesh ref={instanceRef} args={[undefined, undefined, count]}>
      <sphereGeometry args={[0.055, 10, 8]} />
      <meshStandardMaterial color="#1a3d2e" emissive="#58d68d" emissiveIntensity={0.4} />
    </instancedMesh>
  )
}

// ─── Ambient floating particles ───────────────────────────────────────────────
const AMBIENT_COUNT = 18
function AmbientParticles({ reduced }: { reduced: boolean }) {
  const ref = React.useRef<THREE.Points>(null!)
  const positions = React.useMemo(() => {
    const arr = new Float32Array(AMBIENT_COUNT * 3)
    for (let i = 0; i < AMBIENT_COUNT; i++) {
      arr[i * 3]     = (Math.random() - 0.5) * 3.2
      arr[i * 3 + 1] = (Math.random() - 0.5) * 2.8
      arr[i * 3 + 2] = (Math.random() - 0.5) * 0.5
    }
    return arr
  }, [])

  useFrame(({ clock }) => {
    if (!ref.current || reduced) return
    const t = clock.elapsedTime
    const pos = ref.current.geometry.attributes.position
    for (let i = 0; i < AMBIENT_COUNT; i++) {
      pos.setY(i, pos.getY(i) + Math.sin(t * 0.4 + i * 0.8) * 0.003)
    }
    pos.needsUpdate = true
    ;(ref.current.material as THREE.PointsMaterial).opacity = 0.28 + Math.sin(t * 0.9) * 0.1
  })

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial color="#71e3a5" size={0.035} transparent opacity={0.28} sizeAttenuation />
    </points>
  )
}

// ─── Full scene ───────────────────────────────────────────────────────────────
function ChunkScene({
  status,
  chunkCount,
  reduced,
}: {
  status: 'idle' | 'extracted'
  chunkCount: number
  reduced: boolean
}) {
  return (
    <>
      <ambientLight intensity={0.8} />
      <pointLight position={[0, 2, 3]} color="#71e3a5" intensity={4} distance={8} />
      <pointLight position={[-1.5, -1, 2]} color="#39c681" intensity={2} distance={6} />

      <AnimatedNode position={[0, 1.3, 0]} radius={0.16} reduced={reduced} delay={0} />
      <AnimatedNode position={[0, 0.35, 0]} radius={0.13} reduced={reduced} delay={1.1} />
      <AnimatedNode position={[0, -1.1, 0]} radius={0.13} reduced={reduced} delay={2.2} />

      <Connector from={[0, 1.15, 0]} to={[0, 0.5, 0]} reduced={reduced} />
      <Connector from={[0, 0.2, 0]} to={[0, -0.35, 0]} reduced={reduced} />

      {status === 'extracted' && chunkCount > 0 && (
        <ChunkCloud count={Math.min(chunkCount, MAX_RENDERED_CHUNKS)} reduced={reduced} />
      )}

      <AmbientParticles reduced={reduced} />
    </>
  )
}

// ─── Exported component ───────────────────────────────────────────────────────
export function DocumentChunkVisualization({
  status,
  chunkCount,
  wordCount,
  pages,
}: {
  status: 'idle' | 'extracted'
  chunkCount: number
  wordCount: number
  pages: number | null
}) {
  const reduced = usePrefersReducedMotion()
  const fallback = (
    <div className="canvas-fallback">
      Document → Extraction → Chunking ({chunkCount} chunks) → Embeddings
    </div>
  )
  return (
    <div className="chunk-visual">
      <Stage height={240} fallback={fallback} camera={{ position: [0, 0, 3.6], fov: 42 }}>
        <ChunkScene status={status} chunkCount={chunkCount} reduced={reduced} />
      </Stage>
      <div className="chunk-visual-stats">
        <span><b>{wordCount.toLocaleString()}</b> words</span>
        <span><b>{chunkCount}</b> chunks</span>
        {pages != null && <span><b>{pages}</b> pages</span>}
        {chunkCount > MAX_RENDERED_CHUNKS && (
          <small>Showing {MAX_RENDERED_CHUNKS} of {chunkCount} chunks for performance</small>
        )}
      </div>
    </div>
  )
}
