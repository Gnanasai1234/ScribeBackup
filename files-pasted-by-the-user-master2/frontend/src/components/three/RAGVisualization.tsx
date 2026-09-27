import * as React from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { ProcessingNode } from './ProcessingNode'
import { Stage, colorForState, usePrefersReducedMotion, type NodeState } from './shared'

export type RagChunk = { chunk_id: string; page: number | null; section: string; text: string }
export type RagPhase = 'idle' | 'retrieving' | 'answered'

function Particle({ target, active, reduced }: { target: THREE.Vector3; active: boolean; reduced: boolean }) {
  const ref = React.useRef<THREE.Mesh>(null!)
  useFrame(({ clock }) => {
    if (!ref.current) return
    if (!active || reduced) {
      ref.current.visible = false
      return
    }
    ref.current.visible = true
    const t = (clock.elapsedTime * 0.7) % 1
    ref.current.position.lerpVectors(new THREE.Vector3(0, 1.1, 0), target, t)
    ;(ref.current.material as THREE.MeshBasicMaterial).opacity = Math.sin(t * Math.PI)
  })
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[0.035, 8, 8]} />
      <meshBasicMaterial color={colorForState('processing')} transparent opacity={0.8} />
    </mesh>
  )
}

function RagScene({ phase, chunks, selected, onSelect, reduced }: { phase: RagPhase; chunks: RagChunk[]; selected: string | null; onSelect: (id: string) => void; reduced: boolean }) {
  const questionState: NodeState = 'completed'
  const embeddingState: NodeState = phase === 'idle' ? 'idle' : 'completed'
  const llmState: NodeState = phase === 'retrieving' ? 'processing' : phase === 'answered' ? 'completed' : 'idle'
  const answerState: NodeState = phase === 'answered' ? 'completed' : 'idle'

  const positions = chunks.map((_, i) => {
    const spread = chunks.length > 1 ? (i - (chunks.length - 1) / 2) : 0
    return new THREE.Vector3(spread * 0.55, 0.2, 0)
  })

  return (
    <>
      <ambientLight intensity={1.3} />
      <pointLight position={[0, 1, 3]} intensity={2} />
      <ProcessingNode position={[0, 1.1, 0]} state={questionState} label="Question" radius={0.12} />
      <ProcessingNode position={[0, 0.65, 0]} state={embeddingState} label="Embedding" radius={0.1} />
      {chunks.map((c, i) => (
        <group key={c.chunk_id}>
          <mesh position={[positions[i].x / 2, (0.65 + 0.2) / 2, 0]} rotation={[0, 0, Math.PI / 4]}>
            <cylinderGeometry args={[0.008, 0.008, 0.55, 6]} />
            <meshBasicMaterial color={colorForState(phase === 'idle' ? 'idle' : 'completed')} transparent opacity={0.5} />
          </mesh>
          <ProcessingNode
            position={[positions[i].x, positions[i].y, positions[i].z]}
            state={phase === 'idle' ? 'idle' : 'completed'}
            label={`${c.section}${c.page ? ` · p.${c.page}` : ''}`}
            radius={0.09}
            onClick={() => onSelect(c.chunk_id)}
            selected={selected === c.chunk_id}
            reduced={reduced}
          />
          <Particle target={new THREE.Vector3(0, -0.35, 0)} active={phase === 'retrieving'} reduced={reduced} />
        </group>
      ))}
      <ProcessingNode position={[0, -0.35, 0]} state={llmState} label="LLM" radius={0.12} />
      <ProcessingNode position={[0, -0.85, 0]} state={answerState} label="Answer" radius={0.1} />
    </>
  )
}

export function RAGVisualization({ phase, chunks, onSelectChunk, selectedChunk }: { phase: RagPhase; chunks: RagChunk[]; onSelectChunk?: (id: string) => void; selectedChunk?: string | null }) {
  const reduced = usePrefersReducedMotion()
  const fallback = (
    <div className="canvas-fallback">
      Question → Embedding → {chunks.length ? `${chunks.length} retrieved chunks` : 'Retrieval'} → LLM → Answer
    </div>
  )
  return (
    <div className="rag-visual">
      <Stage height={260} fallback={fallback} camera={{ position: [0, 0, 3.6], fov: 46 }}>
        <RagScene phase={phase} chunks={chunks} selected={selectedChunk ?? null} onSelect={(id) => onSelectChunk?.(id)} reduced={reduced} />
      </Stage>
    </div>
  )
}
