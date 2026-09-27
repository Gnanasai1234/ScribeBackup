import * as React from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { ProcessingNode } from './ProcessingNode'
import { Stage, colorForState, usePrefersReducedMotion, type NodeState } from './shared'

export type PipelineStep = { label: string; state: NodeState }

function Edge({ from, to, active, reduced }: { from: number; to: number; active: boolean; reduced?: boolean }) {
  const color = active ? colorForState('processing') : colorForState('completed')
  const mid: [number, number, number] = [(from + to) / 2, 0, 0]
  const len = Math.abs(to - from)
  return (
    <mesh position={mid} rotation={[0, 0, Math.PI / 2]}>
      <cylinderGeometry args={[0.016, 0.016, len - 0.08, 8]} />
      <meshStandardMaterial color={active ? color : '#31463a'} emissive={active ? color : '#000'} emissiveIntensity={active ? 0.5 : 0} transparent opacity={active ? 1 : 0.6} />
    </mesh>
  )
}

/** Small particles that flow from `from` to `to` on the x-axis, looping while active. */
function FlowParticles({ from, to, count = 5, reduced }: { from: number; to: number; count?: number; reduced?: boolean }) {
  const group = React.useRef<THREE.Group>(null!)
  useFrame(({ clock }) => {
    if (reduced || !group.current) return
    const t = clock.elapsedTime
    group.current.children.forEach((child, i) => {
      const phase = ((t * 0.6 + i / count) % 1)
      child.position.x = THREE.MathUtils.lerp(from, to, phase)
      const mat = (child as THREE.Mesh).material as THREE.MeshBasicMaterial
      mat.opacity = Math.sin(phase * Math.PI)
    })
  })
  return (
    <group ref={group}>
      {Array.from({ length: count }).map((_, i) => (
        <mesh key={i}>
          <sphereGeometry args={[0.035, 8, 8]} />
          <meshBasicMaterial color={colorForState('processing')} transparent opacity={0.8} />
        </mesh>
      ))}
    </group>
  )
}

function PipelineScene({ steps, onSelect, selected, reduced }: { steps: PipelineStep[]; onSelect?: (i: number) => void; selected?: number | null; reduced: boolean }) {
  const spacing = 0.92
  const start = -((steps.length - 1) * spacing) / 2
  const xs = steps.map((_, i) => start + i * spacing)
  const activeEdge = steps.findIndex((s) => s.state === 'processing')
  return (
    <>
      <ambientLight intensity={1.3} />
      <pointLight position={[0, 1, 3]} intensity={2} />
      {xs.slice(0, -1).map((x, i) => (
        <Edge key={`edge-${i}`} from={x} to={xs[i + 1]} active={activeEdge === i + 1} reduced={reduced} />
      ))}
      {activeEdge > 0 && <FlowParticles from={xs[activeEdge - 1]} to={xs[activeEdge]} reduced={reduced} />}
      {steps.map((step, i) => (
        <ProcessingNode
          key={step.label}
          position={[xs[i], 0, 0]}
          state={step.state}
          label={step.label}
          selected={selected === i}
          onClick={onSelect ? () => onSelect(i) : undefined}
          reduced={reduced}
        />
      ))}
    </>
  )
}

export function AIProcessingPipeline({ steps, compact }: { steps: PipelineStep[]; compact?: boolean }) {
  const reduced = usePrefersReducedMotion()
  const [selected, setSelected] = React.useState<number | null>(null)
  const fallback = (
    <div className="canvas-fallback">{steps.map((s) => s.label).join(' → ')}</div>
  )
  return (
    <div className="pipeline-visual">
      <Stage height={compact ? 90 : 120} fallback={fallback} camera={{ position: [0, 0, 6.5], fov: 40 }}>
        <PipelineScene steps={steps} onSelect={(i) => setSelected((cur) => (cur === i ? null : i))} selected={selected} reduced={reduced} />
      </Stage>
      <div className="pipeline-labels">
        {steps.map((step, i) => (
          <span key={step.label} className={step.state} onClick={() => setSelected((cur) => (cur === i ? null : i))} role="button" tabIndex={0}>
            {step.label}
          </span>
        ))}
      </div>
      {selected != null && (
        <p className="pipeline-detail">
          <b>{steps[selected].label}:</b> {steps[selected].state}
        </p>
      )}
    </div>
  )
}
