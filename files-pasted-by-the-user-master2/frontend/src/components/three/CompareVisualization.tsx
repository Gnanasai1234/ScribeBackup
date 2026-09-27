import * as React from 'react'
import { Html } from '@react-three/drei'
import { Stage, colorForState, usePrefersReducedMotion } from './shared'

const MAX_PER_SIDE = 14

function Column({ items, x, color, label, onSelect, selected }: { items: string[]; x: number; color: string; label: string; onSelect: (t: string) => void; selected: string | null }) {
  const shown = items.slice(0, MAX_PER_SIDE)
  return (
    <group position={[x, 0, 0]}>
      {shown.map((text, i) => {
        const y = ((shown.length - 1) / 2 - i) * 0.24
        const isSelected = selected === text
        return (
          <group key={i} position={[0, y, 0]}>
            <mesh onClick={() => onSelect(text)} scale={isSelected ? 1.3 : 1}>
              <sphereGeometry args={[0.06, 12, 10]} />
              <meshStandardMaterial color={color} emissive={color} emissiveIntensity={isSelected ? 0.6 : 0.25} />
            </mesh>
          </group>
        )
      })}
      <Html position={[0, (shown.length / 2) * 0.24 + 0.25, 0]} center distanceFactor={9} style={{ pointerEvents: 'none' }}>
        <div className="three-tooltip">
          <b>{label}</b>
          <span>{items.length}</span>
        </div>
      </Html>
    </group>
  )
}

export function CompareVisualization({ added, removed }: { added: string[]; removed: string[] }) {
  usePrefersReducedMotion()
  const [selected, setSelected] = React.useState<string | null>(null)
  const fallback = <div className="canvas-fallback">+{added.length} added · −{removed.length} removed</div>
  if (added.length === 0 && removed.length === 0) {
    return (
      <div className="knowledge-graph-empty">
        <p>No differences to visualize.</p>
      </div>
    )
  }
  return (
    <div className="compare-visual">
      <Stage height={280} fallback={fallback} camera={{ position: [0, 0, 4.6], fov: 46 }}>
        <ambientLight intensity={1.3} />
        <pointLight position={[0, 1, 3]} intensity={2} />
        <Column items={removed} x={-1} color={colorForState('failed')} label="Removed" onSelect={setSelected} selected={selected} />
        <Column items={added} x={1} color={colorForState('completed')} label="Added" onSelect={setSelected} selected={selected} />
      </Stage>
      <div className="compare-visual-detail">{selected ? <p>{selected}</p> : <p className="muted">Click a node to read that sentence.</p>}</div>
    </div>
  )
}
