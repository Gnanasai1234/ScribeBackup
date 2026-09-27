import * as React from 'react'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { ProcessingNode } from './ProcessingNode'
import { Stage, colorForState, usePrefersReducedMotion } from './shared'

type Category = { key: string; label: string; items: string[] }
const MAX_LEAVES_PER_CATEGORY = 5

function polar(angle: number, radius: number): [number, number, number] {
  return [Math.cos(angle) * radius, Math.sin(angle) * radius * 0.55, 0]
}

function GraphScene({
  categories,
  onSelect,
  selectedKey,
  reduced,
  controlsRef,
}: {
  categories: Category[]
  onSelect: (key: string, text: string) => void
  selectedKey: string | null
  reduced: boolean
  controlsRef: React.RefObject<any>
}) {
  const catAngleStep = (Math.PI * 2) / Math.max(categories.length, 1)
  return (
    <>
      <ambientLight intensity={1.2} />
      <pointLight position={[0, 2, 4]} intensity={2} />
      <ProcessingNode position={[0, 0, 0]} state="completed" label="Document" radius={0.2} />
      {categories.map((cat, ci) => {
        const catPos = polar(ci * catAngleStep, 1.6)
        const catKey = `cat:${cat.key}`
        const leaves = cat.items.slice(0, MAX_LEAVES_PER_CATEGORY)
        const leafStep = 0.5
        return (
          <group key={cat.key}>
            <Edge from={[0, 0, 0]} to={catPos} />
            <ProcessingNode
              position={catPos}
              state="completed"
              label={`${cat.label} (${cat.items.length})`}
              radius={0.14}
              onClick={() => onSelect(catKey, `${cat.label}: ${cat.items.length} item${cat.items.length === 1 ? '' : 's'}`)}
              selected={selectedKey === catKey}
              reduced={reduced}
            />
            {leaves.map((item, ii) => {
              const leafPos: [number, number, number] = [
                catPos[0] + (ii - (leaves.length - 1) / 2) * leafStep * 0.5,
                catPos[1] + (catPos[1] >= 0 ? 1 : -1) * (0.55 + ii * 0.02),
                0,
              ]
              const leafKey = `${cat.key}:${ii}`
              return (
                <group key={leafKey}>
                  <Edge from={catPos} to={leafPos} thin />
                  <ProcessingNode
                    position={leafPos}
                    state="idle"
                    radius={0.07}
                    onClick={() => onSelect(leafKey, item)}
                    selected={selectedKey === leafKey}
                    reduced={reduced}
                  />
                </group>
              )
            })}
          </group>
        )
      })}
      <OrbitControls ref={controlsRef} enablePan={false} minDistance={2} maxDistance={7} makeDefault />
    </>
  )
}

function Edge({ from, to, thin }: { from: [number, number, number]; to: [number, number, number]; thin?: boolean }) {
  const a = new THREE.Vector3(...from)
  const b = new THREE.Vector3(...to)
  const mid = a.clone().add(b).multiplyScalar(0.5)
  const dir = b.clone().sub(a)
  const len = dir.length()
  const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize())
  return (
    <mesh position={mid} quaternion={quaternion}>
      <cylinderGeometry args={[thin ? 0.006 : 0.012, thin ? 0.006 : 0.012, len, 6]} />
      <meshBasicMaterial color={colorForState('completed')} transparent opacity={thin ? 0.4 : 0.7} />
    </mesh>
  )
}

export function KnowledgeGraph({
  findings = [],
  facts = [],
  risks = [],
  decisions = [],
  recommendations = [],
  actionItems = [],
}: {
  findings?: string[]
  facts?: string[]
  risks?: string[]
  decisions?: string[]
  recommendations?: string[]
  actionItems?: string[]
}) {
  const reduced = usePrefersReducedMotion()
  const controlsRef = React.useRef<any>(null)
  const [selected, setSelected] = React.useState<{ key: string; text: string } | null>(null)

  const categories: Category[] = [
    { key: 'findings', label: 'Findings', items: findings },
    { key: 'risks', label: 'Risks', items: risks },
    { key: 'decisions', label: 'Decisions', items: decisions },
    { key: 'facts', label: 'Facts', items: facts },
    { key: 'recommendations', label: 'Recommendations', items: recommendations },
    { key: 'action_items', label: 'Action items', items: actionItems },
  ].filter((c) => c.items.length > 0)

  if (categories.length === 0) {
    return (
      <div className="knowledge-graph-empty">
        <p>No analysis data yet. Run an analysis to generate the knowledge graph.</p>
      </div>
    )
  }

  const fallback = (
    <div className="canvas-fallback">
      Document {categories.map((c) => `→ ${c.label} (${c.items.length})`).join(' ')}
    </div>
  )

  return (
    <div className="knowledge-graph">
      <div className="knowledge-graph-head">
        <h4>Knowledge graph</h4>
        <button className="secondary-btn" onClick={() => controlsRef.current?.reset()}>
          Reset camera
        </button>
      </div>
      <Stage height={340} fallback={fallback} camera={{ position: [0, 0.4, 4.2], fov: 48 }}>
        <GraphScene categories={categories} onSelect={(key, text) => setSelected({ key, text })} selectedKey={selected?.key ?? null} reduced={reduced} controlsRef={controlsRef} />
      </Stage>
      <div className="knowledge-graph-detail">
        {selected ? (
          <p>
            <b>Selected:</b> {selected.text}
          </p>
        ) : (
          <p className="muted">Click a node to see its detail here. Drag to orbit, scroll to zoom.</p>
        )}
      </div>
    </div>
  )
}
