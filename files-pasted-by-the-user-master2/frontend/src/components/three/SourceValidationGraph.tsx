import * as React from 'react'
import { Html } from '@react-three/drei'
import { Stage, colorForState, usePrefersReducedMotion } from './shared'

export type Claim = {
  id: string
  claim: string
  status: string
  confidence?: number | null
  evidence?: string
  sources: { page: number | null; section: string; chunk_id: string }[]
  reason: string
}

const MAX_RENDERED_CLAIMS = 10

function statusColor(status: string) {
  if (status === 'SUPPORTED') return colorForState('completed')
  if (status === 'PARTIALLY_SUPPORTED') return colorForState('processing')
  return colorForState('failed')
}

function ClaimNode({ claim, y, selected, onSelect, reduced }: { claim: Claim; y: number; selected: boolean; onSelect: () => void; reduced: boolean }) {
  const [hovered, setHovered] = React.useState(false)
  const color = statusColor(claim.status)
  const hasEvidence = claim.status !== 'UNSUPPORTED' && claim.sources.length > 0
  const evidenceX = claim.status === 'SUPPORTED' ? 1.5 : 1.1

  return (
    <group position={[-1.3, y, 0]}>
      {/* claim shape: sphere = supported/partial, box = unsupported (shape, not just color) */}
      <mesh
        onPointerOver={(e) => {
          e.stopPropagation()
          setHovered(true)
        }}
        onPointerOut={() => setHovered(false)}
        onClick={(e) => (e.stopPropagation(), onSelect())}
        scale={selected || hovered ? 1.2 : 1}
      >
        {claim.status === 'UNSUPPORTED' ? <boxGeometry args={[0.22, 0.22, 0.22]} /> : <sphereGeometry args={[0.14, 18, 14]} />}
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.4} />
      </mesh>
      {(hovered || selected) && (
        <Html center distanceFactor={9} style={{ pointerEvents: 'none' }}>
          <div className="three-tooltip">
            <b>{claim.status.replace('_', ' ')}</b>
            <span>{claim.confidence != null ? `${Math.round(claim.confidence * 100)}% confidence` : 'confidence unavailable'}</span>
          </div>
        </Html>
      )}
      {hasEvidence && (
        <>
          <mesh position={[evidenceX / 2, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
            <cylinderGeometry args={[claim.status === 'SUPPORTED' ? 0.014 : 0.008, claim.status === 'SUPPORTED' ? 0.014 : 0.008, evidenceX, 6]} />
            <meshBasicMaterial color={color} transparent opacity={claim.status === 'SUPPORTED' ? 0.85 : 0.45} />
          </mesh>
          {/* evidence shape: octahedron, distinct from the claim sphere/box */}
          <mesh position={[evidenceX, 0, 0]} scale={claim.status === 'SUPPORTED' ? 1 : 0.7}>
            <octahedronGeometry args={[0.1, 0]} />
            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} transparent opacity={claim.status === 'SUPPORTED' ? 1 : 0.55} />
          </mesh>
        </>
      )}
    </group>
  )
}

export function SourceValidationGraph({ claims }: { claims: Claim[] }) {
  const reduced = usePrefersReducedMotion()
  const [selectedId, setSelectedId] = React.useState<string | null>(null)
  const shown = claims.slice(0, MAX_RENDERED_CLAIMS)
  const selected = shown.find((c) => c.id === selectedId) || null

  if (claims.length === 0) {
    return (
      <div className="knowledge-graph-empty">
        <p>No validated claims yet. Run an analysis to see the claim → evidence graph.</p>
      </div>
    )
  }

  const spacing = 0.42
  const start = ((shown.length - 1) * spacing) / 2

  const fallback = (
    <div className="canvas-fallback">
      {shown.map((c) => `${c.claim.slice(0, 40)}… (${c.status})`).join(' · ')}
    </div>
  )

  return (
    <div className="source-graph">
      <Stage height={Math.max(220, shown.length * 46)} fallback={fallback} camera={{ position: [0.3, 0, 4], fov: 46 }}>
        <ambientLight intensity={1.3} />
        <pointLight position={[0, 1, 3]} intensity={2} />
        {shown.map((c, i) => (
          <ClaimNode key={c.id} claim={c} y={start - i * spacing} selected={selectedId === c.id} onSelect={() => setSelectedId((cur) => (cur === c.id ? null : c.id))} reduced={reduced} />
        ))}
      </Stage>
      {claims.length > MAX_RENDERED_CLAIMS && <small className="muted">Showing {MAX_RENDERED_CLAIMS} of {claims.length} claims for performance.</small>}
      <div className="source-graph-legend">
        <span><i className="legend-dot completed" /> Supported</span>
        <span><i className="legend-dot processing" /> Partially supported</span>
        <span><i className="legend-dot failed" /> Unsupported</span>
      </div>
      <div className="claim-detail-panel">
        {selected ? (
          <div className="claim-row">
            <div>
              <b>{selected.claim}</b>
              <small>
                {selected.status} · confidence {selected.confidence == null ? 'unavailable' : `${Math.round(selected.confidence * 100)}%`} · source page{' '}
                {selected.sources[0]?.page ?? 'n/a'} · chunk {selected.sources[0]?.chunk_id || 'none'}
              </small>
              {selected.evidence && <p>{selected.evidence}</p>}
            </div>
          </div>
        ) : (
          <p className="muted">Click a claim node above to see its validation status, confidence, source page and evidence text here.</p>
        )}
      </div>
    </div>
  )
}
