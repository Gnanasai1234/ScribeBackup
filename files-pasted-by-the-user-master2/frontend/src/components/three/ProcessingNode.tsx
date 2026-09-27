import * as React from 'react'
import { useFrame } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import * as THREE from 'three'
import { colorForState, type NodeState } from './shared'

export function ProcessingNode({
  position,
  state,
  label,
  radius = 0.16,
  onClick,
  selected,
  reduced,
}: {
  position: [number, number, number]
  state: NodeState
  label?: string
  radius?: number
  onClick?: () => void
  selected?: boolean
  reduced?: boolean
}) {
  const mesh = React.useRef<THREE.Mesh>(null!)
  const [hovered, setHovered] = React.useState(false)
  const color = colorForState(state)

  useFrame(({ clock }) => {
    if (!mesh.current) return
    const t = clock.elapsedTime
    const pulse = !reduced && state === 'processing' ? 1 + Math.sin(t * 3) * 0.18 : 1
    const hoverBoost = hovered || selected ? 1.22 : 1
    mesh.current.scale.setScalar(pulse * hoverBoost)
    if (!reduced) mesh.current.rotation.y = t * 0.3
  })

  return (
    <group position={position}>
      <mesh
        ref={mesh}
        onPointerOver={(e) => {
          e.stopPropagation()
          setHovered(true)
          document.body.style.cursor = onClick ? 'pointer' : 'default'
        }}
        onPointerOut={() => {
          setHovered(false)
          document.body.style.cursor = 'auto'
        }}
        onClick={onClick ? (e) => (e.stopPropagation(), onClick()) : undefined}
      >
        <sphereGeometry args={[radius, 20, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={state === 'processing' ? 0.75 : state === 'failed' ? 0.5 : 0.25}
        />
      </mesh>
      {(hovered || selected) && label && (
        <Html center distanceFactor={8} style={{ pointerEvents: 'none' }}>
          <div className="three-tooltip">
            <b>{label}</b>
            <span>{state}</span>
          </div>
        </Html>
      )}
    </group>
  )
}
