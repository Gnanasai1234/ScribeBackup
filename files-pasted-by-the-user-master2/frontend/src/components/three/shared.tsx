import * as React from 'react'
import { Canvas, type CanvasProps } from '@react-three/fiber'

/** Shared status vocabulary every visualization maps real app state onto. */
export type NodeState = 'idle' | 'processing' | 'completed' | 'failed'

/** Theme colors pulled from style.css custom properties, so 3D matches 2D UI. */
export const THEME = {
  idle: '#52615c',
  processing: '#edb879',
  completed: '#58d68d',
  failed: '#ff6868',
  line: '#26342f',
  panel: '#0d1915',
  text: '#e8efeb',
  muted: '#82948b',
  green: '#71e3a5',
  green2: '#39c681',
} as const

export function colorForState(state: NodeState): string {
  return THEME[state]
}

/** One error boundary implementation, reused everywhere a canvas is mounted. */
export class CanvasBoundary extends React.Component<
  { children: React.ReactNode; fallback: React.ReactNode },
  { failed: boolean }
> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  componentDidCatch(error: Error) {
    console.warn('3D visualization unavailable; using text fallback.', error)
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

/** Respect prefers-reduced-motion: components read this to skip continuous animation. */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = React.useState(
    () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches,
  )
  React.useEffect(() => {
    if (!window.matchMedia) return
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)')
    const onChange = () => setReduced(mq.matches)
    mq.addEventListener?.('change', onChange)
    return () => mq.removeEventListener?.('change', onChange)
  }, [])
  return reduced
}

/** Pause per-frame animation work when a visualization scrolls out of view. */
export function useInView<T extends HTMLElement>(): [React.RefObject<T>, boolean] {
  const ref = React.useRef<T>(null!)
  const [inView, setInView] = React.useState(true)
  React.useEffect(() => {
    if (!ref.current || typeof IntersectionObserver === 'undefined') return
    const io = new IntersectionObserver(([entry]) => setInView(entry.isIntersecting), { threshold: 0.1 })
    io.observe(ref.current)
    return () => io.disconnect()
  }, [])
  return [ref, inView]
}

/**
 * One canvas per visualization, with sane, performance-conscious defaults
 * (capped device pixel ratio, no shadows, transparent background) so every
 * component below only has to describe its scene, not set up WebGL.
 */
export function Stage({
  height,
  fallback,
  children,
  camera,
  ...rest
}: {
  height: number | string
  fallback: React.ReactNode
  children: React.ReactNode
  camera?: CanvasProps['camera']
} & Omit<CanvasProps, 'camera' | 'children'>) {
  const [ref, inView] = useInView<HTMLDivElement>()
  const supportsWebGL = React.useMemo(() => {
    try {
      const canvas = document.createElement('canvas')
      return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))
    } catch {
      return false
    }
  }, [])
  return (
    <div ref={ref} style={{ height, position: 'relative' }} className="three-stage">
      <CanvasBoundary fallback={fallback}>
        {supportsWebGL ? (
          <Canvas
            dpr={[1, 1.5]}
            gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }}
            camera={camera ?? { position: [0, 0, 6], fov: 45 }}
            frameloop={inView ? 'always' : 'never'}
            {...rest}
          >
            {children}
          </Canvas>
        ) : (
          fallback
        )}
      </CanvasBoundary>
    </div>
  )
}
