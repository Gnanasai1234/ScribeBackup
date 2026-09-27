import type { PipelineStep } from '../components/three/AIProcessingPipeline'
import type { NodeState } from '../components/three/shared'

export type PipelineInputs = {
  hasDoc: boolean
  busy: boolean
  error: boolean
  resultStatus: string | null | undefined // 'validated' | 'partial' | 'failed' | undefined
  regenerationCount: number
  audioActive: boolean // audio currently generated/playing
}

/** Derives the seven pipeline node states from real, already-fetched app state. */
export function derivePipelineSteps({ hasDoc, busy, error, resultStatus, regenerationCount, audioActive }: PipelineInputs): PipelineStep[] {
  const validated = resultStatus === 'validated' || resultStatus === 'partial' || resultStatus === 'translation_unavailable'
  const failedResult = resultStatus === 'failed'
  const failedProcessing = error && hasDoc && !resultStatus && !busy
  const analyzing = busy && hasDoc

  const documentState: NodeState = hasDoc ? 'completed' : 'idle'
  const analysisState: NodeState = !hasDoc
    ? 'idle'
    : validated || failedResult
      ? 'completed'
      : failedProcessing
        ? 'failed'
        : analyzing
          ? 'processing'
          : 'idle'
  const chunkingState: NodeState = hasDoc ? 'completed' : 'idle'
  const summarizationState: NodeState = validated ? 'completed' : failedResult || failedProcessing ? 'failed' : analyzing ? 'processing' : 'idle'
  const validationState: NodeState = validated
    ? 'completed'
    : failedResult
      ? 'failed'
      : regenerationCount > 0
        ? 'processing'
        : analyzing
          ? 'processing'
          : 'idle'
  const qualityState: NodeState = validated ? 'completed' : failedResult ? 'failed' : 'idle'
  const audioState: NodeState = audioActive ? 'completed' : 'idle'

  return [
    { label: 'Document', state: documentState },
    { label: 'Analysis', state: analysisState },
    { label: 'Chunking', state: chunkingState },
    { label: 'Summarization', state: summarizationState },
    { label: 'Validation', state: validationState },
    { label: 'Quality', state: qualityState },
    { label: 'Audio', state: audioState },
  ]
}
