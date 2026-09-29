import type { PipelineStep } from '../components/three/AIProcessingPipeline'
import type { NodeState } from '../components/three/shared'

export type PipelineInputs = {
  hasDoc: boolean
  busy: boolean
  error: boolean
  resultStatus: string | null | undefined // 'validated' | 'partial' | 'failed' | undefined
  summaryUsable?: boolean
  validationState?: string | null
  regenerationCount: number
  audioActive: boolean // audio currently generated/playing
}

/** Derives the seven pipeline node states from real, already-fetched app state. */
export function derivePipelineSteps({ hasDoc, busy, error, resultStatus, summaryUsable, validationState, regenerationCount, audioActive }: PipelineInputs): PipelineStep[] {
  const summarized = summaryUsable ?? (resultStatus === 'validated' || resultStatus === 'partial' || resultStatus === 'translation_unavailable')
  const validationComplete = summarized && (validationState === 'VALIDATED' || validationState === 'NO_CLAIMS' || (!validationState && resultStatus === 'validated'))
  const failedResult = resultStatus === 'failed' || (!!resultStatus && summaryUsable === false)
  const failedProcessing = error && hasDoc && !resultStatus && !busy
  const analyzing = busy && hasDoc

  const documentState: NodeState = hasDoc ? 'completed' : 'idle'
  const analysisState: NodeState = !hasDoc
    ? 'idle'
    : summarized
      ? 'completed'
      : failedResult || failedProcessing
        ? 'failed'
        : analyzing
          ? 'processing'
          : 'idle'
  const chunkingState: NodeState = hasDoc ? 'completed' : 'idle'
  const summarizationState: NodeState = summarized ? 'completed' : failedResult || failedProcessing ? 'failed' : analyzing ? 'processing' : 'idle'
  const validationNodeState: NodeState = validationComplete
    ? 'completed'
    : validationState === 'PARTIALLY_VALIDATED' || validationState === 'VALIDATION_FAILED_AFTER_RETRIES' || validationState === 'VALIDATION_UNAVAILABLE' || failedResult
      ? 'failed'
      : regenerationCount > 0
        ? 'processing'
        : analyzing
          ? 'processing'
          : 'idle'
  const qualityState: NodeState = validationComplete ? 'completed' : summarized ? 'failed' : failedResult ? 'failed' : 'idle'
  const audioState: NodeState = audioActive ? 'completed' : 'idle'

  return [
    { label: 'Document', state: documentState },
    { label: 'Analysis', state: analysisState },
    { label: 'Chunking', state: chunkingState },
    { label: 'Summarization', state: summarizationState },
    { label: 'Validation', state: validationNodeState },
    { label: 'Quality', state: qualityState },
    { label: 'Audio', state: audioState },
  ]
}
