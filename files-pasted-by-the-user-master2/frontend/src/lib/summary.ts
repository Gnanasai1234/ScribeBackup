export const MIN_USABLE_SUMMARY_CHARS = 12

export function summaryText(value: unknown): string {
  if (typeof value === 'string') {
    const text = value.trim()
    if (text.startsWith('{')) {
      try {
        const parsed: unknown = JSON.parse(text)
        if (typeof parsed === 'object' && parsed !== null) return summaryText(parsed)
      } catch {
        return text
      }
    }
    return text
  }
  if (Array.isArray(value)) return value.map(summaryText).filter(Boolean).join(' ')
  if (typeof value === 'object' && value !== null) {
    const data = value as Record<string, unknown>
    return ['executive_summary', 'summary', 'overview', 'text', 'key_findings', 'findings', 'facts', 'risks', 'decisions', 'recommendations', 'action_items']
      .filter(key => key in data)
      .map(key => summaryText(data[key]))
      .filter(Boolean)
      .join(' ')
  }
  return ''
}

export function isUsableSummary(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length >= MIN_USABLE_SUMMARY_CHARS
}
