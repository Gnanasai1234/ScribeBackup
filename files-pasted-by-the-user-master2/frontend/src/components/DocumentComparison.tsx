import * as React from 'react'
import { useMemo, useState } from 'react'
import {
  AlertCircle,
  ChevronDown,
  ChevronUp,
  FileText,
  FolderOpen,
  Sparkles,
  Upload,
} from 'lucide-react'
import { CompareVisualization } from './three/CompareVisualization'
import { API, apiFetch as fetch } from '../lib/api'

export type Doc = {
  id: string
  name: string
  language: string
  status: string
  created_at?: string
  ocr_detected?: boolean
  ocr_warnings?: string[]
  stats: { words: number; chunks: number; pages: number | null }
  dataset?: { name: string; split: string; record_id: string }
}

export function DocumentComparison({ docs }: { docs?: Doc[] }) {
  const [a, setA] = useState<File | null>(null)
  const [b, setB] = useState<File | null>(null)
  const [selectedDocA, setSelectedDocA] = useState<string>('')
  const [selectedDocB, setSelectedDocB] = useState<string>('')
  const [useExisting, setUseExisting] = useState<boolean>(false)
  const [busy, setBusy] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [data, setData] = useState<any>(null)
  const [activeTab, setActiveTab] = useState<string>('ALL')
  const [expandedContext, setExpandedContext] = useState<Record<string, boolean>>({})

  const toggleContext = (id: string) => {
    setExpandedContext(prev => ({ ...prev, [id]: !prev[id] }))
  }

  async function run() {
    if (useExisting) {
      if (!selectedDocA || !selectedDocB) return
      if (selectedDocA === selectedDocB) {
        setError('Please select two distinct documents to compare.')
        return
      }
    } else {
      if (!a || !b) return
    }

    setBusy(true)
    setError('')
    try {
      const fd = new FormData()
      if (useExisting) {
        fd.append('doc_id_a', selectedDocA)
        fd.append('doc_id_b', selectedDocB)
      } else {
        if (a) fd.append('file_a', a)
        if (b) fd.append('file_b', b)
      }
      const r = await fetch(`${API}/documents/compare`, { method: 'POST', body: fd })
      const res = await r.json()
      if (!r.ok) throw new Error(res.detail || 'Comparison failed')
      setData(res)
    } catch (err: any) {
      setError(err.message || 'Comparison request failed')
    } finally {
      setBusy(false)
    }
  }

  const items = data?.comparison_items || []
  const filteredItems = useMemo(() => {
    if (!items.length) return []
    if (activeTab === 'ALL') return items
    if (activeTab === 'MODIFIED') return items.filter((x: any) => x.status === 'MODIFIED')
    if (activeTab === 'ADDED') return items.filter((x: any) => x.status === 'ADDED')
    if (activeTab === 'REMOVED') return items.filter((x: any) => x.status === 'REMOVED')
    if (activeTab === 'MOVED') return items.filter((x: any) => x.status === 'MOVED')
    if (activeTab === 'UNCHANGED') {
      return items.filter(
        (x: any) => x.status === 'UNCHANGED' || x.status === 'TRANSLATED/EQUIVALENT'
      )
    }
    if (activeTab === 'UNCERTAIN') return items.filter((x: any) => x.status === 'UNCERTAIN')
    return items
  }, [items, activeTab])

  const stats = data?.statistics || {
    total: items.length,
    unchanged: 0,
    modified: 0,
    added: 0,
    removed: 0,
    moved: 0,
    uncertain: 0,
  }

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case 'MODIFIED':
        return 'status-badge-modified'
      case 'ADDED':
        return 'status-badge-added'
      case 'REMOVED':
        return 'status-badge-removed'
      case 'MOVED':
        return 'status-badge-moved'
      case 'UNCHANGED':
        return 'status-badge-unchanged'
      case 'TRANSLATED/EQUIVALENT':
        return 'status-badge-translated'
      default:
        return 'status-badge-uncertain'
    }
  }

  const getItemCardBorderClass = (status: string) => {
    switch (status) {
      case 'MODIFIED':
        return 'is-modified'
      case 'ADDED':
        return 'is-added'
      case 'REMOVED':
        return 'is-removed'
      case 'MOVED':
        return 'is-moved'
      case 'UNCHANGED':
        return 'is-unchanged'
      default:
        return ''
    }
  }

  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <span /> CONTEXT-AWARE INTELLIGENCE
          </div>
          <h1>Compare documents</h1>
          <p>
            Context-aware semantic comparison understanding additions, removals, modifications,
            and value changes.
          </p>
        </div>
      </div>

      <div className="compare-card">
        {/* Mode Toggle: Upload New vs Workspace Docs */}
        {docs && docs.length >= 2 && (
          <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
            <button
              type="button"
              className={`secondary-btn ${!useExisting ? 'active-btn' : ''}`}
              style={{
                background: !useExisting ? '#1a3325' : '#101d16',
                borderColor: !useExisting ? '#51cc87' : '#22382c',
              }}
              onClick={() => setUseExisting(false)}
            >
              <Upload size={14} /> Upload new files
            </button>
            <button
              type="button"
              className={`secondary-btn ${useExisting ? 'active-btn' : ''}`}
              style={{
                background: useExisting ? '#1a3325' : '#101d16',
                borderColor: useExisting ? '#51cc87' : '#22382c',
              }}
              onClick={() => setUseExisting(true)}
            >
              <FolderOpen size={14} /> Choose from workspace ({docs.length} docs)
            </button>
          </div>
        )}

        {useExisting ? (
          <div className="compare-inputs">
            <div className="compare-drop" style={{ cursor: 'default' }}>
              <b>Document A (Original / Base)</b>
              <select
                style={{
                  width: '90%',
                  padding: '8px 10px',
                  background: '#0a1410',
                  border: '1px solid #233b2e',
                  borderRadius: 6,
                  color: '#e8efeb',
                  fontSize: 11,
                  marginTop: 8,
                }}
                value={selectedDocA}
                onChange={e => setSelectedDocA(e.target.value)}
              >
                <option value="">Select Document A…</option>
                {docs?.map(d => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.language.toUpperCase()})
                  </option>
                ))}
              </select>
            </div>
            <div className="compare-drop" style={{ cursor: 'default' }}>
              <b>Document B (Comparison / Revision)</b>
              <select
                style={{
                  width: '90%',
                  padding: '8px 10px',
                  background: '#0a1410',
                  border: '1px solid #233b2e',
                  borderRadius: 6,
                  color: '#e8efeb',
                  fontSize: 11,
                  marginTop: 8,
                }}
                value={selectedDocB}
                onChange={e => setSelectedDocB(e.target.value)}
              >
                <option value="">Select Document B…</option>
                {docs?.map(d => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.language.toUpperCase()})
                  </option>
                ))}
              </select>
            </div>
          </div>
        ) : (
          <div className="compare-inputs">
            {(
              [
                [a, setA, 'Version A (Original)'],
                [b, setB, 'Version B (Comparison)'],
              ] as const
            ).map(([f, set, label]: any) => (
              <label key={label} className="compare-drop">
                <Upload size={20} />
                <b>{f?.name || `Add ${label}`}</b>
                <small>PDF, DOCX, TXT, CSV or JSON · 25 MB max</small>
                <input
                  type="file"
                  accept=".pdf,.docx,.txt,.csv,.json"
                  onChange={e => set(e.target.files?.[0] || null)}
                />
              </label>
            ))}
          </div>
        )}

        {error && (
          <div
            className="config-note"
            style={{
              borderLeftColor: '#ef4444',
              background: 'rgba(239, 68, 68, 0.1)',
              color: '#fca5a5',
              marginTop: 12,
            }}
          >
            <AlertCircle size={16} color="#ef4444" />
            <span>
              <b>Notice:</b> {error}
            </span>
          </div>
        )}

        <button
          className="primary-btn"
          style={{ marginTop: 14 }}
          disabled={(useExisting ? !selectedDocA || !selectedDocB : !a || !b) || busy}
          onClick={run}
        >
          {busy ? (
            <>
              <span className="spinner" /> Analyzing semantic differences…
            </>
          ) : (
            <>
              <Sparkles size={16} /> Compare documents →
            </>
          )}
        </button>

        {data && (
          <div className="compare-container">
            {/* Visualizer */}
            {data.added && data.removed && (
              <CompareVisualization added={data.added} removed={data.removed} />
            )}

            {/* Document Metadata Cards */}
            {data.document_a && data.document_b && (
              <div className="compare-meta-grid">
                <div className="compare-doc-card">
                  <div className="compare-doc-header">
                    <div className="file-icon">
                      <FileText size={16} />
                    </div>
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <span className="compare-doc-tag">DOCUMENT A (BASE)</span>
                      <div className="compare-doc-title" title={data.document_a.name}>
                        {data.document_a.name}
                      </div>
                    </div>
                  </div>
                  <div className="compare-doc-badges">
                    <span className="compare-pill">{data.document_a.type}</span>
                    <span className="compare-pill">{data.document_a.language}</span>
                    <span className="compare-pill">
                      {data.document_a.word_count.toLocaleString()} words
                    </span>
                    <span className="compare-pill">
                      {data.document_a.section_count} sections
                    </span>
                    {data.document_a.page_count && (
                      <span className="compare-pill">{data.document_a.page_count} pages</span>
                    )}
                  </div>
                </div>

                <div className="compare-doc-card doc-b">
                  <div className="compare-doc-header">
                    <div
                      className="file-icon"
                      style={{ background: '#1c3d2a', color: '#8cf0b7' }}
                    >
                      <FileText size={16} />
                    </div>
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <span className="compare-doc-tag" style={{ color: '#8cf0b7' }}>
                        DOCUMENT B (TARGET)
                      </span>
                      <div className="compare-doc-title" title={data.document_b.name}>
                        {data.document_b.name}
                      </div>
                    </div>
                  </div>
                  <div className="compare-doc-badges">
                    <span className="compare-pill">{data.document_b.type}</span>
                    <span className="compare-pill">{data.document_b.language}</span>
                    <span className="compare-pill">
                      {data.document_b.word_count.toLocaleString()} words
                    </span>
                    <span className="compare-pill">
                      {data.document_b.section_count} sections
                    </span>
                    {data.document_b.page_count && (
                      <span className="compare-pill">{data.document_b.page_count} pages</span>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Low Semantic Overlap Warning */}
            {(data.is_unrelated || data.topic_overlap === 'LOW') && (
              <div className="compare-alert-unrelated">
                <AlertCircle size={20} />
                <div>
                  <b>Low Semantic Overlap Detected</b>
                  Document A and Document B address fundamentally different subject domains.
                  Content is classified across separate domains rather than direct editorial
                  modifications.
                </div>
              </div>
            )}

            {/* Executive Comparison Summary Card */}
            <div className="compare-exec-card">
              <h3>Executive Comparison Summary</h3>
              <p>{data.executive_summary}</p>

              <div className="compare-stats-pills">
                <span className="stat-pill">
                  <b>{stats.total}</b> Total points
                </span>
                <span className="stat-pill">
                  <b>{stats.unchanged}</b> Unchanged
                </span>
                <span className="stat-pill modified">
                  <b>{stats.modified}</b> Modified
                </span>
                <span className="stat-pill added">
                  <b>{stats.added}</b> Added
                </span>
                <span className="stat-pill removed">
                  <b>{stats.removed}</b> Removed
                </span>
                {stats.moved > 0 && (
                  <span className="stat-pill moved">
                    <b>{stats.moved}</b> Repositioned
                  </span>
                )}
                {stats.uncertain > 0 && (
                  <span className="stat-pill">
                    <b>{stats.uncertain}</b> Uncertain
                  </span>
                )}
              </div>
            </div>

            {/* Important Factual Changes Highlight */}
            {data.important_changes && data.important_changes.length > 0 && (
              <div className="compare-important-card">
                <h4>
                  <Sparkles size={15} /> Key Factual & Value Differences
                </h4>
                <ul className="compare-important-list">
                  {data.important_changes.map((changeText: string, i: number) => (
                    <li key={i}>
                      <span className="dot" />
                      <span>{changeText}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Section Summary Pills if available */}
            {data.sections && data.sections.length > 0 && (
              <div
                style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: 6,
                  alignItems: 'center',
                }}
              >
                <span style={{ font: '9px "DM Mono", monospace', color: '#7f9187' }}>
                  SECTIONS:
                </span>
                {data.sections.map((sec: any, idx: number) => (
                  <span
                    key={idx}
                    className={`compare-status-badge ${getStatusBadgeClass(sec.status)}`}
                    style={{ fontSize: 9 }}
                  >
                    {sec.section_name} → {sec.status}
                  </span>
                ))}
              </div>
            )}

            {/* Interactive Filter Tabs */}
            <div className="compare-tabs">
              {[
                ['ALL', 'All', stats.total],
                ['MODIFIED', 'Modified', stats.modified],
                ['ADDED', 'Added', stats.added],
                ['REMOVED', 'Removed', stats.removed],
                ['MOVED', 'Repositioned', stats.moved],
                ['UNCHANGED', 'Unchanged', stats.unchanged],
              ].map(([tabKey, label, count]) => (
                <button
                  key={tabKey}
                  type="button"
                  className={`compare-tab ${activeTab === tabKey ? 'active' : ''}`}
                  onClick={() => setActiveTab(tabKey)}
                >
                  <span>{label}</span>
                  <span className="compare-tab-count">{count}</span>
                </button>
              ))}
            </div>

            {/* Filtered Comparison Items */}
            <div style={{ display: 'grid', gap: 14 }}>
              {filteredItems.length === 0 ? (
                <div
                  style={{
                    textAlign: 'center',
                    padding: '30px 10px',
                    color: '#7a8e82',
                    fontSize: 12,
                  }}
                >
                  No items in the "{activeTab}" category.
                </div>
              ) : (
                filteredItems.map((item: any, idx: number) => {
                  const itemId = item.id || `item-${idx}`
                  const isExpanded = !!expandedContext[itemId]

                  return (
                    <div
                      key={itemId}
                      className={`compare-item-card ${getItemCardBorderClass(item.status)}`}
                    >
                      {/* Item Header */}
                      <div className="compare-item-head">
                        <span className="compare-item-title">
                          {item.section_name || 'General Content'}
                        </span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          {item.confidence && (
                            <span className="compare-confidence">
                              {item.confidence} confidence
                            </span>
                          )}
                          <span
                            className={`compare-status-badge ${getStatusBadgeClass(
                              item.status
                            )}`}
                          >
                            {item.status}
                          </span>
                        </div>
                      </div>

                      {/* Side-by-Side Content */}
                      <div className="compare-columns">
                        {/* Column A */}
                        <div className="compare-col">
                          <div className="compare-col-header">
                            <span>DOCUMENT A</span>
                            {item.document_a?.page && <span>Page {item.document_a.page}</span>}
                          </div>
                          {item.document_a?.text ? (
                            <p
                              className={`compare-col-text ${
                                item.status === 'REMOVED' ? 'dimmed' : ''
                              }`}
                            >
                              {item.document_a.text}
                            </p>
                          ) : (
                            <p className="compare-col-text empty-text">
                              — Not present in Document A —
                            </p>
                          )}
                        </div>

                        {/* Column B */}
                        <div
                          className="compare-col"
                          style={{
                            borderColor:
                              item.status === 'ADDED' || item.status === 'MODIFIED'
                                ? '#264e37'
                                : '#1a2a22',
                          }}
                        >
                          <div className="compare-col-header">
                            <span
                              style={{
                                color:
                                  item.status === 'ADDED' || item.status === 'MODIFIED'
                                    ? '#78e2a3'
                                    : '#7f9187',
                              }}
                            >
                              DOCUMENT B
                            </span>
                            {item.document_b?.page && <span>Page {item.document_b.page}</span>}
                          </div>
                          {item.document_b?.text ? (
                            <p className="compare-col-text">{item.document_b.text}</p>
                          ) : (
                            <p className="compare-col-text empty-text">
                              — Not present in Document B —
                            </p>
                          )}
                        </div>
                      </div>

                      {/* Specific Entity / Value Changes */}
                      {item.specific_changes && item.specific_changes.length > 0 && (
                        <div style={{ display: 'grid', gap: 6 }}>
                          {item.specific_changes.map((diff: any, di: number) => (
                            <div key={di} className="compare-value-diff">
                              <span>
                                <b>{diff.type.replace('_', ' ')}:</b>
                              </span>
                              <span>{diff.old_value}</span>
                              <span style={{ color: '#74e3a5' }}>→</span>
                              <span style={{ color: '#8ef5bf', fontWeight: 600 }}>
                                {diff.new_value}
                              </span>
                              <span
                                style={{
                                  color: '#97ab9e',
                                  marginLeft: 'auto',
                                  fontSize: 10,
                                }}
                              >
                                {diff.explanation}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Human Explanation */}
                      {item.explanation && (
                        <p className="compare-explanation">
                          <span style={{ color: '#74e3a5', fontWeight: 600, marginRight: 6 }}>
                            Context:
                          </span>
                          {item.explanation}
                        </p>
                      )}

                      {/* Expandable Surrounding Context Drawer */}
                      {(item.document_a?.context_before ||
                        item.document_a?.context_after ||
                        item.document_b?.context_before ||
                        item.document_b?.context_after) && (
                        <div>
                          <button
                            type="button"
                            className="compare-context-btn"
                            onClick={() => toggleContext(itemId)}
                          >
                            {isExpanded ? (
                              <>
                                <ChevronUp size={13} /> Hide surrounding context
                              </>
                            ) : (
                              <>
                                <ChevronDown size={13} /> Show surrounding context
                              </>
                            )}
                          </button>

                          {isExpanded && (
                            <div className="compare-context-drawer">
                              {item.document_a?.text && (
                                <div style={{ marginBottom: 8 }}>
                                  <small>Document A Context</small>
                                  {item.document_a.context_before && (
                                    <div style={{ opacity: 0.65 }}>
                                      … {item.document_a.context_before}
                                    </div>
                                  )}
                                  <div
                                    style={{
                                      color: '#e8efeb',
                                      fontWeight: 500,
                                      margin: '2px 0',
                                    }}
                                  >
                                    {item.document_a.text}
                                  </div>
                                  {item.document_a.context_after && (
                                    <div style={{ opacity: 0.65 }}>
                                      {item.document_a.context_after} …
                                    </div>
                                  )}
                                </div>
                              )}

                              {item.document_b?.text && (
                                <div>
                                  <small style={{ color: '#8ceab4' }}>Document B Context</small>
                                  {item.document_b.context_before && (
                                    <div style={{ opacity: 0.65 }}>
                                      … {item.document_b.context_before}
                                    </div>
                                  )}
                                  <div
                                    style={{
                                      color: '#e8efeb',
                                      fontWeight: 500,
                                      margin: '2px 0',
                                    }}
                                  >
                                    {item.document_b.text}
                                  </div>
                                  {item.document_b.context_after && (
                                    <div style={{ opacity: 0.65 }}>
                                      {item.document_b.context_after} …
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )
                })
              )}
            </div>

            {/* Technical Method Footer */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                color: '#687c71',
                fontSize: 9,
                marginTop: 10,
              }}
            >
              <span>Method: {data.method}</span>
              <span>{data.limitations}</span>
            </div>
          </div>
        )}
      </div>
    </>
  )
}
