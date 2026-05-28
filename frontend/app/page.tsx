"use client";

import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import {
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  ReactFlowProvider,
  type Edge,
  type Node,
  type NodeProps,
  type OnSelectionChangeParams,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const TYPE_COLORS: Record<string, string> = {
  PERSON: "#60a5fa",
  ORG: "#a78bfa",
  PRODUCT: "#34d399",
  TECH: "#fbbf24",
  CONCEPT: "#f472b6",
  LOCATION: "#38bdf8",
  OTHER: "#94a3b8",
};

type GraphNode = { id: string; label: string; type: string };
type GraphEdge = {
  id: string;
  source: string;
  target: string;
  label: string;
  evidence: string;
};
type GraphResponse = { nodes: GraphNode[]; edges: GraphEdge[] };

type SourceItem = {
  edge_id: string;
  source: string;
  target: string;
  relationship: string;
  evidence: string;
};

type ChunkSourceItem = {
  chunk_id: string;
  page: number | null;
  text: string;
};

type SourcesResponse = {
  graph_edges: SourceItem[];
  chunks: ChunkSourceItem[];
};

type CitationItem = {
  citation_id: number;
  source_type: string;
  source_id: string;
  page: number | null;
  text: string;
};

type AskMetrics = {
  cost_usd?: number;
  stages_ms?: Record<string, number>;
  groq_tokens?: { prompt: number; completion: number };
};

type AskResponse = {
  answer: string;
  citations: CitationItem[];
  citation_valid: boolean;
  sources: SourcesResponse;
  metrics?: AskMetrics;
};

type ChatMessage = {
  id: string;
  question: string;
  answer: string;
  citations: CitationItem[];
  citationValid: boolean;
  graphSources: SourceItem[];
  chunkSources: ChunkSourceItem[];
  metrics?: AskMetrics;
};

type EntityNodeData = { label: string; type: string };

function EntityNode({ data }: NodeProps<Node<EntityNodeData>>) {
  const color = TYPE_COLORS[data.type] || TYPE_COLORS.OTHER;
  return (
    <div
      style={{
        background: "#151b24",
        border: `2px solid ${color}`,
        borderRadius: 8,
        padding: "8px 12px",
        minWidth: 120,
        color: "#e8edf5",
        fontSize: 12,
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: color }} />
      <div style={{ fontWeight: 600 }}>{data.label}</div>
      <div style={{ fontSize: 10, color: "#94a3b8", marginTop: 4 }}>{data.type}</div>
      <Handle type="source" position={Position.Bottom} style={{ background: color }} />
    </div>
  );
}

const nodeTypes = { entity: EntityNode };

function CollapsibleSources({
  title,
  count,
  children,
}: {
  title: string;
  count: number;
  children: ReactNode;
}) {
  if (count === 0) return null;
  return (
    <details style={styles.sourcesDetails}>
      <summary style={styles.sourcesSummary}>
        {title} ({count})
      </summary>
      <div style={styles.sourcesBody}>{children}</div>
    </details>
  );
}

function layoutNodes(graphNodes: GraphNode[]): Node<EntityNodeData>[] {
  const cols = Math.ceil(Math.sqrt(graphNodes.length)) || 1;
  return graphNodes.map((n, i) => {
    const row = Math.floor(i / cols);
    const col = i % cols;
    return {
      id: n.id,
      type: "entity",
      position: { x: col * 240, y: row * 140 },
      data: { label: n.label, type: n.type },
    };
  });
}

function formatFetchError(err: unknown, apiUrl: string, fallback: string): string {
  if (err instanceof TypeError && err.message === "Failed to fetch") {
    return `Cannot reach the backend at ${apiUrl}. Start it with: cd backend && uvicorn main:app --reload --port 8000`;
  }
  return err instanceof Error ? err.message : fallback;
}

function formatRequestMetrics(metrics?: AskMetrics): string | null {
  if (!metrics) return null;
  const stages = metrics.stages_ms ?? {};
  const totalMs = Object.values(stages).reduce((a, b) => a + b, 0);
  const cost = metrics.cost_usd;
  const parts: string[] = [];
  if (totalMs > 0) parts.push(`${Math.round(totalMs)}ms`);
  if (cost != null && cost > 0) parts.push(`$${cost.toFixed(5)}`);
  return parts.length ? parts.join(" · ") : null;
}

function formatApiError(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => (typeof d === "object" && d && "msg" in d ? String((d as { msg: string }).msg) : String(d))).join("; ");
  }
  return fallback;
}

function toFlowEdges(graphEdges: GraphEdge[]): Edge[] {
  return graphEdges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    label: e.label,
    animated: true,
    style: { stroke: "#64748b" },
    labelStyle: { fill: "#94a3b8", fontSize: 10 },
    data: { evidence: e.evidence },
  }));
}

function GraphCanvas({
  nodes,
  edges,
  onSelectionChange,
}: {
  nodes: Node<EntityNodeData>[];
  edges: Edge[];
  onSelectionChange: (params: OnSelectionChangeParams) => void;
}) {
  return (
    <div style={{ width: "100%", height: "100%" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onSelectionChange={onSelectionChange}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#1e293b" gap={20} />
        <Controls />
        <MiniMap
          title="Overview"
          position="bottom-right"
          style={{ width: 140, height: 100, background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }}
          bgColor="#111820"
          nodeColor={(n) =>
            TYPE_COLORS[String((n.data as EntityNodeData)?.type)] || TYPE_COLORS.OTHER
          }
          nodeStrokeWidth={2}
          maskColor="rgba(11, 15, 20, 0.75)"
          maskStrokeColor="#64748b"
          zoomable
          pannable
        />
      </ReactFlow>
    </div>
  );
}

export default function HomePage() {
  const questionInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [nodes, setNodes] = useState<Node<EntityNodeData>[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [status, setStatus] = useState<string>("Upload a PDF to get started.");
  const [statusIsError, setStatusIsError] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [documentReady, setDocumentReady] = useState(false);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [selected, setSelected] = useState<{
    kind: "node" | "edge";
    data: Record<string, string>;
  } | null>(null);

  const onSelectionChange = useCallback(
    ({ nodes: selNodes, edges: selEdges }: OnSelectionChangeParams) => {
      if (selNodes.length === 1) {
        const n = selNodes[0];
        const data = n.data as EntityNodeData;
        setSelected({
          kind: "node",
          data: {
            id: n.id,
            label: String(data.label ?? ""),
            type: String(data.type ?? "OTHER"),
          },
        });
        return;
      }
      if (selEdges.length === 1) {
        const e = selEdges[0];
        setSelected({
          kind: "edge",
          data: {
            id: e.id,
            source: e.source,
            target: e.target,
            label: String(e.label ?? ""),
            evidence: String((e.data as { evidence?: string })?.evidence ?? ""),
          },
        });
        return;
      }
      setSelected(null);
    },
    []
  );

  const applyGraph = useCallback((graph: GraphResponse) => {
    setNodes(layoutNodes(graph.nodes));
    setEdges(toFlowEdges(graph.edges));
    setSelected(null);
  }, []);

  const loadGraph = useCallback(async (options?: { quiet?: boolean }) => {
    if (!options?.quiet) {
      setStatusIsError(false);
      setStatus("Loading graph…");
    }
    try {
      const res = await fetch(`${API_URL}/graph`);
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        if (res.status === 404) {
          setNodes([]);
          setEdges([]);
          setSelected(null);
          setStatusIsError(false);
          setStatus("No graph yet — upload a PDF to get started.");
          return false;
        }
        throw new Error(formatApiError(body.detail, `Failed to load graph (${res.status})`));
      }
      const graph = body as GraphResponse;
      applyGraph(graph);
      if (graph.nodes.length === 0 && graph.edges.length === 0) {
        setStatusIsError(true);
        setStatus(
          "Graph is empty. Upload a PDF (needs a valid GROQ_API_KEY in backend/.env)."
        );
      } else {
        setStatusIsError(false);
        setStatus(
          `Ready — ${graph.nodes.length} nodes, ${graph.edges.length} edges. Ask a question or explore the graph.`
        );
      }
      return true;
    } catch (err) {
      setStatusIsError(true);
      setStatus(formatFetchError(err, API_URL, "Failed to load graph"));
      return false;
    }
  }, [applyGraph]);

  const uploadPdf = useCallback(
    async (file: File) => {
      setUploading(true);
      setStatusIsError(false);
      setStatus(`Processing ${file.name}… (this may take a minute)`);
      const form = new FormData();
      form.append("file", file);

      try {
        const res = await fetch(`${API_URL}/upload`, {
          method: "POST",
          body: form,
        });
        const body = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(formatApiError(body.detail, `Upload failed (${res.status})`));
        }
        if (body.nodes === 0 && body.edges === 0) {
          throw new Error(
            "Graph is empty after upload. Set a valid GROQ_API_KEY in backend/.env and restart the backend."
          );
        }
        setChatMessages([]);
        setDocumentReady(true);
        setStatusIsError(false);
        setStatus(
          `Processed ${body.chunks_processed} chunks → ${body.nodes} nodes, ${body.edges} edges. Loading graph…`
        );
        const uploadMetrics = body.metrics as AskMetrics | undefined;
        const uploadMeta = formatRequestMetrics(uploadMetrics);
        await loadGraph({ quiet: true });
        setStatus(
          `Ready — ${body.nodes} nodes, ${body.edges} edges from ${file.name}.${uploadMeta ? ` (${uploadMeta})` : ""} Ask a question below.`
        );
      } catch (err) {
        setStatusIsError(true);
        setStatus(formatFetchError(err, API_URL, "Upload failed"));
      } finally {
        setUploading(false);
      }
    },
    [loadGraph]
  );

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        uploadPdf(file);
      }
      e.target.value = "";
    },
    [uploadPdf]
  );

  const askQuestion = useCallback(async () => {
    const q = question.trim();
    if (!q || asking) return;

    const historyForApi = chatMessages.slice(-6).map((m) => ({
      question: m.question,
      answer: m.answer,
    }));

    setQuestion("");
    setAsking(true);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, history: historyForApi }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(formatApiError(body.detail, `Ask failed (${res.status})`));
      }
      const result = body as AskResponse;
      setChatMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-${prev.length}`,
          question: q,
          answer: result.answer,
          citations: result.citations ?? [],
          citationValid: result.citation_valid ?? true,
          graphSources: result.sources?.graph_edges ?? [],
          chunkSources: result.sources?.chunks ?? [],
          metrics: result.metrics,
        },
      ]);
    } catch (err) {
      setChatMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-${prev.length}`,
          question: q,
          answer: formatFetchError(err, API_URL, "Failed to get an answer"),
          citations: [],
          citationValid: false,
          graphSources: [],
          chunkSources: [],
        },
      ]);
    } finally {
      setAsking(false);
      questionInputRef.current?.focus();
    }
  }, [question, asking, chatMessages]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, asking]);

  const hasGraph = documentReady && nodes.length > 0;

  const legend = useMemo(
    () =>
      Object.entries(TYPE_COLORS).map(([type, color]) => (
        <span
          key={type}
          style={{ display: "inline-flex", alignItems: "center", gap: 6, marginRight: 12 }}
        >
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: color,
              display: "inline-block",
            }}
          />
          {type}
        </span>
      )),
    []
  );

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <div>
          <h1 style={styles.title}>Ask My Docs</h1>
          <p style={styles.subtitle}>
            Hybrid Graph RAG · BM25 + vector search · cross-encoder reranking · cited answers
          </p>
        </div>
        <div style={styles.actions}>
          <label
            style={{
              ...styles.btnPrimary,
              ...(uploading ? styles.btnDisabled : {}),
            }}
          >
            {uploading ? "Processing…" : "Upload PDF"}
            <input
              type="file"
              accept="application/pdf,.pdf"
              onChange={onFileChange}
              disabled={uploading}
            />
          </label>
        </div>
      </header>

      <p
        style={{
          ...styles.status,
          ...(statusIsError ? styles.statusError : {}),
        }}
      >
        {status}
      </p>
      <div style={styles.legend}>{legend}</div>

      <section style={styles.qaSection}>
        {(chatMessages.length > 0 || asking) && (
          <div style={styles.chatHistory}>
            {chatMessages.map((msg) => (
              <div key={msg.id} style={styles.chatTurn}>
                <div style={styles.userBubble}>{msg.question}</div>
                <div style={styles.answerCard}>
                  <p style={styles.answerText}>{msg.answer}</p>
                  {formatRequestMetrics(msg.metrics) && (
                    <p style={styles.metricsLine}>{formatRequestMetrics(msg.metrics)}</p>
                  )}
                  {!msg.citationValid && (
                    <p style={styles.citationWarning}>Citation check: some claims may lack valid [n] markers.</p>
                  )}
                  <CollapsibleSources title="Citations" count={msg.citations.length}>
                    <ul style={styles.sourcesList}>
                      {msg.citations.map((c) => (
                        <li key={c.citation_id} style={styles.sourceItem}>
                          <span style={styles.sourceRel}>[{c.citation_id}] {c.source_type}</span>
                          {c.page != null && (
                            <span style={styles.sourceType}> · page {c.page}</span>
                          )}
                          <p style={styles.sourceEvidence}>&ldquo;{c.text.slice(0, 300)}{c.text.length > 300 ? "…" : ""}&rdquo;</p>
                        </li>
                      ))}
                    </ul>
                  </CollapsibleSources>
                  <CollapsibleSources title="Graph sources" count={msg.graphSources.length}>
                    <ul style={styles.sourcesList}>
                      {msg.graphSources.map((s) => (
                        <li key={s.edge_id} style={styles.sourceItem}>
                          <span style={styles.sourceRel}>
                            {s.source} → {s.target}
                          </span>
                          <span style={styles.sourceType}> ({s.relationship})</span>
                          {s.evidence && (
                            <p style={styles.sourceEvidence}>&ldquo;{s.evidence}&rdquo;</p>
                          )}
                        </li>
                      ))}
                    </ul>
                  </CollapsibleSources>
                  <CollapsibleSources title="Text sources" count={msg.chunkSources.length}>
                    <ul style={styles.sourcesList}>
                      {msg.chunkSources.map((s) => (
                        <li key={s.chunk_id} style={styles.sourceItem}>
                          <span style={styles.sourceRel}>
                            {s.chunk_id}
                            {s.page != null ? ` · page ${s.page}` : ""}
                          </span>
                          {s.text && <p style={styles.sourceEvidence}>&ldquo;{s.text}&rdquo;</p>}
                        </li>
                      ))}
                    </ul>
                  </CollapsibleSources>
                </div>
              </div>
            ))}
            {asking && (
              <div style={styles.answerCard}>
                <p style={{ ...styles.answerText, color: "#94a3b8" }}>Thinking…</p>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        )}

        <div style={styles.qaRow}>
          <input
            ref={questionInputRef}
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (!asking && question.trim()) askQuestion();
              }
            }}
            placeholder="Ask a question about the document…"
            style={styles.qaInput}
            disabled={asking || uploading}
          />
          <button
            style={styles.btnAsk}
            disabled={asking || uploading || !documentReady || !question.trim()}
            onClick={askQuestion}
          >
            {asking ? "…" : "Ask"}
          </button>
        </div>
      </section>

      <div style={styles.main}>
        <div style={styles.canvas}>
          {hasGraph ? (
            <ReactFlowProvider>
              <GraphCanvas
                nodes={nodes}
                edges={edges}
                onSelectionChange={onSelectionChange}
              />
            </ReactFlowProvider>
          ) : (
            <div style={styles.empty}>
              <p>No graph yet.</p>
              <p style={{ color: "#64748b", fontSize: 14, textAlign: "center", maxWidth: 360 }}>
                Upload a PDF to extract entities and relationships. If upload &ldquo;succeeds&rdquo;
                but nothing appears, check backend/.env has a valid GROQ_API_KEY.
              </p>
            </div>
          )}
        </div>

        <aside style={styles.panel}>
          <h2 style={styles.panelTitle}>Details</h2>
          {selected ? (
            <dl style={styles.dl}>
              <dt style={styles.dt}>Kind</dt>
              <dd style={styles.dd}>{selected.kind}</dd>
              {Object.entries(selected.data).map(([key, value]) => (
                <Fragment key={key}>
                  <dt style={styles.dt}>{key}</dt>
                  <dd style={styles.dd}>{value || "—"}</dd>
                </Fragment>
              ))}
            </dl>
          ) : (
            <p style={{ color: "#64748b", fontSize: 14, lineHeight: 1.6 }}>
              Click a node or edge in the graph to inspect its properties.
            </p>
          )}
        </aside>
      </div>
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  page: {
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    padding: "20px 24px",
    gap: 12,
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    flexWrap: "wrap",
    gap: 16,
  },
  title: {
    margin: 0,
    fontSize: 28,
    fontWeight: 700,
    letterSpacing: "-0.02em",
  },
  subtitle: {
    margin: "6px 0 0",
    color: "#94a3b8",
    fontSize: 14,
  },
  actions: { display: "flex", gap: 10, alignItems: "center" },
  btnPrimary: {
    background: "#3b82f6",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    padding: "10px 16px",
    fontWeight: 600,
    display: "inline-flex",
    alignItems: "center",
    cursor: "pointer",
  },
  btnDisabled: {
    opacity: 0.6,
    cursor: "not-allowed",
  },
  btnSecondary: {
    background: "#1e293b",
    color: "#e8edf5",
    border: "1px solid #334155",
    borderRadius: 8,
    padding: "10px 16px",
    fontWeight: 600,
  },
  status: {
    margin: 0,
    fontSize: 13,
    color: "#94a3b8",
    padding: "10px 12px",
    borderRadius: 8,
    background: "#111820",
    border: "1px solid transparent",
  },
  statusError: {
    color: "#fecaca",
    background: "#2a1515",
    border: "1px solid #7f1d1d",
  },
  legend: {
    fontSize: 11,
    color: "#64748b",
  },
  qaSection: {
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  chatHistory: {
    display: "flex",
    flexDirection: "column",
    gap: 16,
    maxHeight: 280,
    overflowY: "auto",
    paddingRight: 4,
  },
  chatTurn: {
    display: "flex",
    flexDirection: "column",
    gap: 10,
  },
  userBubble: {
    alignSelf: "flex-end",
    maxWidth: "85%",
    background: "#1e3a5f",
    border: "1px solid #334155",
    borderRadius: "12px 12px 4px 12px",
    padding: "10px 14px",
    fontSize: 14,
    color: "#e8edf5",
    lineHeight: 1.5,
  },
  qaRow: {
    display: "flex",
    gap: 10,
    flexWrap: "wrap",
    alignItems: "stretch",
  },
  qaInput: {
    flex: 1,
    minWidth: 220,
    background: "#111820",
    border: "1px solid #334155",
    borderRadius: 8,
    padding: "10px 14px",
    color: "#e8edf5",
    fontSize: 14,
    outline: "none",
    minHeight: 42,
    boxSizing: "border-box",
  },
  btnAsk: {
    background: "#8b5cf6",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    padding: "10px 20px",
    fontWeight: 600,
    minHeight: 42,
    boxSizing: "border-box",
  },
  answerCard: {
    background: "#111820",
    border: "1px solid #1e293b",
    borderRadius: "12px 12px 12px 4px",
    padding: "14px 16px",
    maxWidth: "95%",
  },
  answerText: {
    margin: 0,
    fontSize: 15,
    lineHeight: 1.6,
    color: "#e8edf5",
  },
  citationWarning: {
    margin: "8px 0 0",
    fontSize: 12,
    color: "#fbbf24",
  },
  metricsLine: {
    margin: "6px 0 0",
    fontSize: 11,
    color: "#64748b",
    fontFamily: "ui-monospace, monospace",
  },
  sourcesDetails: {
    marginTop: 12,
    borderTop: "1px solid #1e293b",
    paddingTop: 8,
  },
  sourcesSummary: {
    fontSize: 12,
    fontWeight: 600,
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    cursor: "pointer",
    listStyle: "none",
    userSelect: "none",
  },
  sourcesBody: {
    marginTop: 10,
    maxHeight: 220,
    overflowY: "auto",
    paddingRight: 4,
  },
  sourcesList: {
    margin: 0,
    padding: 0,
    listStyle: "none",
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  sourceItem: {
    fontSize: 13,
    color: "#cbd5e1",
  },
  sourceRel: {
    fontWeight: 600,
    color: "#e8edf5",
  },
  sourceType: {
    color: "#94a3b8",
  },
  sourceEvidence: {
    margin: "6px 0 0",
    fontSize: 12,
    color: "#64748b",
    fontStyle: "italic",
    lineHeight: 1.5,
  },
  main: {
    flex: 1,
    display: "grid",
    gridTemplateColumns: "1fr 300px",
    gap: 16,
    minHeight: 0,
    height: "calc(100vh - 380px)",
  },
  canvas: {
    background: "#111820",
    border: "1px solid #1e293b",
    borderRadius: 12,
    overflow: "hidden",
    height: "100%",
    minHeight: 400,
  },
  empty: {
    height: "100%",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    color: "#94a3b8",
  },
  panel: {
    background: "#111820",
    border: "1px solid #1e293b",
    borderRadius: 12,
    padding: 16,
    overflow: "auto",
  },
  panelTitle: {
    margin: "0 0 12px",
    fontSize: 16,
    fontWeight: 600,
  },
  dl: {
    margin: 0,
    fontSize: 13,
    display: "grid",
    gridTemplateColumns: "72px 1fr",
    gap: "8px 12px",
    alignItems: "baseline",
  },
  dt: {
    margin: 0,
    color: "#64748b",
    fontWeight: 600,
    textTransform: "capitalize",
  },
  dd: {
    margin: 0,
    color: "#e8edf5",
    wordBreak: "break-word",
  },
};
