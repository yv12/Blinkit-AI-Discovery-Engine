import re

with open('web/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
'''    CheckCircle2,
    Send,
    Sparkles,
  } from "lucide-react";''',
'''    CheckCircle2,
    Send,
    Sparkles,
    Info,
  } from "lucide-react";'''
)

tabs_target = '''      <div className="dq-tabs dq-tabs-underline">
        <button className={"dq-tab" + (tab === "ask" ? " active" : "")} onClick={() => setTab("ask")}>
          <MessageSquare size={15} /> Ask
        </button>
        <button className={"dq-tab" + (tab === "narrative" ? " active" : "")} onClick={() => setTab("narrative")}>
          <BarChart3 size={15} /> Narrative
        </button>
      </div>'''

tabs_replacement = '''      <div className="dq-tabs dq-tabs-underline">
        <button className={"dq-tab" + (tab === "ask" ? " active" : "")} onClick={() => setTab("ask")}>
          <MessageSquare size={15} /> Ask
        </button>
        <button className={"dq-tab" + (tab === "narrative" ? " active" : "")} onClick={() => setTab("narrative")}>
          <BarChart3 size={15} /> Narrative
        </button>
        <button className={"dq-tab" + (tab === "methodology" ? " active" : "")} onClick={() => setTab("methodology")}>
          <Info size={15} /> Methodology
        </button>
      </div>'''

content = content.replace(tabs_target, tabs_replacement)

body_target = '''          </div>
        )}
      </div>
    </div>
  );
}

const root = createRoot(document.getElementById("root"));'''

body_replacement = '''          </div>
        )}
        
        {tab === "methodology" && (
          <div className="methodology-panel" style={{ padding: '24px 32px', overflowY: 'auto', maxHeight: '100%', paddingBottom: '60px' }}>
            <h2 className="narrative-heading display" style={{ marginBottom: '24px' }}>Data Methodology & Transparency Report</h2>
            <div style={{ maxWidth: '800px', fontSize: '15px', lineHeight: '1.6', color: 'var(--ink)' }}>
              <p>This document outlines the end-to-end data pipeline used to analyze user reviews and extract category discovery barriers. It provides complete transparency into how reviews were sourced, filtered, and categorized.</p>
              
              <h3 style={{ marginTop: '32px', marginBottom: '16px', fontSize: '18px' }}>1. Data Collection</h3>
              <p>To ensure a comprehensive understanding of user sentiment, we sourced raw reviews from multiple channels over a fixed, recent timeframe.</p>
              <ul style={{ paddingLeft: '20px', marginBottom: '16px' }}>
                <li><strong>Date Range:</strong> January 2026 to July 2026</li>
                <li><strong>Sources:</strong>
                  <ul style={{ paddingLeft: '20px', marginTop: '8px' }}>
                    <li><strong>Google Play Store:</strong> 52,980 reviews (Primary Source)</li>
                    <li><strong>MouthShut:</strong> 123 reviews (Supplementary Source)</li>
                  </ul>
                </li>
                <li><strong>Total Reviews Sourced:</strong> 53,103</li>
              </ul>

              <h3 style={{ marginTop: '32px', marginBottom: '16px', fontSize: '18px' }}>2. The Processing Funnel</h3>
              <p>Reviews are often multi-faceted, containing multiple distinct thoughts. The pipeline splits these reviews into granular "units" (sentences/clauses) and filters them through a strict funnel before they reach the final insights dashboard.</p>
              <ul style={{ paddingLeft: '20px', marginBottom: '16px' }}>
                <li><strong>Scraped (156,342 units):</strong> The raw number of distinct statements extracted from the 53,103 source reviews.</li>
                <li><strong>Cleaned (84,111 units | 54%):</strong> Units remaining after stripping out spam, gibberish, non-English text, and uninformative one-word reviews.</li>
                <li><strong>In Engine (72,566 units | 86%):</strong> The high-quality units that were successfully converted into semantic embeddings (vectorized) for AI analysis.</li>
                <li><strong>Relevant (12,706 units | 18%):</strong> The subset of units that formed strong, cohesive thematic clusters and were highly relevant to our specific business questions (e.g., category exploration and user friction).</li>
              </ul>

              <h3 style={{ marginTop: '32px', marginBottom: '16px', fontSize: '18px' }}>3. Categorization Logic (Two-Layer Approach)</h3>
              <p>Rather than relying on basic keyword matching, we employ a sophisticated two-layer AI categorization system to ensure insights are both organic and actionable.</p>
              
              <h4 style={{ marginTop: '24px', marginBottom: '12px', fontSize: '16px' }}>Layer 1: Open Thematic Clustering</h4>
              <p>The 72,566 engine-ready units are plotted into a high-dimensional space based on their semantic meaning. We use network graph community detection (the Louvain algorithm) to group similar units together. This allows organic themes to emerge from the bottom up without being forced into predefined boxes.</p>

              <h4 style={{ marginTop: '24px', marginBottom: '12px', fontSize: '16px' }}>Layer 2: Fixed Barrier Mapping</h4>
              <p>To make the data actionable for Growth and Product teams, we pass the organic clusters through a fixed analytical lens to understand <em>why</em> users fail to explore new categories.</p>
              <ul style={{ paddingLeft: '20px', marginBottom: '16px' }}>
                <li><strong>Stratified Sampling:</strong> For every cluster, we sample a representative batch of reviews strictly spanning the full rating range (1-star through 5-stars). This prevents the analysis from being biased solely by angry 1-star reviews.</li>
                <li><strong>Fixed Taxonomy:</strong> A Large Language Model (LLM) evaluates the sample and maps the cluster to exactly one of six predefined primary barriers:
                  <ol style={{ paddingLeft: '24px', marginTop: '8px' }}>
                    <li><strong>Trust/Risk:</strong> Hesitation due to unpredictable quality or authenticity.</li>
                    <li><strong>Economic:</strong> Fees/pricing make trying new items not worth it.</li>
                    <li><strong>Reliability:</strong> Late or missing orders push users back to safe staples.</li>
                    <li><strong>Discovery:</strong> App navigation only surfaces known items.</li>
                    <li><strong>Recovery:</strong> Poor customer support kills willingness to experiment.</li>
                    <li><strong>Habit Load:</strong> Reordering is effortless; exploring requires too much cognitive effort.</li>
                  </ol>
                </li>
                <li><strong>Strict Guardrails:</strong> If a cluster does not genuinely fit any of these barriers, it is forcibly tagged as <code>out_of_scope</code>. The AI is strictly prohibited from inventing new categories.</li>
              </ul>

              <h3 style={{ marginTop: '32px', marginBottom: '16px', fontSize: '18px' }}>4. Auditable Traceability</h3>
              <p>Every single number presented in the final insights dashboard is fully auditable. We do not use LLMs to guess or estimate volumes.</p>
              <p>When a barrier like "Reliability" shows a specific count, that number represents the exact sum of the member units within the clusters mapped to that barrier. Every sub-theme and representative quote can be traced directly back to a real <code>review_id</code> from the source data, ensuring 100% data integrity and transparency.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const root = createRoot(document.getElementById("root"));'''

content = content.replace(body_target, body_replacement)

with open('web/index.html', 'w', encoding='utf-8') as f:
    f.write(content)
