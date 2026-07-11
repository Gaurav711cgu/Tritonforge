---
version: 1.0
name: RailMind-design-system
description: >
  A high-stakes operations console for Indian Railways autonomous dispatch.
  The system is built on near-black surfaces with a single high-voltage red
  accent. Every pixel signals operational authority — this is not a SaaS
  dashboard, it is a safety-critical command interface. Typography is split
  between a condensed all-caps label face for section headers and a monospace
  face for all live data, telemetry readouts, hash values, and latency numbers.
  The red accent (#E53935) is used exclusively for: active nav tab underlines,
  CTA buttons, alert/conflict labels, confidence bars, and section header
  eyebrows. It never appears as a background fill except on full-width CTA
  buttons. Cream, gradients, rounded pills, and glow effects are strictly
  forbidden.

colors:
  # Surfaces — layered dark system, no pure black
  surface-base: "#0d0d0b"          # Page background — near-black with warm tint
  surface-panel: "#13120f"         # Main panel/card background
  surface-elevated: "#1a1916"      # Elevated card, inner sections
  surface-input: "#111110"         # Input fields, dropdowns
  surface-row: "#161513"           # Table rows, log lines (alternating)

  # Accent — single red, three states
  accent: "#E53935"                # Primary red — buttons, active tabs, bars, labels
  accent-dim: "#B71C1C"            # Pressed / active state of red
  accent-subtle: "#3a1010"         # Red tint for alert card borders/backgrounds

  # Text
  ink: "#F5F5F0"                   # Primary text — headings, values
  ink-soft: "#A09D96"              # Secondary text — descriptions, subtitles
  ink-muted: "#5c5a55"             # Tertiary — disabled, placeholders
  ink-on-red: "#FFFFFF"            # Text on red buttons

  # Status
  status-ok: "#4CAF50"             # SYSTEM OK dot, HEALTHY badge
  status-warn: "#FFA726"           # Warning states
  status-fail: "#E53935"           # FAILED, critical — same as accent
  status-pending: "#5c5a55"        # PENDING label

  # Borders
  border: "#2a2825"                # Default panel border — 1px
  border-accent: "#E53935"         # Active panel border, selected card
  border-soft: "#1e1d1a"           # Subtle dividers inside panels

typography:
  # Section headers — all caps, wide tracking, red
  label-section:
    fontFamily: "'JetBrains Mono', 'Courier New', monospace"
    fontSize: 13px
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: 2px
    textTransform: uppercase
    color: "{colors.accent}"

  # Page/panel titles — white, condensed
  title-lg:
    fontFamily: "'Inter', sans-serif"
    fontSize: 22px
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: -0.3px
    textTransform: none
    color: "{colors.ink}"

  title-md:
    fontFamily: "'Inter', sans-serif"
    fontSize: 16px
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: 0
    color: "{colors.ink}"

  # Nav items — spaced caps
  nav-item:
    fontFamily: "'Inter', sans-serif"
    fontSize: 12px
    fontWeight: 600
    lineHeight: 1
    letterSpacing: 1.5px
    textTransform: uppercase
    color: "{colors.ink-soft}"

  # All live data, metrics, hashes, latency numbers
  data-readout:
    fontFamily: "'JetBrains Mono', 'Courier New', monospace"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: 0
    color: "{colors.ink}"

  # Large metric values (94.5%, 104.2 km/h)
  metric-value:
    fontFamily: "'JetBrains Mono', 'Courier New', monospace"
    fontSize: 22px
    fontWeight: 700
    lineHeight: 1
    letterSpacing: -0.5px
    color: "{colors.ink}"

  metric-label:
    fontFamily: "'Inter', sans-serif"
    fontSize: 10px
    fontWeight: 500
    lineHeight: 1.2
    letterSpacing: 1.5px
    textTransform: uppercase
    color: "{colors.ink-muted}"

  body:
    fontFamily: "'Inter', sans-serif"
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.55
    letterSpacing: 0
    color: "{colors.ink-soft}"

  caption:
    fontFamily: "'JetBrains Mono', 'Courier New', monospace"
    fontSize: 11px
    fontWeight: 400
    lineHeight: 1.4
    letterSpacing: 0.5px
    color: "{colors.ink-muted}"

  button:
    fontFamily: "'Inter', sans-serif"
    fontSize: 12px
    fontWeight: 700
    lineHeight: 1
    letterSpacing: 2px
    textTransform: uppercase

rounded:
  none: 0px
  xs: 2px       # Minimal rounding — badges, status pills
  sm: 4px       # Inputs, small buttons
  md: 6px       # Cards, panels
  lg: 8px       # Large panels only

# NO pill/full rounding. This is an ops console, not a consumer app.

spacing:
  xxs: 4px
  xs: 8px
  sm: 12px
  md: 16px
  lg: 24px
  xl: 32px
  xxl: 48px
  section: 80px   # Between major page sections

components:
  # ─── NAV ───────────────────────────────────────────────────────────────────

  top-nav:
    height: 56px
    background: "{colors.surface-base}"
    borderBottom: "1px solid {colors.border}"
    layout: "logo-left | nav-center | status-right"
    description: >
      Dark near-black bar. Logo: RailMind wordmark in {typography.title-md}
      bold white + "OPERATIONS SOLVER · INDIAN RAILWAYS" in {typography.caption}
      muted below. Nav items in {typography.nav-item} spaced caps. Active tab
      has a 2px {colors.accent} underline and {colors.ink} text. Status
      indicator (NOMINAL/ALERT) sits far right as a small dot + label.

  scenario-bar:
    height: 44px
    background: "{colors.surface-elevated}"
    borderBottom: "1px solid {colors.border}"
    description: >
      Thin strip below top-nav. Shows current step label ("NOMINAL OPERATION —
      SECTOR NORTH") in {typography.title-md} left, scrolling status ticker
      center in {typography.data-readout} muted, step indicator dots + NEXT
      STEP / RESET buttons right. NEXT STEP uses {component.button-primary}.

  # ─── BUTTONS ───────────────────────────────────────────────────────────────

  button-primary:
    background: "{colors.accent}"
    color: "{colors.ink-on-red}"
    font: "{typography.button}"
    padding: "10px 20px"
    height: 36px
    borderRadius: "{rounded.sm}"
    border: none
    description: >
      Full red fill button. Used for NEXT STEP, SEARCH, CALCULATE, SUBMIT,
      BROADCAST. Active state uses {colors.accent-dim}. Never use outline
      variant on dark backgrounds — it reads as disabled.

  button-secondary:
    background: "transparent"
    color: "{colors.ink}"
    font: "{typography.button}"
    padding: "10px 20px"
    height: 36px
    borderRadius: "{rounded.sm}"
    border: "1px solid {colors.border}"
    description: >
      Used for RESET, EXPORT JSON, VERIFY LEDGER, secondary actions.
      On hover, border shifts to {colors.ink-soft}.

  button-ghost:
    background: "transparent"
    color: "{colors.ink-soft}"
    font: "{typography.button}"
    padding: "8px 14px"
    height: 32px
    borderRadius: "{rounded.xs}"
    border: none
    description: >
      Category filter tabs (ALL, Monitor, Conflict, etc.). Active state:
      background {colors.accent}, color {colors.ink-on-red}.

  # ─── PANELS / CARDS ────────────────────────────────────────────────────────

  panel:
    background: "{colors.surface-panel}"
    border: "1px solid {colors.border}"
    borderRadius: "{rounded.md}"
    padding: "{spacing.lg}"
    description: >
      Standard content container. No shadow, no glow. Border is the only
      separation from the page background. Header inside panel: {typography.title-md}
      white + subtitle in {typography.body} muted, separated by {spacing.sm}.

  panel-alert:
    background: "{colors.accent-subtle}"
    border: "1px solid {colors.border-accent}"
    borderRadius: "{rounded.md}"
    padding: "{spacing.lg}"
    description: >
      Panel variant for emergency/conflict states. Red-tinted background with
      red border. Used for EMERGENCY HOTLINE DESK, active alert cards.

  agent-card:
    background: "{colors.surface-elevated}"
    border: "1px solid {colors.border}"
    borderRadius: "{rounded.md}"
    padding: "{spacing.md}"
    description: >
      Used in Decision Flow grid (Monitor, Conflict, Cascade, etc.). Carries:
      agent name in {typography.title-md}, class name in {typography.caption}
      muted, description in {typography.body}, CONFIDENCE label + red progress
      bar, LAST RUN in {typography.caption} muted. Status badge top-right.

  metric-cell:
    background: "transparent"
    description: >
      Used in SYSTEM METRICS and status footers. Stack: {typography.metric-label}
      all-caps muted on top, {typography.metric-value} large mono below.
      Red value = bad (CAPACITY 34%), white = normal (LINE EFF 94.5%).

  log-terminal:
    background: "{colors.surface-input}"
    border: "1px solid {colors.border}"
    borderRadius: "{rounded.sm}"
    padding: "{spacing.md}"
    fontFamily: "'JetBrains Mono', monospace"
    fontSize: 13px
    lineHeight: 1.7
    description: >
      Monospace terminal-style log output. Log type label in brackets in
      {colors.accent} bold ([Monitor], [Conflict], [Audit]). Log text in
      {colors.ink}. Background is the darkest surface in the system.

  data-table-row:
    borderBottom: "1px solid {colors.border-soft}"
    padding: "10px {spacing.md}"
    description: >
      Table rows for segment blocks, procedures, ledger entries. Label left in
      {typography.body} white, value/badge right. No zebra striping — border
      only.

  # ─── STATUS / BADGES ───────────────────────────────────────────────────────

  badge-status:
    padding: "3px 8px"
    borderRadius: "{rounded.xs}"
    font: "{typography.caption}"
    letterSpacing: 1px
    textTransform: uppercase
    variants:
      HEALTHY: "background: rgba(76,175,80,0.15); color: {colors.status-ok}; border: 1px solid rgba(76,175,80,0.3)"
      FAILED:  "background: {colors.accent-subtle}; color: {colors.accent}; border: 1px solid {colors.border-accent}"
      PENDING: "background: transparent; color: {colors.ink-muted}; border: 1px solid {colors.border}"
      IN_REVIEW: "background: rgba(255,167,38,0.1); color: {colors.status-warn}; border: 1px solid rgba(255,167,38,0.3)"
      RESOLVED: "background: transparent; color: {colors.ink-muted}; border: 1px solid {colors.border}"

  confidence-bar:
    height: 3px
    background: "{colors.border}"
    fill: "{colors.accent}"
    borderRadius: 0
    description: >
      Flat 3px red bar. No border-radius. Used for CONFIDENCE in agent cards
      and SHAP feature impact bars. Negative contribution bars use {colors.accent},
      positive use {colors.ink-soft}.

  status-footer:
    height: 32px
    background: "{colors.surface-base}"
    borderTop: "1px solid {colors.border}"
    description: >
      Fixed bottom bar. Left: green dot + SYSTEM OK in {typography.caption}
      green. Then: UPTIME, REQUESTS, LATENCY, AGENTS, ML status — all in
      {typography.caption} with muted labels and white values, pipe-separated.
      Right: RAILMIND v2.1 in {typography.caption} muted.

  # ─── INPUTS ────────────────────────────────────────────────────────────────

  input:
    background: "{colors.surface-input}"
    color: "{colors.ink}"
    border: "1px solid {colors.border}"
    borderRadius: "{rounded.sm}"
    padding: "8px 12px"
    height: 36px
    font: "{typography.data-readout}"
    description: >
      All text inputs and dropdowns. Focus state: border becomes
      {colors.accent}. Placeholder in {colors.ink-muted}.

layout:
  max-width: 1440px
  grid: 12-column
  gutter: 24px
  margin: 24px

  page-structure: >
    top-nav (56px fixed) →
    scenario-bar (44px fixed) →
    main content area (fills remaining viewport) →
    status-footer (32px fixed)

  main-split: >
    Most pages use a primary content area (left, ~70%) and a sidebar panel
    (right, ~30%) — e.g. telemetry map + dispatch solutions, or ledger + checklist.
    The split is achieved with CSS grid, not absolute positioning.

rules:
  - "Dark surface only. Never use a light background anywhere in the UI."
  - "Red ({colors.accent}) is reserved for: active tab underlines, CTA buttons, section header labels, alert borders, and confidence/SHAP bars. Nowhere else."
  - "All live data, metrics, hashes, IDs, latency values, and code must use JetBrains Mono."
  - "Section headers are always {typography.label-section} — all-caps, red, mono, tracked. Never use sentence-case section headers."
  - "No border-radius above 8px. No pill shapes. No rounded-full on anything."
  - "No gradients. No glow effects. No box-shadows. Panels are separated by border lines only."
  - "No hover animations beyond color/border state changes. No scroll-triggered animations."
  - "No Inter for data values. No monospace for body paragraphs."
  - "Status dots are 6px circles — green for OK, red for FAIL, amber for WARN. No pulsing animation."
  - "Buttons are always uppercase, tracked. Never sentence-case."
  - "The page background bleeds through as texture — railway locomotive imagery is used as a very dark (10–15% opacity) background layer behind panels."

do:
  - "Use 1px borders in {colors.border} as the only surface separator."
  - "Use {typography.label-section} (red, mono, caps) for every section eyebrow."
  - "Show real operational data in readouts — train numbers, segment codes, hash values."
  - "Use {component.log-terminal} for any output that is machine-generated."
  - "Use pipe characters (|) and dashes (—) as structural dividers in text where appropriate."
  - "Keep the status footer visible and populated at all times."
  - "Use {component.badge-status} variants exactly — don't invent new badge colors."

dont:
  - "Don't use white or any light color as a surface background."
  - "Don't use coral, teal, purple, blue, or any color not in this system."
  - "Don't bold section headers beyond what {typography.label-section} specifies."
  - "Don't add card shadows or glows — border is the only depth signal."
  - "Don't use placeholder icons or stock imagery."
  - "Don't use Tailwind default border-radius values (rounded-lg = 8px is the max allowed here, rounded-xl and above are forbidden)."
  - "Don't add loading skeletons with shimmer animations — use static '--' placeholder text."
  - "Don't use Inter for anything labeled as live data, a metric value, a hash, or a system code."

responsive:
  mobile:
    width: "<768px"
    changes: "Top nav collapses to hamburger. Panels stack single-column. Status footer truncates to SYSTEM OK + latency only. Metric cells wrap 2-up."
  tablet:
    width: "768–1024px"
    changes: "Main split becomes 60/40. Agent cards 2-up."
  desktop:
    width: ">1024px"
    changes: "Full layout as described above."
