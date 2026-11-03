// Fakt interface, shared by the side panel and the web app. Everything shown here comes
// from the internet, so it only ever reaches the page as text nodes, never as markup.

const inExtension = location.protocol === "chrome-extension:";
const API_ROOT = inExtension ? "http://127.0.0.1:8741" : "";
const SERVER_DOWN =
  "The Fakt server isn't running. Start it with `uv run fakt serve`, then try again.";
const SHOWN_TABS = [
  ["summary", "Summary"],
  ["evidence", "Evidence"],
  ["report", "Full report"],
  ["ask", "Ask"],
];
const VERDICT_TONES = {
  True: "true",
  "Mostly true": "true",
  Contested: "contested",
  Misleading: "false",
  "Mostly false": "false",
  False: "false",
  Unproven: "unproven",
};
const STANCES = {
  supports: { word: "Supports", tone: "true", lean: 0.85 },
  refutes: { word: "Refutes", tone: "false", lean: 0.15 },
  mixed: { word: "Mixed", tone: "contested", lean: 0.5 },
};
const SOURCE_KINDS = {
  paper: "Paper",
  dataset: "Dataset",
  official: "Official source",
  news: "News",
  fact_check: "Fact-check",
  web: "Web page",
};
const PLATFORMS = {
  instagram: "Instagram",
  x: "X",
  facebook: "Facebook",
  tiktok: "TikTok",
  youtube: "YouTube",
  reddit: "Reddit",
  article: "Article",
  web: "Web page",
  text: "Pasted text",
};
const STAGE_NAMES = {
  start: "Starting",
  capture: "Reading the post",
  claims: "Finding the claims",
  evidence: "Searching for evidence",
  appraisal: "Grading the sources",
  verdict: "Weighing the evidence",
  report: "Writing the report",
  failed: "One claim couldn't be finished",
};
const SCALE_LANES = 3;
const ASK_STARTERS = [
  "Which source matters most here?",
  "What would a sceptic say?",
  "Explain this to a twelve-year-old.",
];

const HEADING = /^#{1,6}\s/;
const BLOCK_START = /^(#{1,6}\s|>|```)/;

const main = document.getElementById("main");
const pasteBox = document.getElementById("paste");
const intakeStatus = document.getElementById("intake-status");

const state = {
  report: null,
  navigation: 0,
  stream: null,
  pastedImage: null,
  askSource: null,
  focusSource: null,
};

start();

function start() {
  document.getElementById("paste-form").addEventListener("submit", submitPaste);
  pasteBox.addEventListener("paste", takePastedImage);
  document.getElementById("remove-image").addEventListener("click", () => setPastedImage(null));
  window.addEventListener("hashchange", route);

  if (inExtension) {
    const checkPost = document.getElementById("check-post");
    checkPost.hidden = false;
    checkPost.addEventListener("click", checkCurrentPost);
    runQueuedCheck();
    chrome.storage.session.onChanged.addListener((changes) => {
      if (changes.queuedCheck?.newValue) runQueuedCheck();
    });
  }

  const running = remember("activeJob");
  if (running && !location.hash) location.hash = `#/job/${running.id}`;
  route();
  refreshRail();
}

// Routing

function route() {
  const [, kind, id, tab] = location.hash.split("/");
  state.navigation += 1;
  stopStream();
  if (kind === "job" && id) return showJob(id);
  if (kind === "report" && id) return showReport(id, tab || "summary");
  document.body.dataset.view = "home";
  markCurrentHistoryItem();
  main.replaceChildren(guide());
}

function reportPath(id, tab = "summary") {
  return `#/report/${id}/${tab}`;
}

// Starting a check

async function submitPaste(event) {
  event.preventDefault();
  const entered = pasteBox.value.trim();
  const isLink = /^https?:\/\/\S+$/.test(entered);
  const image = state.pastedImage;
  if (!entered && !image) return showIntakeMessage("Paste something to check first.", true);
  const started = await startCheck(
    {
      url: isLink ? entered : undefined,
      text: isLink ? undefined : entered || undefined,
      image_base64: image ?? undefined,
    },
    { text: entered || "A pasted screenshot" },
  );
  if (!started) return;
  pasteBox.value = "";
  setPastedImage(null);
}

async function checkCurrentPost() {
  const button = document.getElementById("check-post");
  button.disabled = true;
  showIntakeMessage("");
  try {
    const { id: windowId } = await chrome.windows.getCurrent();
    const found = await chrome.runtime.sendMessage({ type: "read-post", windowId });
    if (found.error) showIntakeMessage(found.error, true);
    else await startCheck(found.payload, found.preview);
  } finally {
    button.disabled = false;
  }
}

async function runQueuedCheck() {
  const queuedCheck = await chrome.runtime.sendMessage({ type: "take-queued-check" });
  if (!queuedCheck) return;
  if (queuedCheck.error) {
    location.hash = "#/";
    showIntakeMessage(queuedCheck.error, true);
  } else {
    await startCheck(queuedCheck.payload, queuedCheck.preview);
  }
}

async function startCheck(payload, preview) {
  showIntakeMessage("");
  try {
    const { id } = await api("/checks", postJson(payload));
    remember("activeJob", { id, preview });
    location.hash = `#/job/${id}`;
    return true;
  } catch (error) {
    location.hash = "#/";
    showIntakeMessage(error.message, true);
    return false;
  }
}

function takePastedImage(event) {
  const file = [...event.clipboardData.files].find((item) => item.type.startsWith("image/"));
  if (!file) return;
  event.preventDefault();
  const reader = new FileReader();
  reader.onload = () => setPastedImage(reader.result);
  reader.readAsDataURL(file);
}

function setPastedImage(dataUrl) {
  state.pastedImage = dataUrl;
  document.getElementById("pasted-image").hidden = !dataUrl;
  document.getElementById("pasted-preview").src = dataUrl ?? "";
}

function showIntakeMessage(message, isError = false) {
  intakeStatus.classList.toggle("is-error", isError);
  intakeStatus.replaceChildren(inlineMarkdown(message));
}

// Running check

function showJob(id) {
  document.body.dataset.view = "job";
  const preview = remember("activeJob")?.preview;
  const stages = h("ol", { class: "stages" });
  main.replaceChildren(
    h(
      "section",
      { class: "running", "aria-live": "polite" },
      h("h1", {}, "Checking this post"),
      preview?.text && h("p", { class: "post-excerpt" }, preview.text),
      stages,
    ),
  );
  addStage(stages, { stage: "start", message: "Connecting" });

  const stream = new EventSource(`${API_ROOT}/checks/${id}/events`);
  state.stream = stream;
  stream.onmessage = (event) => addStage(stages, JSON.parse(event.data));
  stream.addEventListener("done", (event) => {
    stopStream();
    remember("activeJob", null);
    refreshRail();
    location.replace(reportPath(JSON.parse(event.data).report_id));
  });
  stream.addEventListener("failed", (event) => {
    failJob(JSON.parse(event.data).message);
  });
  stream.onerror = () => {
    if (stream.readyState === EventSource.CLOSED) {
      failJob("This check is no longer running. The server may have restarted.");
    }
  };
}

function addStage(list, { stage, message }) {
  if (stage === "done") return;
  const last = list.lastElementChild;
  if (last?.dataset.stage === stage) {
    last.querySelector(".stage-message").textContent = message;
    return;
  }
  last?.classList.remove("is-current");
  list.append(
    h(
      "li",
      { class: "stage is-current", "data-stage": stage },
      h("div", { class: "stage-name" }, STAGE_NAMES[stage] ?? stage),
      h("div", { class: "stage-message" }, message),
    ),
  );
}

function failJob(message) {
  stopStream();
  remember("activeJob", null);
  main.replaceChildren(
    h(
      "section",
      { class: "running" },
      h("h1", {}, "That check didn't finish"),
      h("p", { class: "notice" }, inlineMarkdown(message)),
      h("a", { href: "#/" }, "Start another check"),
    ),
  );
}

function stopStream() {
  state.stream?.close();
  state.stream = null;
}

// Report

async function showReport(id, tab) {
  document.body.dataset.view = "report";
  markCurrentHistoryItem();
  if (state.report?.id !== id) {
    const opened = state.navigation;
    main.replaceChildren(h("p", { class: "loading" }, "Opening the report"));
    try {
      const report = await api(`/reports/${encodeURIComponent(id)}`);
      if (opened !== state.navigation) return;
      state.report = report;
      state.askSource = null;
    } catch (error) {
      if (opened !== state.navigation) return;
      main.replaceChildren(
        h("p", { class: "notice" }, inlineMarkdown(error.message)),
        h("a", { href: "#/" }, "Back to your checks"),
      );
      return;
    }
  }
  renderReport(state.report, tab);
}

function renderReport(report, tab) {
  const views = { summary: summaryTab, evidence: evidenceTab, report: fullReportTab, ask: askTab };
  const citations = citationNumbers(report);
  main.replaceChildren(
    h("a", { class: "back", href: "#/" }, "All checks"),
    postHeader(report),
    tabBar(report, tab),
    h("div", {}, (views[tab] ?? summaryTab)(report, citations)),
  );
  if (state.focusSource) focusSource(state.focusSource);
}

function postHeader({ capture, created }) {
  const place = PLATFORMS[capture.platform] ?? capture.platform;
  const byline = capture.author ? `${place}, ${capture.author}` : place;
  return h(
    "header",
    { class: "post-header" },
    h(
      "div",
      { class: "post-origin" },
      h("span", {}, byline),
      h("span", {}, shortDate(created)),
    ),
    excerptOf(capture) && h("p", { class: "post-excerpt" }, excerptOf(capture)),
    capture.url && externalLink(capture.url, "Open the original"),
  );
}

function excerptOf(capture) {
  return [capture.text, capture.transcript, capture.image_text].find(Boolean) ?? "";
}

function tabBar(report, current) {
  return h(
    "nav",
    { class: "tabs", "aria-label": "Report sections" },
    SHOWN_TABS.map(([tab, name]) =>
      h(
        "a",
        { href: reportPath(report.id, tab), "aria-current": tab === current ? "page" : null },
        name,
      ),
    ),
  );
}

function summaryTab(report, citations) {
  return [
    h("div", {}, report.checks.map((check) => verdictBlock(report, check, citations))),
    report.checks.length === 0 &&
      h("p", { class: "notice" }, "No checkable claims were found in this post."),
    profileBlock(report.profile),
    report.one_page &&
      h(
        "section",
        { class: "plain-english" },
        markdown(report.one_page, citations),
      ),
  ];
}

function verdictBlock(report, check, citations) {
  const { claim, verdict, sources } = check;
  const label = verdict?.label ?? "Unproven";
  const tone = VERDICT_TONES[label] ?? "unproven";
  const percent = verdict ? Math.round(verdict.probability_true * 100) : null;
  return h(
    "article",
    { class: `verdict tone-${tone}` },
    h("h2", { class: "verdict-claim" }, claim.text),
    h("p", { class: "verdict-label" }, label),
    percent !== null && h("p", { class: "likelihood" }, `${percent}% likely true`),
    percent !== null && scale(check, percent),
    verdict?.summary &&
      h("div", { class: "verdict-summary" }, inlineMarkdown(verdict.summary, citations)),
    sources.length > 0 &&
      h(
        "details",
        { class: "evidence" },
        h("summary", {}, "Show the evidence"),
        sourceList(report, check, citations),
      ),
  );
}

function scale(check, percent) {
  const marks = graded(check).filter(({ stance }) => stance);
  const heaviest = Math.max(...marks.map(({ appraisal }) => appraisal.weight), 0.01);
  return h(
    "div",
    {
      class: "scale",
      role: "img",
      "aria-label": `${percent} percent likely true. ${stanceCount(marks)}`,
    },
    h(
      "div",
      { class: "scale-track" },
      positioned(h("div", { class: "scale-fill" }), { width: `${percent}%` }),
      positioned(h("div", { class: "scale-diamond" }), { left: `${percent}%` }),
    ),
    positioned(
      h(
        "div",
        { class: "scale-sources" },
        marks.map(({ source, appraisal, stance }, index) => {
          const size = 7 + 9 * (appraisal.weight / heaviest);
          return positioned(
            h("div", {
              class: `source-mark tone-${stance.tone}${source.retracted ? " is-retracted" : ""}`,
              title: `${stance.word}: ${source.title}`,
            }),
            {
              left: `${leanOf(appraisal) * 100}%`,
              top: `${(index % SCALE_LANES) * 13}px`,
              width: `${size}px`,
              height: `${size}px`,
            },
          );
        }),
      ),
      { height: `${(Math.min(marks.length, SCALE_LANES) - 1) * 13 + 18}px` },
    ),
    h("div", { class: "scale-ends" }, h("span", {}, "False"), h("span", {}, "True")),
    marks.length > 0 && h("p", { class: "scale-count" }, stanceCount(marks)),
  );
}

function stanceCount(marks) {
  const counted = marks.filter(({ source }) => !source.retracted);
  const count = (word) => counted.filter(({ stance }) => stance.word === word).length;
  const totals = [
    [count("Supports"), "supports it", "support it"],
    [count("Refutes"), "refutes it", "refute it"],
    [count("Mixed"), "mixed", "mixed"],
  ].filter(([total]) => total > 0);
  const words = totals
    .map(([total, one, many]) => `${total} ${total === 1 ? one : many}`)
    .join(", ");
  return words ? `Sources: ${words}.` : "";
}

function leanOf(appraisal) {
  const { supports = 0, refutes = 0, mixed = 0 } = appraisal.stance_probabilities ?? {};
  const total = supports + refutes + mixed;
  if (total > 0) return (supports + mixed / 2) / total;
  return STANCES[appraisal.stance]?.lean ?? 0.5;
}

function profileBlock(profile) {
  return h(
    "section",
    { class: "profile", "aria-label": "How the post behaves" },
    h(
      "div",
      { class: "profile-meters" },
      meter("Signal or hype", profile.signal_vs_hype, "Hype", "Signal", [
        "Mostly hype",
        "A bit of both",
        "Mostly signal",
      ]),
      meter("Fact or fiction", profile.fact_vs_fiction, "Fiction", "Fact", [
        "Mostly fiction",
        "A bit of both",
        "Mostly fact",
      ]),
    ),
    profile.tactics.length > 0 &&
      h("p", { class: "tactics" }, `Persuasion tactics spotted: ${profile.tactics.join(", ")}.`),
  );
}

function meter(name, value, lowEnd, highEnd, [low, middle, high]) {
  const word = value < 0.35 ? low : value > 0.65 ? high : middle;
  return h(
    "div",
    { class: "meter", role: "img", "aria-label": `${name}: ${word}` },
    h(
      "div",
      { class: "meter-reading" },
      h("span", { class: "meter-name" }, name),
      h("span", { class: "meter-word" }, word),
    ),
    h(
      "div",
      { class: "meter-rail" },
      positioned(h("div", { class: "meter-mark" }), { left: `${value * 100}%` }),
    ),
    h("div", { class: "meter-ends" }, h("span", {}, lowEnd), h("span", {}, highEnd)),
  );
}

function evidenceTab(report, citations) {
  if (report.checks.length === 0) {
    return h("p", { class: "notice" }, "This post had no claims to gather evidence for.");
  }
  return report.checks.map((check) => claimEvidence(report, check, citations));
}

function claimEvidence(report, check, citations) {
  const { claim, verdict } = check;
  const label = verdict?.label ?? "Unproven";
  const percent = verdict ? `, ${Math.round(verdict.probability_true * 100)}% likely true` : "";
  const argument = [
    ["Where experts stand", verdict?.consensus],
    ["The reasoning", verdict?.reasoning],
    ["Strongest case for", verdict?.strongest_for],
    ["Strongest case against", verdict?.strongest_against],
    ["What would change this", verdict?.what_would_change_this],
  ].filter(([, text]) => text);
  return h(
    "section",
    { class: `claim-evidence tone-${VERDICT_TONES[label] ?? "unproven"}` },
    h("h2", {}, claim.text),
    h("p", { class: "claim-evidence-verdict" }, label + percent),
    argument.length > 0 &&
      h(
        "dl",
        { class: "argument" },
        argument.flatMap(([title, text]) => [
          h("dt", {}, title),
          h("dd", {}, markdown(text, citations)),
        ]),
      ),
    h("h3", { class: "sources-title" }, sourceHeading(check)),
    sourceList(report, check, citations),
  );
}

function sourceHeading({ sources }) {
  return sources.length === 1 ? "One source" : `${sources.length} sources`;
}

function sourceList(report, check, citations) {
  const entries = graded(check);
  const relevant = entries.filter(({ stance }) => stance);
  const offTopic = entries.filter(({ stance }) => !stance);
  return [
    h("ol", {}, relevant.map((entry) => sourceEntry(report, entry, citations))),
    offTopic.length > 0 &&
      h(
        "details",
        { class: "off-topic" },
        h("summary", {}, `Left out of the verdict: ${offTopic.length}`),
        h("ol", {}, offTopic.map((entry) => sourceEntry(report, entry, citations))),
      ),
  ];
}

function graded({ sources, appraisals }) {
  const bySource = new Map(appraisals.map((appraisal) => [appraisal.source_id, appraisal]));
  return sources
    .map((source) => {
      const appraisal = bySource.get(source.id) ?? {};
      const leftOut = appraisal.excluded && !source.retracted;
      const stance = leftOut ? undefined : STANCES[appraisal.stance];
      return { source, appraisal, stance };
    })
    .sort((a, b) => (b.appraisal.weight ?? 0) - (a.appraisal.weight ?? 0));
}

function sourceEntry(report, { source, appraisal, stance }, citations) {
  const facts = [
    SOURCE_KINDS[source.kind] ?? "Source",
    appraisal.design && appraisal.design !== "unknown" ? appraisal.design : null,
    source.year,
    source.venue,
  ].filter(Boolean);
  const quality = appraisal.quality ?? 0;
  return h(
    "li",
    { class: "source", "data-source": source.id },
    h(
      "div",
      { class: "source-title" },
      h("span", { class: "source-number" }, String(citations.get(source.id))),
      externalLink(source.url, source.title),
    ),
    h(
      "div",
      { class: "source-body" },
      h("p", { class: "source-facts" }, facts.join(", ")),
      source.retracted &&
        h("p", { class: "retraction" }, "Retracted. This work has been withdrawn, so don't rely on it."),
      stance &&
        h(
          "div",
          { class: "source-grade" },
          h("span", { class: `source-stance tone-${stance.tone}` }, stance.word),
          h(
            "span",
            { class: "quality" },
            "Quality",
            h(
              "span",
              { class: "quality-bar", role: "img", "aria-label": `${Math.round(quality * 100)}%` },
              positioned(h("span", { class: "quality-fill" }), { width: `${quality * 100}%` }),
            ),
          ),
        ),
      appraisal.key_finding &&
        h("p", { class: "source-finding" }, inlineMarkdown(appraisal.key_finding, citations)),
      appraisal.excluded &&
        !source.retracted &&
        h("p", { class: "source-limits" }, appraisal.excluded),
      appraisal.cited_check &&
        h("p", { class: "source-finding" }, inlineMarkdown(appraisal.cited_check, citations)),
      appraisal.quote && h("blockquote", {}, appraisal.quote),
      appraisal.quote_check === "contradicted" &&
        h("p", { class: "source-limits" }, "The passage around this quote cuts against this reading."),
      appraisal.limitations &&
        h("p", { class: "source-limits" }, `Limits: ${appraisal.limitations}`),
      h(
        "div",
        { class: "source-actions" },
        h(
          "button",
          { type: "button", onclick: () => askAboutSource(report, source) },
          source.kind === "paper" ? "Ask this paper" : "Ask this source",
        ),
      ),
    ),
  );
}

function fullReportTab(report, citations) {
  return h(
    "div",
    { class: "full-report" },
    report.detail
      ? markdown(report.detail, citations)
      : h("p", { class: "notice" }, "This check has no long report."),
  );
}

// Citations: [source-id] in Markdown becomes a number that jumps to the source.

function citationNumbers(report) {
  const numbers = new Map();
  for (const check of report.checks) {
    for (const { source } of graded(check)) {
      if (!numbers.has(source.id)) numbers.set(source.id, numbers.size + 1);
    }
  }
  return numbers;
}

function jumpToSource(id) {
  state.focusSource = id;
  const target = reportPath(state.report.id, "evidence");
  if (location.hash === target) focusSource(id);
  else location.hash = target;
}

function focusSource(id) {
  state.focusSource = null;
  const entry = main.querySelector(`.claim-evidence [data-source="${CSS.escape(id)}"]`);
  if (!entry) return;
  entry.closest("details")?.setAttribute("open", "");
  entry.scrollIntoView({ block: "start" });
  entry.classList.remove("is-focused");
  entry.offsetWidth; // restarts the highlight when the same source is cited twice
  entry.classList.add("is-focused");
}

// Ask

function askAboutSource(report, source) {
  state.askSource = source;
  const target = reportPath(report.id, "ask");
  if (location.hash === target) route();
  else location.hash = target;
}

function askTab(report, citations) {
  const turns = remember(`chat.${report.id}`) ?? [];
  const conversation = h("div", { class: "conversation" });
  const paint = () => {
    conversation.replaceChildren(
      ...(turns.length ? turns.map((turn) => renderTurn(turn, citations)) : [askStarters(ask)]),
    );
  };
  const question = h("textarea", { rows: 2, "aria-label": "Your question" });
  const submit = h("button", { class: "primary", type: "submit" }, "Ask");
  const form = h(
    "form",
    {
      class: "ask-form",
      onsubmit: (event) => {
        event.preventDefault();
        ask(question.value.trim());
      },
    },
    state.askSource && askFocus(report),
    question,
    submit,
  );
  async function ask(text) {
    if (!text) return;
    const source = state.askSource;
    const turn = { question: text, about: source?.title };
    turns.push(turn);
    question.value = "";
    submit.disabled = true;
    turn.answer = "Reading the evidence";
    paint();
    try {
      const body = { question: text, source_id: source?.id };
      const reply = await api(`/reports/${report.id}/ask`, postJson(body));
      turn.answer = reply.answer;
      remember(`chat.${report.id}`, turns);
    } catch (error) {
      turn.answer = null;
      turn.error = error.message;
    }
    submit.disabled = false;
    paint();
  }
  paint();
  return h("section", { class: "ask" }, conversation, form);
}

function askFocus(report) {
  return h(
    "div",
    { class: "ask-focus" },
    h("span", {}, `Asking about ${state.askSource.title}`),
    h(
      "button",
      {
        class: "quiet",
        type: "button",
        onclick: () => {
          state.askSource = null;
          renderReport(report, "ask");
        },
      },
      "Whole report",
    ),
  );
}

function askStarters(ask) {
  return h(
    "div",
    { class: "ask-starters" },
    h("p", {}, "Ask anything about this check."),
    ASK_STARTERS.map((starter) =>
      h("button", { type: "button", onclick: () => ask(starter) }, starter),
    ),
  );
}

function renderTurn(turn, citations) {
  return h(
    "div",
    { class: "turn" },
    h(
      "div",
      { class: "conversation" },
      h(
        "p",
        { class: "turn-question" },
        turn.about && h("span", { class: "turn-about" }, turn.about),
        turn.question,
      ),
      turn.error
        ? h("p", { class: "notice" }, inlineMarkdown(turn.error))
        : markdown(turn.answer, citations),
    ),
  );
}

// Rail

async function refreshRail() {
  refreshHistory();
  try {
    const spend = await api("/spend");
    const total = Object.values(spend).reduce((sum, usd) => sum + usd, 0);
    document.getElementById("spend").textContent =
      total > 0 ? `Spent on paid sources so far: about $${total.toFixed(2)}` : "";
  } catch {
    document.getElementById("spend").textContent = "";
  }
}

async function refreshHistory() {
  const list = document.getElementById("history");
  try {
    const reports = await api("/reports");
    list.replaceChildren(
      ...(reports.length
        ? reports.map(historyItem)
        : [h("li", { class: "history-empty" }, "Your checks will appear here.")]),
    );
    markCurrentHistoryItem();
  } catch (error) {
    list.replaceChildren(h("li", { class: "notice" }, inlineMarkdown(error.message)));
  }
}

function historyItem(report) {
  const first = report.claims[0];
  const title = first?.text ?? report.excerpt ?? "Untitled check";
  const label = first?.label ?? "No claims found";
  const more = report.claims.length > 1 ? ` and ${report.claims.length - 1} more` : "";
  const tone = VERDICT_TONES[label] ?? "unproven";
  return h(
    "li",
    {},
    h(
      "a",
      { href: reportPath(report.id), "data-report": report.id },
      h("span", { class: "history-title" }, title),
      h(
        "span",
        { class: "history-meta" },
        h("span", {}, h("span", { class: `history-verdict toned tone-${tone}` }, label), more),
        h("span", {}, shortDate(report.created)),
      ),
    ),
  );
}

function markCurrentHistoryItem() {
  const [, kind, id] = location.hash.split("/");
  for (const link of document.querySelectorAll("#history a")) {
    const current = kind === "report" && link.dataset.report === id;
    if (current) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
}

function guide() {
  return h(
    "section",
    { class: "guide" },
    h("h1", {}, "How likely is it to be true?"),
    markdown(
      "Give Fakt a post, a video, an article or a screenshot. It pulls out each claim, reads what the research says, and tells you how far to trust it.\n\nEach verdict sits on a line from false to true. The diamond is Fakt's judgement. The squares are the sources, placed by which way they lean and sized by how much they counted.",
      new Map(),
    ),
  );
}

// Tiny Markdown renderer. It builds DOM nodes directly, so nothing in the source can
// become markup, and links only survive when they are http or https.

function markdown(source, citations) {
  const root = h("div", { class: "prose" });
  const lines = String(source ?? "").replaceAll("\r", "").split("\n");
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
    } else if (line.startsWith("```")) {
      const end = findLine(lines, index + 1, (candidate) => candidate.startsWith("```"));
      root.append(h("pre", {}, h("code", {}, lines.slice(index + 1, end).join("\n"))));
      index = end + 1;
    } else if (HEADING.test(line)) {
      const [, hashes, title] = line.match(/^(#+)\s+(.*)$/);
      root.append(h(`h${Math.min(hashes.length + 2, 5)}`, {}, inlineMarkdown(title, citations)));
      index += 1;
    } else if (/^(-{3,}|\*{3,})\s*$/.test(line)) {
      root.append(h("hr"));
      index += 1;
    } else if (line.startsWith(">")) {
      const end = findLine(lines, index, (candidate) => !candidate.startsWith(">"));
      const quoted = lines.slice(index, end).map((text) => text.replace(/^>\s?/, ""));
      root.append(h("blockquote", {}, inlineMarkdown(quoted.join(" "), citations)));
      index = end;
    } else if (/^\s*([-*+]|\d+\.)\s/.test(line)) {
      const end = findLine(lines, index, (candidate) => !/^\s*([-*+]|\d+\.)\s/.test(candidate));
      const items = lines
        .slice(index, end)
        .map((text) => h("li", {}, inlineMarkdown(text.replace(/^\s*([-*+]|\d+\.)\s+/, ""), citations)));
      root.append(h(/^\s*\d/.test(line) ? "ol" : "ul", {}, items));
      index = end;
    } else if (line.includes("|") && /^[\s|:-]+$/.test(lines[index + 1] ?? "")) {
      const end = findLine(lines, index, (candidate) => !candidate.includes("|"));
      root.append(markdownTable(lines.slice(index, end), citations));
      index = end;
    } else {
      const end = findLine(lines, index + 1, (candidate) => !candidate.trim() || BLOCK_START.test(candidate));
      root.append(h("p", {}, inlineMarkdown(lines.slice(index, end).join(" "), citations)));
      index = end;
    }
  }
  return root;
}

function findLine(lines, from, matches) {
  const found = lines.findIndex((line, index) => index >= from && matches(line));
  return found === -1 ? lines.length : found;
}

function markdownTable([header, , ...rows], citations) {
  const cells = (row) =>
    row
      .replace(/^\s*\|/, "")
      .replace(/\|\s*$/, "")
      .split("|")
      .map((cell) => cell.trim());
  return h(
    "div",
    { class: "table-scroll" },
    h(
      "table",
      {},
      h("thead", {}, h("tr", {}, cells(header).map((cell) => h("th", {}, inlineMarkdown(cell, citations))))),
      h(
        "tbody",
        {},
        rows.map((row) =>
          h("tr", {}, cells(row).map((cell) => h("td", {}, inlineMarkdown(cell, citations)))),
        ),
      ),
    ),
  );
}

function inlineMarkdown(text, citations = new Map()) {
  const pattern =
    /`([^`]+)`|\*\*(.+?)\*\*|(?<!\w)\*([^*\n]+)\*(?!\w)|(?<!\w)_([^_\n]+)_(?!\w)|\[([^\]]+)\]\(([^)\s]+)\)|\[([\w, -]+)\]|(https?:\/\/[^\s)<>]+)/g;
  const nodes = [];
  let position = 0;
  for (const match of String(text ?? "").matchAll(pattern)) {
    nodes.push(match.input.slice(position, match.index));
    position = match.index + match[0].length;
    const [whole, code, bold, star, underscore, linkText, linkUrl, cited, bareUrl] = match;
    if (code) nodes.push(h("code", {}, code));
    else if (bold) nodes.push(h("strong", {}, inlineMarkdown(bold, citations)));
    else if (star || underscore) nodes.push(h("em", {}, inlineMarkdown(star || underscore, citations)));
    else if (linkText) nodes.push(externalLink(linkUrl, linkText));
    else if (cited) nodes.push(...citationNodes(cited, whole, citations));
    else nodes.push(externalLink(bareUrl, bareUrl));
  }
  nodes.push(String(text ?? "").slice(position));
  const fragment = document.createDocumentFragment();
  fragment.append(...nodes.filter((node) => node !== ""));
  return fragment;
}

function citationNodes(cited, whole, citations) {
  const ids = cited.split(",").map((id) => id.trim());
  if (!ids.every((id) => citations.has(id))) return [whole];
  return ids.map((id) =>
    h(
      "a",
      {
        class: "cite",
        href: reportPath(state.report.id, "evidence"),
        "aria-label": `Source ${citations.get(id)}`,
        onclick: (event) => {
          event.preventDefault();
          jumpToSource(id);
        },
      },
      String(citations.get(id)),
    ),
  );
}

// Plumbing

function externalLink(url, text) {
  const safe = safeUrl(url);
  if (!safe) return document.createTextNode(text);
  return h("a", { href: safe, target: "_blank", rel: "noopener noreferrer" }, text);
}

function safeUrl(url) {
  try {
    const parsed = new URL(url);
    return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : null;
  } catch {
    return null;
  }
}

function h(tag, properties = {}, ...children) {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(properties)) {
    if (value === null || value === undefined) continue;
    if (name.startsWith("on")) element.addEventListener(name.slice(2), value);
    else element.setAttribute(name === "class" ? "class" : name, value);
  }
  element.append(...children.flat(Infinity).filter((child) => child || child === 0));
  return element;
}

function positioned(element, styles) {
  Object.assign(element.style, styles);
  return element;
}

function postJson(body) {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

async function api(path, options) {
  let response;
  try {
    response = await fetch(API_ROOT + path, options);
  } catch {
    throw new Error(SERVER_DOWN);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(errorText(body, response.status));
  return body;
}

function errorText({ detail }, status) {
  if (Array.isArray(detail)) return detail.map((item) => item.msg).join(". ");
  return detail || `The server answered with an error (${status}).`;
}

function shortDate(iso) {
  const date = new Date(iso);
  if (!iso || Number.isNaN(date.valueOf())) return "";
  return date.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

// Per-viewer conveniences. Storage can be missing, so losing it costs nothing.
function remember(key, value) {
  try {
    const stored = `fakt.${key}`;
    if (value === undefined) return JSON.parse(localStorage.getItem(stored));
    if (value === null) localStorage.removeItem(stored);
    else localStorage.setItem(stored, JSON.stringify(value));
  } catch {
    return null;
  }
  return value;
}
