// Fakt interface, shared by the side panel and the web app. Everything shown here comes
// from the internet, so it only ever reaches the page as text nodes, never as markup.

const inExtension = location.protocol === "chrome-extension:";
const PART_FINDINGS = { holds: "Holds up", fails: "Does not hold up" };
const LABEL_MEANINGS = {
  True: "Accurate as stated.",
  "Mostly true": "Right in substance, with a small error or a missing caveat.",
  Contested: "Good evidence points both ways, and the question is not settled.",
  Misleading:
    "The words can be defended, but the conclusion they point you to is false or unsupported.",
  "Mostly false": "The core of it is wrong, though it holds a grain of truth.",
  False: "Good evidence contradicts it.",
  Unproven: "There is too little evidence to tell either way.",
};
const CHANCE_MATCHES_VERDICT =
  "Here the verdict and this number say the same thing twice: the verdict names the band this chance falls in.";
const CHANCE_NOTES = {
  Misleading:
    "This number and the verdict answer different questions. The number scores the words exactly as said. The verdict judges the impression they leave, so a Misleading claim can still score high.",
  Contested: "The number sits near the middle because good evidence pulls both ways.",
  Unproven:
    "The number sits near the middle because there is too little evidence, not because the evidence is evenly split.",
};
const SIDES = [
  {
    title: "The case for the claim",
    argument: "strongest_for",
    stance: "Supports",
    none: "No source we found supports it.",
  },
  {
    title: "The case against it",
    argument: "strongest_against",
    stance: "Refutes",
    none: "No source we found counts against it.",
  },
];
const STRONGEST_SHOWN = 2;
const QUOTE_CHECKS = {
  verified: "Found word for word in the source, and a second check agreed it says this.",
  contradicted: "Found word for word in the source, but the passage around it cuts against this reading.",
};
const QUOTE_FOUND = "Found word for word in the source.";
const HOVER_PAUSE_MS = 250;
const NO_REFERENCES = { numbers: new Map(), terms: null };
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
  hint: null,
  dismissedHint: null,
};

// The one floating panel that shows the detail behind a word or a citation mark.
const hint = h("div", { class: "hint", role: "group", "aria-label": "More detail", hidden: "" });
const hintOpeners = new WeakMap();
let hintTimer;

start();

function start() {
  document.getElementById("paste-form").addEventListener("submit", submitPaste);
  pasteBox.addEventListener("paste", takePastedImage);
  document.getElementById("remove-image").addEventListener("click", () => setPastedImage(null));
  window.addEventListener("hashchange", route);
  document.addEventListener("pointerdown", (event) => {
    if (state.hint && !state.hint.wrapper.contains(event.target)) closeHint();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") dismissHint();
  });
  for (const moved of ["scroll", "resize"]) {
    window.addEventListener(moved, () => state.hint && placeHint(state.hint.trigger), {
      passive: true,
    });
  }

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
  closeHint();
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
  const references = referencesOf(report);
  main.replaceChildren(
    h("a", { class: "back", href: "#/" }, "All checks"),
    postHeader(report),
    tabBar(report, tab),
    h("div", {}, (views[tab] ?? summaryTab)(report, references)),
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

function summaryTab(report, references) {
  return [
    readingTip(report),
    h("div", {}, report.checks.map((check) => verdictBlock(report, check, references))),
    report.checks.length === 0 &&
      h("p", { class: "notice" }, "No checkable claims were found in this post."),
    profileBlock(report.profile),
    report.one_page &&
      h(
        "section",
        { class: "plain-english" },
        markdown(report.one_page, references),
      ),
  ];
}

function readingTip(report) {
  if (!report.checks.some((check) => check.verdict)) return null;
  const tip = "Numbered marks show the sources behind a statement. Hover over one, tap it, or press Enter on it.";
  if (report.glossary?.length > 0) {
    return h("p", { class: "reading-tip" }, `${tip} Words with a dotted underline explain themselves the same way.`);
  }
  const status = h("span", { role: "status" });
  const button = h("button", { type: "button" }, "Explain the technical terms");
  button.addEventListener("click", async () => {
    button.disabled = true;
    status.textContent = "Working out the terms. This takes about a minute.";
    try {
      const { glossary } = await api(`/reports/${encodeURIComponent(report.id)}/glossary`, {
        method: "POST",
      });
      if (glossary.length === 0) throw new Error("The terms could not be explained just now.");
      report.glossary = glossary;
      if (state.report === report) route();
    } catch (error) {
      button.disabled = false;
      status.textContent = error.message;
    }
  });
  return h("div", { class: "reading-tip" }, h("p", {}, tip), h("p", {}, button, " ", status));
}

function verdictBlock(report, check, references) {
  const { claim, verdict, sources } = check;
  const cited = claimReferences(references, check);
  const label = verdict?.label ?? "Not checked";
  const tone = VERDICT_TONES[label] ?? "unproven";
  return h(
    "article",
    { class: `verdict tone-${tone}` },
    h("h2", { class: "verdict-claim" }, claim.text),
    h("p", { class: "verdict-label" }, label),
    h(
      "p",
      { class: "verdict-meaning" },
      verdict ? LABEL_MEANINGS[label] : `We could not check this one. ${check.failure ?? ""}`,
    ),
    verdict?.summary &&
      h("div", { class: "verdict-summary" }, inlineMarkdown(verdict.summary, cited)),
    verdict?.parts?.length > 0 && claimParts(verdict.parts),
    verdict && chance(check, label),
    verdict && bothSides(check, cited),
    verdict?.what_would_change_this &&
      h(
        "section",
        { class: "would-change" },
        h("h3", {}, "What would change this verdict"),
        h("p", {}, inlineMarkdown(verdict.what_would_change_this, cited)),
      ),
    sources.length > 0 &&
      h(
        "details",
        { class: "evidence" },
        h("summary", {}, `Show all ${sources.length === 1 ? "the evidence" : `${sources.length} sources`}`),
        sourceList(report, check, cited),
      ),
  );
}

function claimParts(parts) {
  return h(
    "section",
    { class: "claim-parts" },
    h("h3", {}, "The claim, piece by piece"),
    h(
      "ul",
      {},
      parts.map((part) =>
        h(
          "li",
          { class: `part-${part.finding}` },
          h("strong", {}, `${PART_FINDINGS[part.finding] ?? "Unclear"}: `),
          part.statement,
        ),
      ),
    ),
  );
}

// The chance scores the wording alone, so it sits apart from the verdict and says how the
// two relate.
function chance(check, label) {
  const percent = Math.round(check.verdict.probability_true * 100);
  return h(
    "section",
    { class: "chance" },
    h(
      "h3",
      {},
      "Chance the statement is true as worded ",
      h("span", { class: "chance-figure" }, `${percent}%`),
    ),
    scale(check, percent),
    h("p", { class: "chance-note" }, CHANCE_NOTES[label] ?? CHANCE_MATCHES_VERDICT),
  );
}

function bothSides(check, references) {
  const entries = graded(check).filter(({ source }) => !source.retracted);
  const wasRead = ({ appraisal }) => Boolean(appraisal.key_finding);
  const sides = SIDES.map((side) => {
    const onSide = entries.filter(({ stance }) => stance?.word === side.stance);
    // Sources read in full come first: only they have a finding and a quote to show.
    const strongest = [...onSide.filter(wasRead), ...onSide.filter((entry) => !wasRead(entry))];
    return {
      ...side,
      argument: check.verdict[side.argument],
      strongest: strongest.slice(0, STRONGEST_SHOWN),
    };
  });
  if (sides.every(({ argument, strongest }) => !argument && strongest.length === 0)) return null;
  return h(
    "section",
    { class: "both-sides", "aria-label": "The case for and the case against" },
    sides.map(({ title, argument, stance, strongest, none }) =>
      h(
        "div",
        { class: `side tone-${STANCES[stance.toLowerCase()].tone}` },
        h("h3", {}, title),
        argument && h("p", { class: "side-argument" }, inlineMarkdown(argument, references)),
        h("h4", {}, "Strongest evidence"),
        strongest.length > 0
          ? h("ul", {}, strongest.map((entry) => strongestEvidence(entry, references)))
          : h("p", { class: "side-empty" }, none),
      ),
    ),
  );
}

function strongestEvidence(entry, references) {
  const { source, appraisal } = entry;
  const design = appraisal.design && appraisal.design !== "unknown" ? appraisal.design : null;
  const number = references.numbers.get(source.id);
  const trigger = h(
    "span",
    { class: "cite" },
    appraisal.quote ? `Quote and source ${number}` : `Source ${number}`,
  );
  const item = h(
    "li",
    {},
    h(
      "p",
      { class: "source-finding" },
      appraisal.key_finding ? inlineMarkdown(appraisal.key_finding, references) : source.title,
    ),
    h(
      "p",
      { class: "source-facts" },
      inlineMarkdown([design, source.year].filter(Boolean).join(", "), references),
      " ",
      trigger,
    ),
  );
  return hinted(item, trigger, () => [sourceCard(entry, references, { withFinding: false })]);
}

function scale(check, percent) {
  const marks = graded(check).filter(({ stance }) => stance);
  const heaviest = Math.max(...marks.map(({ appraisal }) => appraisal.weight), 0.01);
  return h(
    "div",
    {
      class: "scale",
      role: "img",
      "aria-label": `${percent} percent chance the statement is true as worded. ${stanceCount(marks)}`,
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

function evidenceTab(report, references) {
  if (report.checks.length === 0) {
    return h("p", { class: "notice" }, "This post had no claims to gather evidence for.");
  }
  return report.checks.map((check) => claimEvidence(report, check, references));
}

function claimEvidence(report, check, references) {
  const { claim, verdict } = check;
  const label = verdict?.label ?? "Unproven";
  const cited = claimReferences(references, check);
  const percent = verdict
    ? `. ${Math.round(verdict.probability_true * 100)}% chance the statement is true as worded.`
    : "";
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
          h("dd", {}, markdown(text, cited)),
        ]),
      ),
    h("h3", { class: "sources-title" }, sourceHeading(check)),
    sourceList(report, check, cited),
  );
}

function sourceHeading({ sources }) {
  return sources.length === 1 ? "One source" : `${sources.length} sources`;
}

function sourceList(report, check, references) {
  const entries = graded(check);
  const relevant = entries.filter(({ stance }) => stance);
  const offTopic = entries.filter(({ stance }) => !stance);
  return [
    h("ol", {}, relevant.map((entry) => sourceEntry(report, entry, references))),
    offTopic.length > 0 &&
      h(
        "details",
        { class: "off-topic" },
        h("summary", {}, `Left out of the verdict: ${offTopic.length}`),
        h("ol", {}, offTopic.map((entry) => sourceEntry(report, entry, references))),
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

function sourceEntry(report, { source, appraisal, stance }, references) {
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
      h("span", { class: "source-number" }, String(references.numbers.get(source.id))),
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
        h("p", { class: "source-finding" }, inlineMarkdown(appraisal.key_finding, references)),
      appraisal.excluded &&
        !source.retracted &&
        h("p", { class: "source-limits" }, appraisal.excluded),
      appraisal.cited_check &&
        h("p", { class: "source-finding" }, inlineMarkdown(appraisal.cited_check, references)),
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

function fullReportTab(report, references) {
  return h(
    "div",
    { class: "full-report" },
    report.detail
      ? markdown(report.detail, references)
      : h("p", { class: "notice" }, "This check has no long report."),
  );
}

// References: what report text points at. [source-id] becomes a numbered mark that shows
// the source, and a glossary term becomes a word that explains itself.

function referencesOf(report) {
  const numbers = new Map();
  for (const check of report.checks) {
    for (const { source } of graded(check)) {
      if (!numbers.has(source.id)) numbers.set(source.id, numbers.size + 1);
    }
  }
  return { numbers, terms: termMatcher(report.glossary) };
}

// A source is graded once per claim, so marks inside a claim show that claim's grading.
// A claim also explains each glossary term only where it first appears.
function claimReferences(references, check) {
  const evidence = new Map(graded(check).map((entry) => [entry.source.id, entry]));
  return { ...references, evidence, explainedTerms: new Set() };
}

function termMatcher(glossary = []) {
  if (glossary.length === 0) return null;
  const literal = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  // Only the first letter may change case, so "OR" never matches the word "or".
  const eitherCase = ([first, ...rest]) => {
    const [lower, upper] = [first.toLowerCase(), first.toUpperCase()];
    return (lower === upper ? literal(first) : `[${lower}${upper}]`) + literal(rest.join(""));
  };
  const longestFirst = glossary.map(({ term }) => term).sort((a, b) => b.length - a.length);
  return {
    meanings: new Map(glossary.map(({ term, meaning }) => [term.toLowerCase(), meaning])),
    pattern: new RegExp(`(?<![\\w-])(${longestFirst.map(eitherCase).join("|")})s?(?![\\w-])`, "g"),
  };
}

function sourceCard({ source, appraisal, stance }, references, { withFinding = true } = {}) {
  const design = appraisal.design && appraisal.design !== "unknown" ? appraisal.design : null;
  return h(
    "div",
    { class: "hint-source" },
    h(
      "p",
      { class: "source-title" },
      h("span", { class: "source-number" }, String(references.numbers.get(source.id))),
      externalLink(source.url, source.title),
    ),
    h("p", { class: "source-facts" }, [stance?.word, design, source.year].filter(Boolean).join(", ")),
    source.retracted && h("p", { class: "retraction" }, "Retracted. Don't rely on it."),
    withFinding && appraisal.key_finding && h("p", { class: "source-finding" }, appraisal.key_finding),
    appraisal.quote && h("blockquote", {}, appraisal.quote),
    appraisal.quote &&
      h("p", { class: "source-limits" }, QUOTE_CHECKS[appraisal.quote_check] ?? QUOTE_FOUND),
    h(
      "a",
      {
        class: "hint-more",
        href: reportPath(state.report.id, "evidence"),
        onclick: (event) => {
          event.preventDefault();
          jumpToSource(source.id);
        },
      },
      "See it in the evidence list",
    ),
  );
}

function jumpToSource(id) {
  closeHint();
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

// Hints: one floating panel for the detail behind a word or a mark. A mouse opens it by
// resting on the text, a keyboard by focus, a finger by a tap. Enter or a click pins it
// open; Escape, a second click or a tap elsewhere closes it.

function hinted(wrapper, trigger, content) {
  const open = (pinned) => openHint(wrapper, trigger, content, pinned);
  hintOpeners.set(wrapper, () => open(false));
  wrapper.classList.add("hinted");
  for (const [name, value] of [["role", "button"], ["tabindex", "0"], ["aria-expanded", "false"]]) {
    trigger.setAttribute(name, value);
  }
  const toggle = (event) => {
    if (hint.contains(event.target) || event.target.closest?.("a")) return;
    if (event.type === "click" && !getSelection().isCollapsed) return;
    event.preventDefault();
    event.stopPropagation();
    if (state.hint?.wrapper === wrapper && state.hint.pinned) closeHint();
    else open(true);
  };
  wrapper.addEventListener("click", toggle);
  trigger.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") toggle(event);
  });
  trigger.addEventListener("focus", () => {
    if (trigger.matches(":focus-visible") && trigger !== state.dismissedHint) open(false);
  });
  wrapper.addEventListener("focusout", (event) => {
    if (!wrapper.contains(event.relatedTarget)) closeUnpinnedHint();
  });
  wrapper.addEventListener("pointerenter", (event) => {
    if (event.pointerType === "mouse") afterPause(() => open(false));
  });
  wrapper.addEventListener("pointerleave", (event) => {
    if (event.pointerType !== "mouse") return;
    // Leaving a term inside a cited statement hands the panel back to the statement.
    const around = event.relatedTarget?.closest?.(".hinted");
    afterPause(hintOpeners.get(around) ?? closeUnpinnedHint);
  });
  return wrapper;
}

function afterPause(act) {
  clearTimeout(hintTimer);
  hintTimer = setTimeout(act, HOVER_PAUSE_MS);
}

function openHint(wrapper, trigger, content, pinned) {
  clearTimeout(hintTimer);
  const current = state.hint?.wrapper === wrapper;
  if (!current) {
    if (state.hint?.pinned && !pinned) return;
    closeHint();
    const close = h("button", { class: "quiet hint-close", type: "button", onclick: dismissHint }, "Close");
    hint.replaceChildren(close, ...content());
    wrapper.append(hint);
  }
  state.hint = { wrapper, trigger, pinned: pinned || (current && state.hint.pinned) };
  hint.hidden = false;
  trigger.setAttribute("aria-expanded", "true");
  placeHint(trigger);
}

function closeUnpinnedHint() {
  if (state.hint && !state.hint.pinned) closeHint();
}

// Closing by hand puts the keyboard back on the trigger, without that focus reopening it.
function dismissHint() {
  if (!state.hint) return;
  const { trigger } = state.hint;
  closeHint();
  state.dismissedHint = trigger;
  trigger.focus();
  state.dismissedHint = null;
}

function closeHint() {
  clearTimeout(hintTimer);
  if (!state.hint) return;
  state.hint.trigger.setAttribute("aria-expanded", "false");
  state.hint = null;
  hint.hidden = true;
  hint.remove();
}

// Wide windows anchor the panel to its trigger. Narrow ones leave it to the stylesheet,
// which docks it to the bottom edge.
function placeHint(trigger) {
  if (!matchMedia("(min-width: 620px)").matches) return positioned(hint, { left: "", top: "" });
  const at = trigger.getBoundingClientRect();
  const gap = 6;
  const edge = 8;
  const left = Math.max(edge, Math.min(at.left, innerWidth - hint.offsetWidth - edge));
  const below = at.bottom + gap;
  const above = at.top - hint.offsetHeight - gap;
  const fitsBelow = below + hint.offsetHeight <= innerHeight - edge;
  return positioned(hint, { left: `${left}px`, top: `${fitsBelow || above < edge ? below : above}px` });
}

// Ask

function askAboutSource(report, source) {
  state.askSource = source;
  const target = reportPath(report.id, "ask");
  if (location.hash === target) route();
  else location.hash = target;
}

function askTab(report, references) {
  const turns = remember(`chat.${report.id}`) ?? [];
  const conversation = h("div", { class: "conversation" });
  const paint = () => {
    conversation.replaceChildren(
      ...(turns.length ? turns.map((turn) => renderTurn(turn, references)) : [askStarters(ask)]),
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

function renderTurn(turn, references) {
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
        : markdown(turn.answer, references),
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
      "Give Fakt a post, a video, an article or a screenshot. It pulls out each claim, reads what the research says, and tells you how far to trust it.\n\nEach claim gets a verdict in words, then the chance that the statement is true exactly as worded, on a line from false to true. The diamond marks that chance. The squares are the sources, placed by which way they lean and sized by how much they counted.",
    ),
  );
}

// Tiny Markdown renderer. It builds DOM nodes directly, so nothing in the source can
// become markup, and links only survive when they are http or https.

function markdown(source, references) {
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
      root.append(h(`h${Math.min(hashes.length + 2, 5)}`, {}, inlineMarkdown(title, references)));
      index += 1;
    } else if (/^(-{3,}|\*{3,})\s*$/.test(line)) {
      root.append(h("hr"));
      index += 1;
    } else if (line.startsWith(">")) {
      const end = findLine(lines, index, (candidate) => !candidate.startsWith(">"));
      const quoted = lines.slice(index, end).map((text) => text.replace(/^>\s?/, ""));
      root.append(h("blockquote", {}, inlineMarkdown(quoted.join(" "), references)));
      index = end;
    } else if (/^\s*([-*+]|\d+\.)\s/.test(line)) {
      const end = findLine(lines, index, (candidate) => !/^\s*([-*+]|\d+\.)\s/.test(candidate));
      const items = lines
        .slice(index, end)
        .map((text) => h("li", {}, inlineMarkdown(text.replace(/^\s*([-*+]|\d+\.)\s+/, ""), references)));
      root.append(h(/^\s*\d/.test(line) ? "ol" : "ul", {}, items));
      index = end;
    } else if (line.includes("|") && /^[\s|:-]+$/.test(lines[index + 1] ?? "")) {
      const end = findLine(lines, index, (candidate) => !candidate.includes("|"));
      root.append(markdownTable(lines.slice(index, end), references));
      index = end;
    } else {
      const end = findLine(lines, index + 1, (candidate) => !candidate.trim() || BLOCK_START.test(candidate));
      root.append(h("p", {}, inlineMarkdown(lines.slice(index, end).join(" "), references)));
      index = end;
    }
  }
  return root;
}

function findLine(lines, from, matches) {
  const found = lines.findIndex((line, index) => index >= from && matches(line));
  return found === -1 ? lines.length : found;
}

function markdownTable([header, , ...rows], references) {
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
      h("thead", {}, h("tr", {}, cells(header).map((cell) => h("th", {}, inlineMarkdown(cell, references))))),
      h(
        "tbody",
        {},
        rows.map((row) =>
          h("tr", {}, cells(row).map((cell) => h("td", {}, inlineMarkdown(cell, references)))),
        ),
      ),
    ),
  );
}

function inlineMarkdown(text, references = NO_REFERENCES) {
  const pattern =
    /`([^`]+)`|\*\*(.+?)\*\*|(?<!\w)\*([^*\n]+)\*(?!\w)|(?<!\w)_([^_\n]+)_(?!\w)|\[([^\]]+)\]\(([^)\s]+)\)|\[([\w, -]+)\]|(https?:\/\/[^\s)<>]+)/g;
  const pieces = [];
  let position = 0;
  for (const match of String(text ?? "").matchAll(pattern)) {
    pieces.push(match.input.slice(position, match.index));
    position = match.index + match[0].length;
    const [whole, code, bold, star, underscore, linkText, linkUrl, cited, bareUrl] = match;
    if (code) pieces.push(h("code", {}, code));
    else if (bold) pieces.push(h("strong", {}, inlineMarkdown(bold, references)));
    else if (star || underscore) pieces.push(h("em", {}, inlineMarkdown(star || underscore, references)));
    else if (linkText) pieces.push(externalLink(linkUrl, linkText));
    else if (cited) pieces.push(citation(cited, whole, references));
    else pieces.push(externalLink(bareUrl, bareUrl));
  }
  pieces.push(String(text ?? "").slice(position));
  const fragment = document.createDocumentFragment();
  fragment.append(...statements(pieces.filter((piece) => piece !== ""), references));
  return fragment;
}

function citation(cited, whole, references) {
  const ids = cited.split(",").map((id) => id.trim());
  return ids.every((id) => references.numbers.has(id)) ? { ids } : whole;
}

// Groups each run of citations with the sentence it closes, so the whole statement can
// show its sources, and marks each glossary term once per claim, or once per passage
// outside a claim.
function statements(pieces, references) {
  const explained = references.explainedTerms ?? new Set();
  const nodes = [];
  let statement = [];
  const written = (piece) =>
    typeof piece === "string" ? termNodes(piece, references, explained) : [piece];
  const endStatement = () => {
    nodes.push(...statement.flatMap(written));
    statement = [];
  };
  const isCitation = (piece) => Boolean(piece?.ids);
  const isSeparator = (piece) => typeof piece === "string" && /^[\s,;]*$/.test(piece);
  for (let index = 0; index < pieces.length; index += 1) {
    const piece = pieces[index];
    if (isCitation(piece)) {
      const ids = [...piece.ids];
      while (
        isCitation(pieces[index + 1]) ||
        (isSeparator(pieces[index + 1]) && isCitation(pieces[index + 2]))
      ) {
        index += isCitation(pieces[index + 1]) ? 1 : 2;
        ids.push(...pieces[index].ids);
      }
      nodes.push(...citedStatement(statement.flatMap(written), [...new Set(ids)], references));
      statement = [];
    } else if (typeof piece === "string") {
      const lead = statement.length === 0 ? piece.match(/^[\s.,;:!?)]*/)[0] : "";
      const text = piece.slice(lead.length);
      const ends = [...text.matchAll(/[.!?]["”’)]*\s+/g)].at(-1);
      const end = ends ? ends.index + ends[0].length : 0;
      statement.push(lead + text.slice(0, end));
      if (lead || end) endStatement();
      statement.push(text.slice(end));
      statement = statement.filter((part) => part !== "");
    } else {
      statement.push(piece);
    }
  }
  endStatement();
  return nodes;
}

function citedStatement(said, ids, references) {
  const entries = ids.map((id) => references.evidence?.get(id)).filter(Boolean);
  if (entries.length === 0) return [...said, ...ids.map((id) => sourceJump(id, references))];
  const numbers = ids.map((id) => references.numbers.get(id)).join(", ");
  const mark = h(
    "span",
    { class: "cite", "aria-label": `Show source${ids.length === 1 ? "" : "s"} ${numbers}` },
    numbers,
  );
  const cards = () => entries.map((entry) => sourceCard(entry, references));
  return [hinted(h("span", { class: "cited" }, said, mark), mark, cards)];
}

function sourceJump(id, references) {
  return h(
    "a",
    {
      class: "cite",
      href: reportPath(state.report.id, "evidence"),
      "aria-label": `Source ${references.numbers.get(id)}`,
      onclick: (event) => {
        event.preventDefault();
        jumpToSource(id);
      },
    },
    String(references.numbers.get(id)),
  );
}

function termNodes(text, { terms }, explained) {
  if (!terms) return [text];
  const nodes = [];
  let position = 0;
  for (const match of text.matchAll(terms.pattern)) {
    const term = match[1].toLowerCase();
    if (explained.has(term)) continue;
    explained.add(term);
    const word = h("span", { class: "term" }, match[0]);
    const meaning = () => [h("p", { class: "hint-term" }, match[1]), h("p", {}, terms.meanings.get(term))];
    nodes.push(text.slice(position, match.index), hinted(h("span", {}, word), word, meaning));
    position = match.index + match[0].length;
  }
  nodes.push(text.slice(position));
  return nodes.filter((node) => node !== "");
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
