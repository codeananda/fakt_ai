// Runs the extension's scripts against a small fake browser. Usage: node <file> <scenario>.
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const EXTENSION = path.join(__dirname, "..", "extension");

class FakeElement {
  constructor(tag = "div") {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
    this.listeners = {};
    this.attributes = {};
    this.classList = { add() {}, remove() {}, toggle() {} };
    this.style = {};
    this.value = "";
    this.textContent = "";
  }
  remove() {}
  focus() {
    this.listeners.focus?.();
  }
  matches() {
    return true;
  }
  contains(other) {
    return other === this || this.children.some((c) => c instanceof FakeElement && c.contains(other));
  }
  // Every element below this one, in document order.
  get all() {
    const below = this.children.filter((c) => c instanceof FakeElement);
    return below.flatMap((c) => [c, ...c.all]);
  }
  withClass(name) {
    return this.all.filter((c) => (c.attributes.class ?? "").split(" ").includes(name));
  }
  append(...items) {
    this.children.push(...items);
  }
  replaceChildren(...items) {
    this.children = items;
  }
  setAttribute(name, value) {
    this.attributes[name] = value;
  }
  addEventListener(name, listener) {
    this.listeners[name] = listener;
  }
  querySelector() {
    return new FakeElement();
  }
  get text() {
    const parts = this.children.map((c) => (c instanceof FakeElement ? c.text : String(c)));
    return [this.textContent, ...parts].join("");
  }
}

function fakeDocument() {
  const byId = {};
  const listeners = {};
  return {
    listeners,
    body: new FakeElement("body"),
    createElement: (tag) => new FakeElement(tag),
    createTextNode: (text) => String(text),
    createDocumentFragment: () => new FakeElement("fragment"),
    getElementById: (id) => (byId[id] ??= new FakeElement()),
    querySelectorAll: () => [],
    addEventListener: (name, listener) => (listeners[name] = listener),
  };
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => ((resolve = yes), (reject = no)));
  return { promise, resolve, reject };
}

function loadApp(fetchReply) {
  const hashListeners = [];
  const context = {
    console,
    document: fakeDocument(),
    location: { hash: "", protocol: "http:", replace() {} },
    window: {
      addEventListener: (name, listener) => name === "hashchange" && hashListeners.push(listener),
    },
    setTimeout,
    clearTimeout,
    getSelection: () => ({ isCollapsed: true }),
    matchMedia: () => ({ matches: false }),
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    fetch: (url, options) => fetchReply(url, options),
    EventSource: class {
      close() {}
    },
    encodeURIComponent,
  };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(EXTENSION, "ui", "app.js"), "utf8"), context);
  context.navigate = (hash) => {
    context.location.hash = hash;
    hashListeners.forEach((listener) => listener());
  };
  return context;
}

const json = (body, ok = true) =>
  Promise.resolve({ ok, status: ok ? 200 : 500, json: async () => body });

function loadBackground(storage) {
  const listeners = {};
  const chrome = {
    sidePanel: { setPanelBehavior() {}, open() {} },
    runtime: {
      onInstalled: { addListener() {} },
      onMessage: { addListener: (listener) => (listeners.message = listener) },
    },
    contextMenus: { create() {}, onClicked: { addListener() {} } },
    commands: { onCommand: { addListener() {} } },
    storage: { session: storage },
  };
  const context = vm.createContext({ chrome, console });
  vm.runInContext(fs.readFileSync(path.join(EXTENSION, "background.js"), "utf8"), context);
  return (message) =>
    new Promise((resolve) => assert.equal(listeners.message(message, {}, resolve), true));
}

function slowStorage(items) {
  const pause = () => new Promise((resolve) => setTimeout(resolve, 1));
  return {
    async get(key) {
      await pause();
      return key in items ? { [key]: items[key] } : {};
    },
    async remove(key) {
      await pause();
      delete items[key];
    },
  };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

// A made-up check: one claim that is literally defensible and still misleading. Sources are
// numbered by weight, so s2 is source 1, s3 is source 2 and s1 is source 3.
function syntheticReport(fields = {}) {
  const source = (id, title) => ({ id, title, url: `https://example.org/${id}`, year: 2020 });
  const appraisal = (source_id, stance, weight, more = {}) => ({
    source_id,
    stance,
    weight,
    design: "cohort study",
    key_finding: `Finding of ${source_id}.`,
    ...more,
  });
  return {
    id: "r1",
    capture: { platform: "text", text: "A post." },
    profile: { signal_vs_hype: 0.5, fact_vs_fiction: 0.5, tactics: [] },
    checks: [
      {
        claim: { id: "c1", text: "A review found tea shortens lives." },
        sources: [source("s1", "The tea review"), source("s2", "A big trial"), source("s3", "Another")],
        appraisals: [
          appraisal("s1", "supports", 0.2),
          appraisal("s2", "refutes", 0.6, { quote: "No effect was seen.", quote_check: "verified" }),
          appraisal("s3", "refutes", 0.4),
        ],
        verdict: {
          label: "Misleading",
          probability_true: 0.6,
          summary: "The review exists [s1]. A trial with an OR of 0.99 found no effect [s2][s3].",
          reasoning: "",
          strongest_for: "The review says so [s1].",
          strongest_against: "The cohort study or trial found nothing [s2].",
          what_would_change_this: "A second trial.",
          parts: [{ statement: "A review said this", finding: "holds" }],
        },
      },
    ],
    glossary: [
      { term: "cohort study", meaning: "Research that follows a group over time." },
      { term: "OR", meaning: "Odds ratio, a measure of how two things go together." },
    ],
    ...fields,
  };
}

function renderedClaim(app, report) {
  app.report = report;
  vm.runInContext("state.report = report", app);
  return app.verdictBlock(report, report.checks[0], app.referencesOf(report));
}

// A report with no glossary yet, opened on its summary, and the button that asks for one.
async function openedWithoutGlossary(glossaryReply) {
  const report = syntheticReport();
  delete report.glossary;
  const calls = [];
  const app = loadApp((url, options) => {
    calls.push([String(url), options?.method]);
    if (String(url).endsWith("/glossary")) return glossaryReply();
    if (String(url).endsWith("/ask")) return new Promise(() => {});
    return String(url).endsWith("/reports/r1") ? json(report) : json([]);
  });
  app.navigate("#/report/r1/summary");
  await flush();
  await flush();
  const main = vm.runInContext("main", app);
  const tip = main.withClass("reading-tip")[0];
  const button = tip.all.find((element) => element.tag === "button");
  return { app, main, tip, button, calls };
}

const click = (target = { closest: () => null }) => ({
  type: "click",
  target,
  preventDefault() {},
  stopPropagation() {},
});
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const HOVER_PAUSE_MS = 300;

// The first claim's summary: two cited statements, their marks and the one hint panel.
function citedSummary() {
  const app = loadApp(() => json([]));
  const card = renderedClaim(app, syntheticReport());
  const [first, second] = card.withClass("verdict-summary")[0].withClass("cited");
  const mark = second.withClass("cite")[0];
  return { app, first, second, mark, panel: vm.runInContext("hint", app) };
}

const scenarios = {
  async "stray hashes render as paragraphs"() {
    const app = loadApp(() => json([]));
    for (const line of ["#hashtag", "####### heading", "#a\n#b", "text\n#tag\nmore"]) {
      const rendered = app.markdown(line);
      assert.equal(rendered.children[0].tag, "p");
    }
    assert.equal(app.markdown("#tag\n# Title").children[1].tag, "h3");
    assert.equal(app.markdown("words\n# Title").children[1].tag, "h3");
  },

  async "a misleading claim says how its chance differs from its verdict"() {
    const card = renderedClaim(loadApp(() => json([])), syntheticReport());
    assert.match(card.text, /The words can be defended/);
    assert.match(card.text, /Chance the statement is true as worded 60%/);
    assert.match(card.text, /answer different questions/);
    assert.doesNotMatch(card.text, /likely true/);
    assert.match(card.text, /Holds up: A review said this/);
  },

  async "a firm verdict says its chance and its label agree"() {
    const report = syntheticReport();
    Object.assign(report.checks[0].verdict, { label: "Mostly false", probability_true: 0.2 });
    const card = renderedClaim(loadApp(() => json([])), report);
    assert.match(card.text, /The core of it is wrong/);
    assert.match(card.text, /say the same thing twice/);
  },

  async "a claim that could not be checked says so on every tab"() {
    const failed = {
      claim: { id: "c9", text: "Second." },
      sources: [],
      appraisals: [],
      verdict: null,
      failure: "search broke",
    };
    const report = syntheticReport({ checks: [failed] });
    const app = loadApp(() => json([]));
    const card = renderedClaim(app, report);
    assert.equal(card.withClass("verdict-label")[0].text, "Not checked");
    assert.match(card.text, /We could not check this one\. search broke/);
    assert.equal(card.withClass("chance").length + card.withClass("both-sides").length, 0);
    assert.equal(app.readingTip(report), null);
    const evidence = app.claimEvidence(report, failed, app.referencesOf(report));
    assert.equal(evidence.withClass("claim-evidence-verdict")[0].text, "Not checked");
    const inHistory = app.historyItem({ id: "r1", claims: [{ text: "Second.", label: null }] });
    assert.equal(inHistory.withClass("history-verdict")[0].text, "Not checked");
  },

  async "a contested or unproven claim says why its chance stays off the ends"() {
    for (const [label, reason] of [["Contested", /pulls both ways/], ["Unproven", /too little evidence/]]) {
      const report = syntheticReport();
      Object.assign(report.checks[0].verdict, { label, probability_true: 0.7 });
      const note = renderedClaim(loadApp(() => json([])), report).withClass("chance-note")[0].text;
      assert.match(note, reason);
      assert.match(note, /stays away from both ends/);
    }
  },

  async "a source panel says how its quote was checked and whether it says what was claimed"() {
    const report = syntheticReport();
    report.checks[0].appraisals[0].cited_check = "No, it does not say this.";
    const app = loadApp(() => json([]));
    const [first, second] = renderedClaim(app, report).withClass("verdict-summary")[0].withClass("cited");
    const panel = vm.runInContext("hint", app);
    first.listeners.click(click());
    assert.match(panel.text, /Finding of s1\.No, it does not say this\./);
    assert.doesNotMatch(panel.text, /word for word/);
    second.listeners.click(click());
    assert.match(panel.text, /agreed with this reading of the passage\..*Finding of s3\./);
  },

  async "each side shows its case and its strongest evidence"() {
    const card = renderedClaim(loadApp(() => json([])), syntheticReport());
    const [caseFor, caseAgainst] = card.withClass("side").map((side) => side.text);
    assert.match(caseFor, /The review says so/);
    assert.match(caseFor, /Finding of s1\./);
    assert.match(caseAgainst, /Finding of s2\..*Quote and source 1.*Finding of s3\./);
    assert.doesNotMatch(caseAgainst, /Finding of s1\./);
  },

  async "a verdict with nothing on either side shows no sides"() {
    const report = syntheticReport();
    Object.assign(report.checks[0], { sources: [], appraisals: [] });
    Object.assign(report.checks[0].verdict, { strongest_for: "", strongest_against: "" });
    const card = renderedClaim(loadApp(() => json([])), report);
    assert.equal(card.withClass("both-sides").length, 0);
    assert.equal(card.withClass("chance").length, 1);
  },

  async "a retracted source is never a side's strongest evidence"() {
    const report = syntheticReport();
    report.checks[0].sources[1].retracted = true;
    const [, caseAgainst] = renderedClaim(loadApp(() => json([])), report).withClass("side");
    assert.doesNotMatch(caseAgainst.text, /Finding of s2\./);
    assert.match(caseAgainst.text, /Finding of s3\./);
  },

  async "a source that was read in full outranks a heavier one that was not"() {
    const report = syntheticReport();
    const check = report.checks[0];
    check.sources.push({ id: "s4", title: "Unread", url: "https://example.org/s4" });
    check.appraisals.push({ source_id: "s4", stance: "refutes", weight: 0.9, design: "unknown" });
    const [, caseAgainst] = renderedClaim(loadApp(() => json([])), report).withClass("side");
    assert.match(caseAgainst.text, /Finding of s2\..*Finding of s3\./);
    assert.doesNotMatch(caseAgainst.text, /Unread/);
    // With one fewer read source, the unread one fills the second place, by its title.
    check.appraisals[2].key_finding = "";
    const [, sparser] = renderedClaim(loadApp(() => json([])), report).withClass("side");
    assert.match(sparser.text, /Finding of s2\..*Unread/);
  },

  async "a side with no evidence says so"() {
    const report = syntheticReport();
    report.checks[0].appraisals.shift();
    const [caseFor] = renderedClaim(loadApp(() => json([])), report).withClass("side");
    assert.match(caseFor.text, /No source we found supports it\./);
  },

  async "a cited statement opens the sources it rests on"() {
    const app = loadApp(() => json([]));
    const card = renderedClaim(app, syntheticReport());
    const [first, second] = card.withClass("verdict-summary")[0].withClass("cited");
    assert.equal(first.text, "The review exists 3");
    assert.equal(second.text, "A trial with an OR of 0.99 found no effect 1, 2");
    const mark = second.withClass("cite")[0];
    assert.equal(mark.attributes.role, "button");
    assert.equal(mark.attributes.tabindex, "0");
    second.listeners.click(click());
    const panel = vm.runInContext("hint", app);
    assert.equal(mark.attributes["aria-expanded"], "true");
    assert.equal(panel.hidden, false);
    assert.equal(second.contains(panel), true);
    assert.match(panel.text, /A big trial.*No effect was seen\..*a second check agreed.*Another/);
    assert.doesNotMatch(panel.text, /it says this/);
    second.listeners.click(click());
    assert.equal(mark.attributes["aria-expanded"], "false");
    assert.equal(panel.hidden, true);
  },

  async "Escape, Close, a tap elsewhere and leaving the page each close the panel"() {
    const { app, first, second, mark, panel } = citedSummary();
    const closers = [
      () => app.document.listeners.keydown({ key: "Escape" }),
      () => panel.withClass("hint-close")[0].listeners.click(),
      () => app.document.listeners.pointerdown({ target: first }),
      () => app.navigate("#/"),
    ];
    for (const close of closers) {
      second.listeners.click(click());
      assert.equal(mark.attributes["aria-expanded"], "true");
      close();
      // Escape and Close put focus back on the mark, and that focus must not reopen it.
      assert.equal(mark.attributes["aria-expanded"], "false");
      assert.equal(panel.hidden, true);
    }
    second.listeners.click(click());
    app.document.listeners.keydown({ key: "a" });
    app.document.listeners.pointerdown({ target: mark });
    second.listeners.click(click(panel.all[0]));
    mark.listeners.keydown({ ...click(), type: "keydown", key: "Tab" });
    assert.equal(mark.attributes["aria-expanded"], "true");
    mark.listeners.keydown({ ...click(), type: "keydown", key: " " });
    assert.equal(mark.attributes["aria-expanded"], "false");
  },

  async "a mouse resting on a statement opens its panel, and leaving closes it"() {
    const { second, mark } = citedSummary();
    second.listeners.pointerenter({ pointerType: "touch" });
    await pause(HOVER_PAUSE_MS);
    assert.equal(mark.attributes["aria-expanded"], "false");
    second.listeners.pointerenter({ pointerType: "mouse" });
    assert.equal(mark.attributes["aria-expanded"], "false");
    await pause(HOVER_PAUSE_MS);
    assert.equal(mark.attributes["aria-expanded"], "true");
    second.listeners.pointerleave({ pointerType: "mouse", relatedTarget: null });
    await pause(HOVER_PAUSE_MS);
    assert.equal(mark.attributes["aria-expanded"], "false");
  },

  async "a pinned panel is not taken over by a mouse resting on other text"() {
    const { first, second, mark } = citedSummary();
    second.listeners.click(click());
    first.listeners.pointerenter({ pointerType: "mouse" });
    await pause(HOVER_PAUSE_MS);
    assert.equal(mark.attributes["aria-expanded"], "true");
    assert.equal(first.withClass("cite")[0].attributes["aria-expanded"], "false");
  },

  async "a pinned panel whose text was redrawn away does not block the next one"() {
    const { first, second, mark } = citedSummary();
    first.listeners.click(click());
    first.isConnected = false;
    mark.focus();
    assert.equal(mark.attributes["aria-expanded"], "true");
    assert.equal(first.withClass("cite")[0].attributes["aria-expanded"], "false");
    assert.equal(second.contains(first), false);
  },

  async "a glossary term explains itself, and an abbreviation is not mistaken for a word"() {
    const app = loadApp(() => json([]));
    const card = renderedClaim(app, syntheticReport());
    const terms = card.withClass("term");
    // Each term is explained once per claim, where it first appears, and "or" is left alone.
    assert.deepEqual(terms.map((term) => term.text), ["OR", "cohort study"]);
    terms[1].listeners.keydown({ ...click(), type: "keydown", key: "Enter" });
    assert.match(vm.runInContext("hint", app).text, /Research that follows a group over time\./);
  },

  async "a term matches whole words and its plural, and case only on its first letter"() {
    const app = loadApp(() => json([]));
    const glossary = [
      { term: "OR", meaning: "Odds ratio." },
      { term: "cohort", meaning: "A group followed over time." },
      { term: "cohort study", meaning: "Research that follows a group." },
      { term: "p-value (two-sided)", meaning: "A measure of surprise." },
      { term: "", meaning: "Nothing to match." },
    ];
    const references = app.referencesOf(syntheticReport({ glossary }));
    const terms = (text) =>
      app.inlineMarkdown(text, references).withClass("term").map((term) => term.text);
    assert.deepEqual(terms("Tea or coffee, for MORE or less. The OR was 0.99."), ["OR"]);
    assert.deepEqual(terms("Three cohort studys and Cohorts."), ["cohort studys", "Cohorts"]);
    assert.deepEqual(terms("The p-value (two-sided) was low; a sub-cohort too."), ["p-value (two-sided)"]);
    assert.deepEqual(terms("A cohort-based design, and a COHORT."), []);
  },

  async "a term is explained where it is first read, not where it is first emphasised"() {
    const app = loadApp(() => json([]));
    const passage = app.inlineMarkdown(
      "A cohort study showed **the cohort study** was fine.",
      app.referencesOf(syntheticReport()),
    );
    assert.equal(passage.withClass("term").length, 1);
    assert.equal(passage.all.find((element) => element.tag === "strong").withClass("term").length, 0);
    const slanted = app.inlineMarkdown(
      "A cohort study showed *the cohort study* was fine.",
      app.referencesOf(syntheticReport()),
    );
    assert.equal(slanted.withClass("term").length, 1);
    assert.equal(slanted.all.find((element) => element.tag === "em").withClass("term").length, 0);
  },

  async "a heading inside a claim's reasoning does not explain its terms again"() {
    const app = loadApp(() => json([]));
    const report = syntheticReport();
    const cited = app.claimReferences(app.referencesOf(report), report.checks[0]);
    const reasoning = app.markdown("A cohort study.\n\n## Detail\n\nThe cohort study again.", cited);
    assert.equal(reasoning.withClass("term").length, 1);
  },

  async "a source's limitations explain their terms like its finding does"() {
    const app = loadApp(() => json([]));
    const report = syntheticReport();
    report.checks[0].appraisals[1].limitations = "It was a cohort study, not a trial.";
    vm.runInContext("state.report = { id: 'r1' }", app);
    const references = app.claimReferences(app.referencesOf(report), report.checks[0]);
    const [heaviest] = app.sourceList(report, report.checks[0], references)[0].withClass("source");
    const limits = heaviest.withClass("source-limits")[0];
    assert.equal(limits.text, "Limits: It was a cohort study, not a trial.");
    assert.deepEqual(limits.withClass("term").map((term) => term.text), ["cohort study"]);
  },

  async "a document outside a claim explains a term once per section"() {
    const app = loadApp(() => json([]));
    const page = app.markdown(
      "## One\n\nA cohort study.\n\n- The cohort study again.\n\n## Two\n\nA cohort study.",
      app.referencesOf(syntheticReport()),
    );
    assert.equal(page.withClass("term").length, 2);
    assert.equal(page.children.at(-1).withClass("term").length, 1);
  },

  async "a stored glossary too large to match does not stop the report opening"() {
    const glossary = [{ term: "x".repeat(40_000), meaning: "Far too long to be a term." }];
    const card = renderedClaim(loadApp(() => json([])), syntheticReport({ glossary }));
    assert.match(card.text, /Misleading/);
    assert.equal(card.withClass("term").length, 0);
  },

  async "only the sentence a citation closes is shown as resting on it"() {
    const app = loadApp(() => json([]));
    const report = syntheticReport({ glossary: [] });
    vm.runInContext("state.report = { id: 'r1' }", app);
    const references = app.claimReferences(app.referencesOf(report), report.checks[0]);
    const cited = (text) =>
      app.inlineMarkdown(text, references).withClass("cited").map((statement) => statement.text);
    assert.deepEqual(
      cited("Tea is safe. Nobody doubts it. A trial found nothing [s2]. More."),
      ["A trial found nothing 1"],
    );
    assert.deepEqual(cited("It is thin. 136 studies were counted [s2]."), ["136 studies were counted 1"]);
    for (const whole of [
      "Smith et al. found no effect in adults vs. children",
      "Smith et al. (2020) found no effect",
      "Smith et al.  found no effect",
      "The drug vs. 12% on placebo cut deaths",
      "As Dr. Hooper reported, risk fell",
    ]) {
      assert.deepEqual(cited(`It is thin. ${whole} [s2].`), [`${whole} 1`]);
    }
    assert.deepEqual(cited("Smith et al. **found** no effect [s2]."), ["Smith et al. found no effect 1"]);
    assert.deepEqual(cited("Tea is fine [s1], [s2]."), ["Tea is fine 3, 1"]);
    assert.deepEqual(cited("See [s2][s2]."), ["See 1"]);
    assert.deepEqual(cited("[s2] says so."), ["1"]);
    assert.deepEqual(cited("Before. **Tea** is *really* fine [s1]."), ["Tea is really fine 3"]);
    assert.deepEqual(cited("Tea is fine [zz] and [s1, zz]."), []);
    assert.equal(app.inlineMarkdown("Tea is fine [zz].", references).text, "Tea is fine [zz].");
  },

  async "a report stored without a glossary still renders and offers one"() {
    const report = syntheticReport();
    delete report.glossary;
    const app = loadApp(() => json([]));
    const card = renderedClaim(app, report);
    assert.equal(card.withClass("term").length, 0);
    assert.equal(card.withClass("cited").length > 0, true);
    assert.match(app.readingTip(report).text, /Explain the technical terms/);
    assert.doesNotMatch(app.readingTip(syntheticReport()).text, /Explain the technical terms/);
    // A report with nothing to explain says nothing about dotted words and asks for nothing.
    const plain = app.readingTip(syntheticReport({ glossary: [] })).text;
    assert.doesNotMatch(plain, /Explain the technical terms|dotted/);
    assert.equal(renderedClaim(app, syntheticReport({ glossary: null })).withClass("term").length, 0);
  },

  async "the terms button fetches the glossary and redraws the summary with it"() {
    const terms = [{ term: "OR", meaning: "Odds ratio." }];
    const { app, main, button, calls } = await openedWithoutGlossary(() => json({ glossary: terms }));
    await button.listeners.click();
    assert.deepEqual(calls.at(-1), ["/reports/r1/glossary", "POST"]);
    assert.deepEqual(vm.runInContext("state.report.glossary", app), terms);
    assert.deepEqual(main.withClass("term").map((term) => term.text), ["OR"]);
    assert.doesNotMatch(main.withClass("reading-tip")[0].text, /Explain the technical terms/);
  },

  async "a glossary that could not be written says so and lets the reader retry"() {
    const busy = () => json({ detail: "Try again later." }, false);
    const { app, tip, button } = await openedWithoutGlossary(busy);
    await button.listeners.click();
    assert.equal(button.disabled, false);
    assert.match(tip.text, /Try again later\./);
    assert.equal(vm.runInContext("state.report.glossary", app), undefined);
  },

  async "a glossary that lands while a question is being asked redraws nothing"() {
    const reply = deferred();
    const { app, main, button } = await openedWithoutGlossary(() => reply.promise);
    const asked = button.listeners.click();
    app.navigate("#/report/r1/ask");
    const form = main.all.find((element) => element.tag === "form");
    form.all.find((element) => element.tag === "textarea").value = "Why?";
    form.listeners.submit({ preventDefault() {} });
    const drawn = vm.runInContext("state.navigation", app);
    reply.resolve(await json({ glossary: [{ term: "OR", meaning: "Odds ratio." }] }));
    await asked;
    assert.equal(vm.runInContext("state.navigation", app), drawn);
    assert.match(main.text, /Why\?.*Reading the evidence/);
    assert.equal(vm.runInContext("state.report.glossary.length", app), 1);
  },

  async "a report with no terms to explain stops offering the button once asked"() {
    const { main, button, tip } = await openedWithoutGlossary(() => json({ glossary: [] }));
    await button.listeners.click();
    assert.doesNotMatch(tip.text, /could not be explained/);
    assert.doesNotMatch(main.withClass("reading-tip")[0].text, /Explain the technical terms|dotted/);
  },

  async "the summary is redrawn however its address is spelt"() {
    for (const address of ["#/report/r1", "#/report/r1/", "#/report/r1/bogus"]) {
      const terms = [{ term: "OR", meaning: "Odds ratio." }];
      const { app, main, button } = await openedWithoutGlossary(() => json({ glossary: terms }));
      app.location.hash = address;
      await button.listeners.click();
      assert.deepEqual(main.withClass("term").map((term) => term.text), ["OR"], address);
    }
  },

  async "a glossary that lands after the reader left the report redraws nothing"() {
    const reply = deferred();
    const { app, button } = await openedWithoutGlossary(() => reply.promise);
    const asked = button.listeners.click();
    app.navigate("#/");
    const drawn = vm.runInContext("state.navigation", app);
    reply.resolve(await json({ glossary: [{ term: "OR", meaning: "Odds ratio." }] }));
    await asked;
    assert.equal(vm.runInContext("state.navigation", app), drawn);
  },

  async "citations outside a claim still jump to the evidence list"() {
    const app = loadApp(() => json([]));
    const report = syntheticReport();
    renderedClaim(app, report);
    const answer = app.markdown("It found nothing [s2].", app.referencesOf(report));
    assert.equal(answer.withClass("cited").length, 0);
    assert.equal(answer.withClass("cite")[0].tag, "a");
  },

  async "two panels take a queued check only once"() {
    const items = { queuedCheck: { payload: { text: "x" } } };
    const send = loadBackground(slowStorage(items));
    const taken = await Promise.all([
      send({ type: "take-queued-check" }),
      send({ type: "take-queued-check" }),
    ]);
    assert.equal(taken.filter(Boolean).length, 1);
    assert.equal(taken[0].payload.text, "x");
    assert.deepEqual(items, {});
  },

  async "an empty queue yields nothing"() {
    const send = loadBackground(slowStorage({}));
    assert.equal(await send({ type: "take-queued-check" }), null);
  },

  async "a late report does not replace the one on screen"() {
    const replies = { a: deferred(), b: deferred() };
    const app = loadApp((url) => {
      const id = String(url).match(/reports\/(\w+)$/)?.[1];
      return id ? replies[id].promise : json([]);
    });
    app.navigate("#/report/a/summary");
    app.navigate("#/report/b/summary");
    app.renderReport = () => {};
    replies.b.resolve(await json({ id: "b" }));
    await flush();
    replies.a.resolve(await json({ id: "a" }));
    await flush();
    assert.equal(vm.runInContext("state.report.id", app), "b");
  },

  async "a late failure does not replace the view on screen"() {
    const reply = deferred();
    const app = loadApp((url) => (String(url).includes("/reports/a") ? reply.promise : json([])));
    app.navigate("#/report/a/summary");
    app.navigate("#/");
    const home = vm.runInContext("main.children.slice()", app);
    reply.resolve(await json({}, false));
    await flush();
    assert.deepEqual(vm.runInContext("main.children", app), home);
    assert.equal(vm.runInContext("state.report", app), null);
  },

  async "a failed submission keeps what was entered"() {
    const app = loadApp((url, options) => (options?.method ? Promise.reject(new Error()) : json([])));
    const box = vm.runInContext("pasteBox", app);
    box.value = "A claim";
    vm.runInContext('state.pastedImage = "data:image/png;base64,x"', app);
    await app.submitPaste({ preventDefault() {} });
    assert.equal(box.value, "A claim");
    assert.equal(vm.runInContext("state.pastedImage", app), "data:image/png;base64,x");
  },

  async "a successful submission clears the box"() {
    const app = loadApp((url, options) => (options?.method ? json({ id: "j1" }) : json([])));
    const box = vm.runInContext("pasteBox", app);
    box.value = "A claim";
    await app.submitPaste({ preventDefault() {} });
    assert.equal(box.value, "");
    assert.equal(app.location.hash, "#/job/j1");
  },
};

scenarios[process.argv[2]]().catch((error) => {
  console.error(error);
  process.exit(1);
});
