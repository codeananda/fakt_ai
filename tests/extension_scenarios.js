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
  return {
    body: new FakeElement("body"),
    createElement: (tag) => new FakeElement(tag),
    createTextNode: (text) => String(text),
    createDocumentFragment: () => new FakeElement("fragment"),
    getElementById: (id) => (byId[id] ??= new FakeElement()),
    querySelectorAll: () => [],
    addEventListener() {},
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

const click = (target) => ({
  type: "click",
  target: { closest: () => null },
  preventDefault() {},
  stopPropagation() {},
});

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

  async "each side shows its case and its strongest evidence"() {
    const card = renderedClaim(loadApp(() => json([])), syntheticReport());
    const [caseFor, caseAgainst] = card.withClass("side").map((side) => side.text);
    assert.match(caseFor, /The review says so/);
    assert.match(caseFor, /Finding of s1\./);
    assert.match(caseAgainst, /Finding of s2\..*Quote and source 1.*Finding of s3\./);
    assert.doesNotMatch(caseAgainst, /Finding of s1\./);
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
    assert.match(panel.text, /A big trial.*No effect was seen\..*a second check agreed.*Another/);
    second.listeners.click(click());
    assert.equal(mark.attributes["aria-expanded"], "false");
    assert.equal(panel.hidden, true);
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

  async "a report stored without a glossary still renders and offers one"() {
    const report = syntheticReport();
    delete report.glossary;
    const app = loadApp(() => json([]));
    const card = renderedClaim(app, report);
    assert.equal(card.withClass("term").length, 0);
    assert.equal(card.withClass("cited").length > 0, true);
    assert.match(app.readingTip(report).text, /Explain the technical terms/);
    assert.doesNotMatch(app.readingTip(syntheticReport()).text, /Explain the technical terms/);
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
