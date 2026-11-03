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
    this.value = "";
    this.textContent = "";
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
    window: { addEventListener: (_name, listener) => hashListeners.push(listener) },
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
