// Owns everything that needs the page: opening the panel, reading posts, screenshots.

const SELECTION_MENU = "fact-check-selection";

chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: SELECTION_MENU,
    title: "Fact-check this",
    contexts: ["selection"],
  });
});

// sidePanel.open only works inside the user gesture, so it runs before any await.
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== SELECTION_MENU) return;
  chrome.sidePanel.open({ tabId: tab.id });
  const text = info.selectionText.trim();
  queueCheck({ payload: { text }, preview: { text } });
});

chrome.commands.onCommand.addListener(async (command, tab) => {
  if (command !== "check-post") return;
  chrome.sidePanel.open({ tabId: tab.id });
  queueCheck(await readPost(tab));
});

chrome.runtime.onMessage.addListener((message, _sender, reply) => {
  if (message.type === "take-queued-check") {
    takeQueuedCheck().then(reply);
    return true;
  }
  if (message.type !== "read-post") return false;
  chrome.tabs
    .query({ active: true, windowId: message.windowId })
    .then(([tab]) => readPost(tab))
    .then(reply);
  return true;
});

function queueCheck(check) {
  chrome.storage.session.set({ queuedCheck: check });
}

// Panels take checks through here one at a time, so each is submitted exactly once.
let lastTake = Promise.resolve();

function takeQueuedCheck() {
  const taken = lastTake.then(async () => {
    const { queuedCheck } = await chrome.storage.session.get("queuedCheck");
    if (queuedCheck) await chrome.storage.session.remove("queuedCheck");
    return queuedCheck ?? null;
  });
  lastTake = taken.catch(() => {});
  return taken;
}

async function readPost(tab) {
  if (!/^https?:/.test(tab?.url ?? "https:")) {
    return { error: "Fakt can't read this page. Paste a link or some text instead." };
  }
  let post;
  try {
    const [injection] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["extract_post.js"],
    });
    post = injection.result;
  } catch {
    return {
      error:
        "Fakt needs your go-ahead on this page. Press Alt+Shift+F, or click the Fakt toolbar button, while viewing it.",
    };
  }
  const image_base64 = post.url ? undefined : await screenshot(tab.windowId);
  if (!post.url && !post.text && !image_base64) {
    return { error: "Nothing to check here. Scroll to a post and try again." };
  }
  return { payload: buildPayload(post, image_base64), preview: post };
}

function buildPayload(post, image_base64) {
  const byline = post.author && !post.url ? `Posted by ${post.author}\n\n` : "";
  return {
    url: post.url ?? undefined,
    text: post.text ? byline + post.text : undefined,
    image_base64,
  };
}

async function screenshot(windowId) {
  try {
    return await chrome.tabs.captureVisibleTab(windowId, { format: "jpeg", quality: 85 });
  } catch (error) {
    console.warn("Screenshot failed", error);
    return undefined;
  }
}
