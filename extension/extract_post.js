// Injected on demand. Finds the post the reader is looking at and returns it as the
// script's completion value: {platform, url, author, text}.
(() => {
  const MAX_TEXT = 6000;
  const SITES = [
    {
      platform: "x",
      host: /(^|\.)(x|twitter)\.com$/,
      posts: 'article[data-testid="tweet"]',
      links: 'a[href*="/status/"]',
      permalink: /^\/[^/]+\/status\/\d+/,
      author: '[data-testid="User-Name"]',
      text: '[data-testid="tweetText"]',
    },
    {
      platform: "instagram",
      host: /(^|\.)instagram\.com$/,
      posts: 'article, [role="presentation"]:has(video)',
      links: 'a[href*="/p/"], a[href*="/reel/"]',
      permalink: /^(\/[^/]+)?\/(p|reel|reels)\/[\w-]+/,
      author: 'header a, a[role="link"]',
      text: "h1, h2 ~ div span[dir]",
    },
    {
      platform: "facebook",
      host: /(^|\.)facebook\.com$/,
      posts: '[role="article"], [aria-posinset]',
      links:
        'a[href*="/posts/"], a[href*="/videos/"], a[href*="/reel/"], a[href*="/permalink"], a[href*="story_fbid"], a[href*="/photo"]',
      permalink:
        /^\/(?:[^/]+\/(?:posts|videos)\/[\w.-]+|reel\/\d+|watch\/?\?v=\d+|permalink\.php\?story_fbid=[^&]+&id=\d+|photo\/?\?fbid=\d+|groups\/[^/]+\/(?:posts|permalink)\/\d+)/,
      author: "h2 a, h3 a, h4 a, strong a",
      text: '[data-ad-preview="message"], [data-ad-comet-preview="message"]',
    },
    {
      platform: "tiktok",
      host: /(^|\.)tiktok\.com$/,
      posts: '[data-e2e="recommend-list-item-container"], article',
      links: 'a[href*="/video/"]',
      permalink: /^\/@[^/]+\/video\/\d+/,
      author: '[data-e2e="video-author-uniqueid"], [data-e2e="browse-username"]',
      text: '[data-e2e="video-desc"], [data-e2e="browse-video-desc"]',
    },
    {
      platform: "youtube",
      host: /(^|\.)youtube\.com$/,
      posts:
        "ytd-reel-video-renderer[is-active], ytd-rich-item-renderer, ytd-video-renderer, ytd-compact-video-renderer",
      links: 'a[href*="/watch?v="], a[href*="/shorts/"]',
      permalink: /^\/(?:shorts\/[\w-]+|watch\?v=[\w-]+)/,
      author: "ytd-channel-name a, #channel-name a",
      text: "#video-title, h1, h2",
    },
    {
      platform: "reddit",
      host: /(^|\.)reddit\.com$/,
      posts: "shreddit-post, article",
      links: 'a[href*="/comments/"]',
      permalink: /^\/r\/[^/]+\/comments\/\w+(\/[^/?#]+)?/,
      author: '[slot="authorName"], a[href*="/user/"]',
      text: '[slot="title"], [slot="text-body"], h1, h3',
    },
  ];

  return readPage();

  function readPage() {
    const selected = getSelection()?.toString().trim() || "";
    const site = SITES.find(({ host }) => host.test(location.hostname));
    if (!site) return { platform: "web", url: location.href, author: null, text: selected };

    const post = centredPost(site.posts);
    return {
      platform: site.platform,
      url: (post && permalinkIn(post, site)) || pageIfItIsThePost(site),
      author: post ? authorOf(post, site) : null,
      text: clip(selected || (post ? textOf(post, site) : visibleText())),
    };
  }

  function centredPost(selector) {
    const middle = innerHeight / 2;
    let best = null;
    let bestDistance = Infinity;
    for (const element of document.querySelectorAll(selector)) {
      if (element.parentElement?.closest(selector)) continue;
      const box = element.getBoundingClientRect();
      if (box.height === 0 || box.bottom <= 0 || box.top >= innerHeight) continue;
      const nearest = Math.min(Math.max(middle, box.top), box.bottom);
      const distance = Math.abs(nearest - middle);
      if (distance < bestDistance) [best, bestDistance] = [element, distance];
    }
    return best;
  }

  function permalinkIn(post, site) {
    const anchors = [...post.querySelectorAll(site.links)];
    anchors.sort((a, b) => hasTime(b) - hasTime(a));
    for (const anchor of anchors) {
      const url = canonical(anchor.href, site);
      if (url) return url;
    }
    return null;
  }

  function hasTime(anchor) {
    return anchor.querySelector("time") ? 1 : 0;
  }

  function pageIfItIsThePost(site) {
    return canonical(location.href, site);
  }

  function canonical(href, site) {
    const url = new URL(href, location.href);
    const match = (url.pathname + url.search).match(site.permalink);
    return match ? url.origin + match[0] : null;
  }

  function authorOf(post, site) {
    const attribute = post.getAttribute("author");
    if (attribute) return attribute;
    const text = post.querySelector(site.author)?.innerText ?? "";
    return text.split("\n").map((line) => line.trim()).find(Boolean) ?? null;
  }

  function textOf(post, site) {
    const title = post.getAttribute("post-title");
    const parts = [...post.querySelectorAll(site.text)].map((part) => part.innerText.trim());
    const found = [...new Set([title, ...parts].filter(Boolean))].join("\n\n");
    return found || post.innerText.trim();
  }

  function visibleText() {
    const lines = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let size = 0;
    while (walker.nextNode() && size < MAX_TEXT) {
      const line = walker.currentNode.nodeValue.trim();
      const parent = walker.currentNode.parentElement;
      if (!line || parent.closest("script, style, noscript, nav, [aria-hidden='true']")) continue;
      const box = parent.getBoundingClientRect();
      if (box.width === 0 || box.bottom <= 0 || box.top >= innerHeight) continue;
      lines.push(line);
      size += line.length;
    }
    return lines.join("\n");
  }

  function clip(text) {
    return text.slice(0, MAX_TEXT);
  }
})();
