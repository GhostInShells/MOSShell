// @name code
// @ext .py .js .ts .tsx .jsx .json .yml .yaml .toml .sh .css .html .go .rs .java .c .h .cpp
// @description 代码：行号 + 语法着色（有高亮库时用它，没有时用自带的轻词法）
//
// 两档着色，一条判据：「读得动」。
//
// - 页面里引了 highlight.js（见 index.html，外链，不进仓库），就按后缀名挑语言交给它，
//   着色准、覆盖广；库没加载到（没网、被拦）或这门语言它不认识，就退回下面这套轻词法：
//   字符串 / 注释 / 数字 / 关键字，跨语言取交集，宁可少染，不要错染。
// - 两档都必须吐「每行一个 doc-line，行号在外面」的形状：着色只改颜色，绝不改字符本身，
//   否则选区、复制、行号、行区间编辑全会错位。
//
// 换行是最麻烦的一处：库的 <span> 会横跨多行（多行字符串、块注释），直接按 \n 切会把
// 标签切碎。所以切的时候维护一个开标签栈，行末补齐闭合、行首重新打开 —— 渲染出来的行
// 依然是每一行自洽，页面那套按行寻址的东西才成立。

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ESC[c]);

// 一小把通用关键字：跨语言取交集，宁可少染，不要错染。
const KEYWORDS = new Set((
  "if else elif for while return def class import from as with try except finally " +
  "raise pass break continue lambda and or not in is None True False null true false " +
  "function const let var new this async await export default interface type extends " +
  "public private static void int float string bool char switch case do struct impl fn " +
  "match use mut pub mod go defer chan package func range nil"
).split(/\s+/));

// 一行里按「字符串 → 注释 → 数字 → 关键字」的顺序切；先切出来的不被后面的规则改写，
// 否则字符串里的 # 会被当成注释、注释里的 for 会被染成关键字。
function paint(line) {
  const parts = [];
  let i = 0;
  const push = (cls, text) => {
    if (!text) return;
    parts.push(cls ? '<span class="tk-' + cls + '">' + esc(text) + "</span>" : esc(text));
  };
  while (i < line.length) {
    const rest = line.slice(i);
    const str = /^("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)/.exec(rest);
    if (str) { push("str", str[1]); i += str[1].length; continue; }
    const cmt = /^(\/\/.*|#.*|--.*)$/.exec(rest);
    if (cmt) { push("cmt", cmt[1]); break; }
    const num = /^\b\d[\d_.]*\b/.exec(rest);
    if (num) { push("num", num[0]); i += num[0].length; continue; }
    const word = /^[A-Za-z_$][\w$]*/.exec(rest);
    if (word) { push(KEYWORDS.has(word[0]) ? "kw" : "", word[0]); i += word[0].length; continue; }
    push("", line[i]);
    i += 1;
  }
  return parts.join("");
}

// -- the library path ------------------------------------------------------

// Extension → the language id highlight.js knows. Only languages the library really
// carries: a wrong id throws, and a throw here would cost the whole document its colour.
const LANG_BY_EXT = {
  ".py": "python",
  ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
  ".ts": "typescript", ".tsx": "typescript",
  ".json": "json", ".yml": "yaml", ".yaml": "yaml", ".toml": "ini",
  ".sh": "bash", ".zsh": "bash", ".css": "css", ".html": "xml",
  ".go": "go", ".rs": "rust", ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp",
};

function langFor(path) {
  const p = String(path || "").toLowerCase();
  const dot = p.lastIndexOf(".");
  if (dot < 0) return "";
  return LANG_BY_EXT[p.slice(dot)] || "";
}

// Split highlighted HTML into per-line HTML, keeping every <span> balanced.
// A tag opened on one line is closed at that line's end and reopened on the next.
function splitHighlighted(html) {
  const out = [];
  const open = [];
  let buf = "";
  let i = 0;
  while (i < html.length) {
    if (html[i] === "<") {
      const end = html.indexOf(">", i);
      if (end === -1) { buf += html.slice(i); break; }
      const tag = html.slice(i, end + 1);
      if (tag.startsWith("</")) open.pop();
      else if (!tag.endsWith("/>")) open.push(tag);
      buf += tag;
      i = end + 1;
      continue;
    }
    const nl = html.indexOf("\n", i);
    if (nl === -1) { buf += html.slice(i); break; }
    buf += html.slice(i, nl);
    out.push(buf + "</span>".repeat(open.length));
    buf = open.join("");
    i = nl + 1;
  }
  out.push(buf);
  return out;
}

// 库那一档暂时关掉：整篇喂进去再按行切开时，某些跨行 token 会让 span 失去平衡，
// 结果是整整一大片被染成关键字的颜色。在切行这件事被证明是稳的之前，先走自带那套。
const USE_LIBRARY = false;

function paintViaLibrary(text, language) {
  if (!USE_LIBRARY) return null;
  const hljs = typeof window !== "undefined" ? window.hljs : null;
  if (!hljs || !language || typeof hljs.highlight !== "function") return null;
  try {
    return splitHighlighted(hljs.highlight(text, { language, ignoreIllegals: true }).value);
  } catch (e) {
    // Unknown language or a library that changed shape: the hand-rolled painter still
    // works, and "no colour" must not be the answer to "the library could not help".
    return null;
  }
}

export default {
  name: "code",
  ext: [".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yml", ".yaml", ".toml", ".sh",
        ".css", ".html", ".go", ".rs", ".java", ".c", ".h", ".cpp"],
  description: "代码（行号 + 语法着色）",

  render(text, ctx) {
    const source = String(text ?? "").replace(/\r\n?/g, "\n");
    const lines = source.split("\n");
    if (lines.length && lines[lines.length - 1] === "") lines.pop();
    const highlighted = paintViaLibrary(source, langFor(ctx && ctx.path));
    const out = [];
    for (let i = 0; i < lines.length; i += 1) {
      const n = i + 1;
      const body = highlighted && highlighted[i] !== undefined ? highlighted[i] : paint(lines[i]);
      out.push(
        '<div class="doc-line code" data-n="' + n + '" data-lines="' + n + '">' +
        '<span class="ln">' + n + "</span>" +
        '<span class="lt">' + body + "</span>" +
        "</div>",
      );
    }
    return out.join("");
  },

  decorate(root, { cursor, hit } = {}) {
    let scrolled = false;
    for (const row of root.querySelectorAll(".doc-line")) {
      const n = parseInt(row.dataset.n, 10);
      if (hit && n >= hit[0] && n <= hit[1]) row.classList.add("hit");
      if (cursor && n >= cursor.start && n <= cursor.end) {
        row.classList.add("cursor");
        if (cursor.scroll && !scrolled) { row.scrollIntoView({ block: "center" }); scrolled = true; }
      }
    }
    return scrolled;
  },
};