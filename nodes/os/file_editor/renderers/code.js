// @name code
// @ext .py .js .ts .tsx .jsx .json .yml .yaml .toml .sh .css .html .go .rs .java .c .h .cpp
// @description 代码：行号 + 轻词法着色（字符串 / 注释 / 数字 / 关键字）
//
// 刻意不做真语法高亮 —— 那要按语言一套文法，是一个独立的活。这里只做一件事：
// 让代码在屏幕上是代码。判据是"读得动"，不是"和编辑器一模一样"。
//
// 着色走行号之外的第二个 span，绝不改变字符本身：选区、复制、行号都对得上，
// 这才是渲染面该有的本分。

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

// 一行里按"字符串 → 注释 → 数字 → 关键字"的顺序切；先切出来的不被后面的规则改写，
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

export default {
  name: "code",
  ext: [".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yml", ".yaml", ".toml", ".sh",
        ".css", ".html", ".go", ".rs", ".java", ".c", ".h", ".cpp"],
  description: "代码（行号 + 轻词法）",

  render(text) {
    const lines = String(text ?? "").replace(/\r\n?/g, "\n").split("\n");
    if (lines.length && lines[lines.length - 1] === "") lines.pop();
    const out = [];
    for (let i = 0; i < lines.length; i += 1) {
      const n = i + 1;
      out.push(
        '<div class="doc-line code" data-n="' + n + '" data-lines="' + n + '">' +
        '<span class="ln">' + n + "</span>" +
        '<span class="lt">' + paint(lines[i]) + "</span>" +
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
