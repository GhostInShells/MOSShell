// @name markdown
// @ext .md .markdown .mdx
// @description 文档：标题、段落、列表、表格、代码块；每块记着它来自哪几行
//
// 自带的渲染器，故意很小：页面不能依赖 CDN（离线时会退化成纯文本），而且有些地方
// 我们比 markdown 本身知道得更多——文件头、diff、真的是一张表的表——那就要自己说出来。
//
// 安全靠构造：整篇先转义，再把结构包在转义后的文本外面。于是文件永远没法注入我们
// 没打算要的标记，也就不需要消毒器。
//
// 每块都带 data-lines="起-止"：渲染面是给人读的，但块仍然知道自己的行区间，
// 于是"高亮这一段"和"他选了这一段"都能换算回行号——一块只读的玻璃，透过去能指、
// 能对上号。行内不到字符级，块级为止，这是诚实的边界。

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ESC[c]);

// 开头那段 YAML 不是 markdown。交给渲染器会把三行 key: value 变成一行粗体——
// 段落下面三条横线在 markdown 里是 setext 标题。所以把它当文件头，不当正文。
function leadingFrontmatter(text) {
  const m = /^---\r?\n[\s\S]*?\r?\n---[ \t]*(\r?\n|$)/.exec(text);
  return m ? m[0] : "";
}

function inline(text) {
  let t = esc(text);
  t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  t = t.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, label, href) => {
    // 相对目标是故意放行的：一份文件的链接指向它的邻居，而这页就是写那些文件的人看的。
    const ok = /^(https?:|\.{0,2}\/|#|\/)/.test(href) && !/["'<>]/.test(href);
    return ok
      ? "<a href='" + href + "' target='_blank' rel='noreferrer'>" + label + "</a>"
      : label;
  });
  return t;
}

const isTableStart = (lines, i) => {
  if (!lines[i] || !lines[i].includes("|")) return false;
  const sep = lines[i + 1];
  return !!sep && sep.includes("-") && /^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$/.test(sep);
};
const splitRow = (line) =>
  line.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map((c) => c.trim());

function renderBlocks(src) {
  const lines = String(src || "").replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  const para = [];
  let paraFrom = 0;            // 段落从哪一行开始（1-based）
  const attr = (a, b) => ' data-lines="' + a + (b > a ? "-" + b : "") + '"';
  const flush = (endLine) => {
    if (!para.length) return;
    out.push("<p" + attr(paraFrom, endLine) + ">" + inline(para.join(" ")) + "</p>");
    para.length = 0;
  };
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const start = i + 1;
    if (/^\s*```/.test(line)) {
      flush(start - 1);
      const buf = [];
      i += 1;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) { buf.push(lines[i]); i += 1; }
      i += 1;
      out.push("<pre class='code'" + attr(start, i) + "><code>" + esc(buf.join("\n")) + "</code></pre>");
      continue;
    }
    if (/^\s*$/.test(line)) { flush(start - 1); i += 1; continue; }
    const head = /^(#{1,6})\s+(.*)$/.exec(line);
    if (head) {
      flush(start - 1);
      const n = head[1].length;
      out.push("<h" + n + attr(start, start) + ">" + inline(head[2]) + "</h" + n + ">");
      i += 1;
      continue;
    }
    if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      flush(start - 1);
      out.push("<hr class='rule'" + attr(start, start) + ">");
      i += 1;
      continue;
    }
    if (isTableStart(lines, i)) {
      flush(start - 1);
      const cols = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|")) { rows.push(splitRow(lines[i])); i += 1; }
      // 表格天生可能比正文栏宽：列多、格子长。标准解法是给它一层可横向滚的容器，
      // 而不是把单元格压扁 —— 压扁的代价是 "P0" 折成两行、每列都读不顺。
      out.push(
        "<div class=\"table-wrap\"><table" + attr(start, i) + "><thead><tr>" +
        cols.map((c) => "<th>" + inline(c) + "</th>").join("") +
        "</tr></thead><tbody>" +
        rows.map((r) => "<tr>" + r.map((c) => "<td>" + inline(c) + "</td>").join("") + "</tr>").join("") +
        "</tbody></table></div>",
      );
      continue;
    }
    if (/^\s*([-*+]|\d+\.)\s+/.test(line)) {
      flush(start - 1);
      const ordered = /^\s*\d+\./.test(line);
      const items = [];
      const re = ordered ? /^\s*\d+\.\s+/ : /^\s*[-*+]\s+/;
      while (i < lines.length && re.test(lines[i])) { items.push(inline(lines[i].replace(re, ""))); i += 1; }
      const tag = ordered ? "ol" : "ul";
      out.push("<" + tag + attr(start, i) + ">" + items.map((t) => "<li>" + t + "</li>").join("") + "</" + tag + ">");
      continue;
    }
    if (/^\s*>\s?/.test(line)) {
      flush(start - 1);
      const buf = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) { buf.push(lines[i].replace(/^\s*>\s?/, "")); i += 1; }
      out.push("<blockquote" + attr(start, i) + ">" + inline(buf.join(" ")) + "</blockquote>");
      continue;
    }
    if (!para.length) paraFrom = start;
    para.push(line.trim());
    i += 1;
  }
  flush(lines.length);
  return out.join("");
}

export default {
  name: "markdown",
  ext: [".md", ".markdown", ".mdx"],
  description: "markdown 文档（块级可寻址）",

  render(text) {
    const full = String(text ?? "");
    const front = leadingFrontmatter(full);
    let html = "";
    if (front) {
      const n = front.replace(/\n$/, "").split("\n").length;
      html += "<pre class='frontmatter' data-lines=\"1-" + n + "\">" + esc(front) + "</pre>";
    }
    return html + renderBlocks(front ? full.slice(front.length) : full);
  },

  // 渲染面上高亮"这几行"：找出与之有交集的块，整块点亮。块级为止，不假装字符级。
  decorate(root, { cursor, hit } = {}) {
    const spans = () => [...root.querySelectorAll("[data-lines]")];
    const range = (el) => {
      const parts = String(el.dataset.lines || "").split("-");
      const a = parseInt(parts[0], 10);
      const b = parseInt(parts[1] || parts[0], 10);
      return [a, b];
    };
    if (hit) {
      for (const el of spans()) {
        const [a, b] = range(el);
        if (a <= hit[1] && b >= hit[0]) el.classList.add("hit");
      }
    }
    if (cursor) {
      let done = false;
      for (const el of spans()) {
        const [a, b] = range(el);
        if (a <= cursor.end && b >= cursor.start) {
          el.classList.add("cursor");
          if (cursor.scroll && !done) { el.scrollIntoView({ block: "center" }); done = true; }
        }
      }
      return done;
    }
    return false;
  },
};
