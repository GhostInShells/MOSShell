// @name raw
// @ext
// @description 带行号的原样文本 —— 我那一面：地址只在这里精确
//
// 这是库里的第一个渲染器，也是"兜底"这个职位的持有者：别的渲染器坏了就退到它。
// 但它不是"没渲染"的同义词 —— 行号是它存在的理由，因为我和人之间唯一的共同坐标
// 就是行号，而 read / edit / find / rewind 全都吃它。
//
// 契约（见 DESIGN.md 第六节）：render(text) 吐 HTML；可选 decorate(root, {cursor, hit})
// 负责把"我们正在看的那几行"画出来。行区间的桥是 data-lines 属性，页面靠它把
// 选区、光标换算回行号。

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ESC[c]);

export default {
  name: "raw",
  ext: [],
  description: "带行号的原样文本",

  render(text) {
    const lines = String(text ?? "").replace(/\r\n?/g, "\n").split("\n");
    if (lines.length && lines[lines.length - 1] === "") lines.pop();
    const out = [];
    for (let i = 0; i < lines.length; i += 1) {
      const n = i + 1;
      out.push(
        '<div class="doc-line" data-n="' + n + '" data-lines="' + n + '">' +
        '<span class="ln">' + n + "</span>" +
        '<span class="lt">' + esc(lines[i]) + "</span>" +
        "</div>",
      );
    }
    return out.join("");
  },

  // 读的那一行 / 被选中的那一段，只有 raw 画得准 —— 因为它的每一行就是地址本身。
  decorate(root, { cursor, hit } = {}) {
    let scrolled = false;
    for (const row of root.querySelectorAll(".doc-line")) {
      const n = parseInt(row.dataset.n, 10);
      if (hit && n >= hit[0] && n <= hit[1]) row.classList.add("hit");
      if (cursor && n >= cursor.start && n <= cursor.end) {
        row.classList.add("cursor");
        if (cursor.scroll && !scrolled) {
          row.scrollIntoView({ block: "center" });
          scrolled = true;
        }
      }
    }
    return scrolled;
  },
};
