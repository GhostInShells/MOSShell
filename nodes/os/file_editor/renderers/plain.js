// @name plain
// @ext .txt .log .csv .ini .cfg .env
// @description 纯文本：不带行号的读面，每行仍记着自己的行号
//
// 和 raw 差在"读"与"改"：raw 为了被寻址而存在，行号必须在；plain 为了一封邮件、
// 一段日志读起来不被打断而存在，所以行号不画，但 data-lines 仍然挂在每一行上——
// 不画不等于不知道，人和我指的仍然是同一行。

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ESC[c]);

export default {
  name: "plain",
  ext: [".txt", ".log", ".csv", ".ini", ".cfg", ".env"],
  description: "纯文本（不带行号）",

  render(text) {
    const lines = String(text ?? "").replace(/\r\n?/g, "\n").split("\n");
    if (lines.length && lines[lines.length - 1] === "") lines.pop();
    return lines
      .map((line, i) =>
        '<div class="doc-line plain" data-n="' + (i + 1) + '" data-lines="' + (i + 1) + '">' +
        '<span class="lt">' + (esc(line) || "&nbsp;") + "</span></div>")
      .join("");
  },
};
