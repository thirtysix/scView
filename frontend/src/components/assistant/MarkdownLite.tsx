import { type ReactNode } from "react";
import type { Claim } from "@/api/assistant";
import { CitationChip } from "@/components/assistant/CitationChip";
import { ClaimMark } from "@/components/assistant/ClaimMark";
import { findClaimMarks } from "@/lib/claimMarks";

/** Minimal, dependency-free markdown renderer for co-pilot answers.
 *  Supports: paragraphs, bullet/numbered lists, ATX headers, ``` code blocks, GFM
 *  pipe tables (with column alignment), inline **bold**, *italic*, `code` (also
 *  inside bold or italic), and [kind:...] citation chips. Renders via React text
 *  nodes (no raw HTML), so it's XSS-safe. With `claims` (the claims ledger), each
 *  checked sentence is marked in place by its verdict (ClaimMark). */

// bold | code | italic | citation tag. The citation alternative matches any
// `[prefix: ...]` token (prefix = letters/spaces/-/_) that isn't a markdown link,
// so the model's reformatted citation tags ("[cell-type annotation: ...]") still
// render as chips instead of raw brackets.
const INLINE =
  /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*|\[[A-Za-z][\w '-]*:[^\]\n]*\](?!\())/g;

function renderInline(
  text: string,
  keyPrefix: string,
  onCitation?: (tag: string) => void
): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  // A fresh regex per call: bold and italic render their inner text through this
  // function again, which would reset a shared regex's position.
  const re = new RegExp(INLINE.source, "g");
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    const key = `${keyPrefix}-${k++}`;
    if (tok.startsWith("**")) {
      out.push(<strong key={key}>{renderInline(tok.slice(2, -2), key, onCitation)}</strong>);
    } else if (tok.startsWith("`")) {
      out.push(
        <code key={key} className="rounded bg-muted px-1 py-0.5 text-[0.85em]">
          {tok.slice(1, -1)}
        </code>
      );
    } else if (tok.startsWith("[")) {
      out.push(<CitationChip key={key} tag={tok.slice(1, -1)} onClick={onCitation} />);
    } else {
      out.push(<em key={key}>{renderInline(tok.slice(1, -1), key, onCitation)}</em>);
    }
    last = re.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

type Align = "left" | "center" | "right" | undefined;

// The row under a table's header: | --- | :--: | --: |
const TABLE_RULE = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/;

/** The cells of one pipe-table row (outer pipes optional; \| is a literal pipe). */
function tableCells(line: string): string[] {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
  return s.split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|"));
}

function alignOf(rule: string): Align {
  const r = rule.trim();
  if (r.startsWith(":") && r.endsWith(":")) return "center";
  if (r.endsWith(":")) return "right";
  if (r.startsWith(":")) return "left";
  return undefined;
}

/** A header line followed by a rule line starts a table. */
function startsTable(line: string, next: string | undefined): boolean {
  return line.includes("|") && next !== undefined && TABLE_RULE.test(next) && next.includes("-");
}

export function MarkdownLite({
  text,
  onCitation,
  claims,
}: {
  text: string;
  onCitation?: (tag: string) => void;
  claims?: Claim[];
}) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  const marks = claims?.filter((c) => c.span) ?? [];
  const marked = new Set<number>();

  // One block's inline content, with each checked sentence in it marked by its verdict.
  const inline = (s: string, k: string): ReactNode[] => {
    const spans = marks.length ? findClaimMarks(s, marks, marked) : [];
    if (!spans.length) return renderInline(s, k, onCitation);
    const out: ReactNode[] = [];
    let at = 0;
    spans.forEach((m, j) => {
      if (m.start > at) out.push(...renderInline(s.slice(at, m.start), `${k}-t${j}`, onCitation));
      out.push(
        <ClaimMark key={`${k}-c${j}`} claim={m.claim}>
          {renderInline(s.slice(m.start, m.end), `${k}-c${j}`, onCitation)}
        </ClaimMark>,
      );
      at = m.end;
    });
    if (at < s.length) out.push(...renderInline(s.slice(at), `${k}-end`, onCitation));
    return out;
  };
  let para: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  let inCode = false;
  let code: string[] = [];
  let key = 0;

  const flushPara = () => {
    if (para.length) {
      blocks.push(
        <p key={`p-${key++}`} className="my-1 first:mt-0 last:mb-0">
          {inline(para.join(" "), `p${key}`)}
        </p>
      );
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      const Tag = list.ordered ? "ol" : "ul";
      const cls = list.ordered ? "list-decimal" : "list-disc";
      blocks.push(
        <Tag key={`l-${key++}`} className={`${cls} my-1 space-y-0.5 pl-5`}>
          {list.items.map((it, i) => (
            <li key={i}>{inline(it, `li${key}-${i}`)}</li>
          ))}
        </Tag>
      );
      list = null;
    }
  };

  const flushCode = () => {
    blocks.push(
      <pre key={`c-${key++}`} className="my-1 overflow-x-auto rounded bg-muted p-2 text-xs leading-snug">
        <code>{code.join("\n")}</code>
      </pre>
    );
    code = [];
  };

  const flushTable = (header: string[], aligns: Align[], rows: string[][]) => {
    const t = key++;
    const n = header.length;
    blocks.push(
      <div key={`t-${t}`} className="my-1 overflow-x-auto">
        <table className="border-collapse text-xs">
          <thead>
            <tr>
              {header.map((h, i) => (
                <th
                  key={i}
                  scope="col"
                  style={{ textAlign: aligns[i] ?? "left" }}
                  className="border-b border-border px-2 py-1 font-semibold"
                >
                  {renderInline(h, `t${t}-h${i}`, onCitation)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={ri} className="border-b border-border/50 last:border-0">
                {Array.from({ length: n }, (_, i) => (
                  <td key={i} style={{ textAlign: aligns[i] }} className="px-2 py-1 align-top">
                    {inline(r[i] ?? "", `t${t}-${ri}-${i}`)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  for (let li = 0; li < lines.length; li++) {
    const line = (lines[li] ?? "").trimEnd();
    if (/^\s*```/.test(line)) {
      if (inCode) {
        flushCode();
      } else {
        flushPara();
        flushList();
      }
      inCode = !inCode;
      continue;
    }
    if (inCode) {
      code.push(line);
      continue;
    }
    const bullet = /^\s*[-*]\s+(.*)$/.exec(line);
    const numbered = /^\s*\d+\.\s+(.*)$/.exec(line);
    const header = /^(#{1,6})\s+(.*)$/.exec(line);

    if (!line.trim()) {
      flushPara();
      flushList();
      continue;
    }
    if (startsTable(line, lines[li + 1])) {
      flushPara();
      flushList();
      const header = tableCells(line);
      const aligns = tableCells(lines[li + 1] ?? "").map(alignOf);
      const rows: string[][] = [];
      li += 2;
      while (li < lines.length && (lines[li] ?? "").includes("|") && (lines[li] ?? "").trim()) {
        rows.push(tableCells(lines[li] ?? ""));
        li++;
      }
      li--; // the for loop's li++ moves to the line after the table
      flushTable(header, aligns, rows);
      continue;
    }
    if (header) {
      flushPara();
      flushList();
      blocks.push(
        <p key={`h-${key++}`} className="mb-0.5 mt-2 font-semibold first:mt-0">
          {renderInline(header[2] ?? "", `h${key}`, onCitation)}
        </p>
      );
      continue;
    }
    if (bullet) {
      flushPara();
      if (!list || list.ordered) {
        flushList();
        list = { ordered: false, items: [] };
      }
      list.items.push(bullet[1] ?? "");
      continue;
    }
    if (numbered) {
      flushPara();
      if (!list || !list.ordered) {
        flushList();
        list = { ordered: true, items: [] };
      }
      list.items.push(numbered[1] ?? "");
      continue;
    }
    flushList();
    para.push(line);
  }
  if (inCode) flushCode(); // unclosed fence, e.g. while the answer is still streaming
  flushPara();
  flushList();

  return <div className="text-sm leading-relaxed">{blocks}</div>;
}
