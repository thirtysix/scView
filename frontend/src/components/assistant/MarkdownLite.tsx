import { type ReactNode } from "react";
import { CitationChip } from "@/components/assistant/CitationChip";

/** Minimal, dependency-free markdown renderer for co-pilot answers.
 *  Supports: paragraphs, bullet/numbered lists, ATX headers, inline
 *  **bold**, *italic*, `code`, and [kind:...] citation chips. Renders via React
 *  text nodes (no raw HTML), so it's XSS-safe. */

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
  INLINE.lastIndex = 0;
  while ((m = INLINE.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) {
      out.push(<strong key={`${keyPrefix}-${k++}`}>{tok.slice(2, -2)}</strong>);
    } else if (tok.startsWith("`")) {
      out.push(
        <code key={`${keyPrefix}-${k++}`} className="rounded bg-muted px-1 py-0.5 text-[0.85em]">
          {tok.slice(1, -1)}
        </code>
      );
    } else if (tok.startsWith("[")) {
      out.push(<CitationChip key={`${keyPrefix}-${k++}`} tag={tok.slice(1, -1)} onClick={onCitation} />);
    } else {
      out.push(<em key={`${keyPrefix}-${k++}`}>{tok.slice(1, -1)}</em>);
    }
    last = INLINE.lastIndex;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function MarkdownLite({
  text,
  onCitation,
}: {
  text: string;
  onCitation?: (tag: string) => void;
}) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let para: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  let key = 0;

  const flushPara = () => {
    if (para.length) {
      blocks.push(
        <p key={`p-${key++}`} className="my-1 first:mt-0 last:mb-0">
          {renderInline(para.join(" "), `p${key}`, onCitation)}
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
            <li key={i}>{renderInline(it, `li${key}-${i}`, onCitation)}</li>
          ))}
        </Tag>
      );
      list = null;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const bullet = /^\s*[-*]\s+(.*)$/.exec(line);
    const numbered = /^\s*\d+\.\s+(.*)$/.exec(line);
    const header = /^(#{1,6})\s+(.*)$/.exec(line);

    if (!line.trim()) {
      flushPara();
      flushList();
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
  flushPara();
  flushList();

  return <div className="text-sm leading-relaxed">{blocks}</div>;
}
