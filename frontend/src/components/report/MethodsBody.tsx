import { formatDate } from "@/lib/formatting";

interface Methods {
  summary: string[];
  steps: string[];
  sections: string[];
}

/** The methods section: the recorded analysis steps and how each report section was made. */
export function MethodsBody({ d, generatedAt, version }: { d: Methods; generatedAt: string; version: string }) {
  return (
    <div className="space-y-1 text-[11px] text-slate-700">
      {(d.summary ?? []).map((line) => <p key={line}>{line}</p>)}
      {d.steps?.length > 0 && (
        <ol className="list-decimal space-y-0.5 pl-5" aria-label="Recorded analysis steps">
          {d.steps.map((s, i) => <li key={`${i}-${s}`} className="break-words">{s}</li>)}
        </ol>
      )}
      {d.sections?.length > 0 && (
        <details>
          <summary className="cursor-pointer text-[11px] font-medium text-slate-600">How each section was made</summary>
          <ul className="mt-1 list-disc space-y-0.5 pl-5 text-slate-600">
            {d.sections.map((s) => <li key={s}>{s}</li>)}
          </ul>
        </details>
      )}
      <p className="text-[10px] text-slate-400">
        scView {version}, report made {formatDate(generatedAt)}.
      </p>
    </div>
  );
}
