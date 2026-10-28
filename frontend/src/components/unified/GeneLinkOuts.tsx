import { ExternalLink } from "lucide-react";
import { gtexGeneUrl, gwasCatalogGeneUrl, openTargetsUrl } from "@/api/gwas";

/**
 * A gene's pages in the GWAS Catalog, Open Targets (by Ensembl id when known) and
 * GTEx. Plain links: scView itself sends nothing; the browser opens the page.
 */
export function GeneLinkOuts({ gene, ensembl, className = "" }: { gene: string; ensembl?: string | null; className?: string }) {
  const links = [
    { label: "GWAS Catalog", href: gwasCatalogGeneUrl(gene), title: `${gene} in the GWAS Catalog` },
    {
      label: "Open Targets",
      href: openTargetsUrl(gene, ensembl),
      title: ensembl ? `${gene} (${ensembl}) in the Open Targets Platform` : `Search Open Targets for ${gene}`,
    },
    { label: "GTEx", href: gtexGeneUrl(gene), title: `${gene} expression across tissues in GTEx` },
  ];
  return (
    <p className={`flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-slate-500 ${className}`}>
      <span>{gene} elsewhere:</span>
      {links.map((l) => (
        <a
          key={l.label}
          href={l.href}
          target="_blank"
          rel="noopener noreferrer"
          title={l.title}
          className="inline-flex items-center gap-0.5 text-primary hover:underline"
        >
          {l.label}
          <ExternalLink className="h-2.5 w-2.5" aria-hidden />
        </a>
      ))}
    </p>
  );
}
