import { InfoTip } from "@/components/common/InfoTip";
import { nonlinearEmbedding } from "@/lib/clusterDistances";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useViewStore } from "@/stores/viewStore";

/**
 * The always-visible note on a UMAP or t-SNE plot (nothing for PCA): the gaps
 * and blob sizes between its clusters are not distances. Its link opens the
 * Unified View's cluster-distance heatmap (PCA space) under the plot.
 */
export function EmbeddingCaveat({
  embedding,
  className = "",
}: {
  /** The embedding's obsm key (X_umap, X_tsne, X_pca, ...) */
  embedding: string | null | undefined;
  className?: string;
}) {
  const method = nonlinearEmbedding(embedding);
  if (!method) return null;
  const showDistances = () => {
    useUnifiedViewStore.getState().showDistances();
    useViewStore.getState().setPanel("unified");
  };
  return (
    <div
      data-testid="embedding-caveat"
      className={`flex items-center gap-1 whitespace-nowrap rounded bg-white/80 px-1.5 py-0.5 text-[10px] text-slate-500 ${className}`}
    >
      <InfoTip align="right" width={280} label={`About distances on a ${method} plot`}>
        {method} keeps each cell&apos;s nearest neighbours, not the distances between clusters: how
        far apart two clusters sit, and how large each looks, says little about how different they
        are (Chari &amp; Pachter 2023, PLoS Comput Biol). The cluster distances are measured in the
        principal components the neighbour graph was built on.
      </InfoTip>
      <span>{method}: gaps between clusters aren&apos;t distances</span>
      <button type="button" onClick={showDistances} className="font-medium text-primary hover:underline">
        Cluster distances
      </button>
    </div>
  );
}
