# scView Future Roadmap

## Phase 6: Polish & Testing (Next Up)

These items were deferred during the initial build and are the immediate next steps.

### Visual Polish
- **Inter font**: Load Inter from Google Fonts (or bundle); apply globally via Tailwind config
- **Skeleton loading states**: Replace spinner placeholders with shimmer skeletons matching panel layout
- **Panel transitions**: 150ms fade+slide when switching between panels
- **Glassmorphism**: Subtle backdrop-blur on floating plot controls (point size, opacity sliders)
- **Dark mode**: Full dark theme option (sidebar already dark; extend to content area)

### Performance Optimization
- **React.lazy + Suspense**: Code-split each panel so only the active panel's JS is loaded
- **React.memo**: Wrap expensive components (EmbeddingScatter, ViolinPlot) to prevent unnecessary re-renders
- **deck.gl DataFilterExtension**: GPU-based filtering for selections instead of CPU-side dimming
- **Debounce optimization**: Ensure all search inputs and slider controls are properly debounced
- **Virtualized tables**: TanStack Virtual for marker gene tables with 10k+ rows
- **Arrow response compression**: GZip middleware is enabled; verify it applies to Arrow IPC responses

### Error Handling
- **ErrorBoundary per panel**: One panel crashing shouldn't break others — add boundaries in PanelContainer
- **Corrupt file detection**: Validate h5ad files on upload (check HDF5 structure)
- **Network retry with backoff**: React Query's retry config for transient failures
- **Conversion failure reporting**: Surface R error messages to the user with actionable hints
- **WebSocket reconnection**: Auto-reconnect with exponential backoff for progress updates

### Testing
- **Backend (pytest + httpx)**:
  - Unit tests for AnnData adaptor with fixture h5ad files
  - API integration tests for all endpoints (upload, embeddings, expression, assessment)
  - Pipeline runner tests (verify output h5ad is valid)
  - Arrow serializer tests (roundtrip: numpy → Arrow → numpy)
- **Frontend (Vitest + React Testing Library)**:
  - Component tests for each panel
  - Store tests for Zustand actions
  - Hook tests for useEmbedding
  - Arrow decoder worker tests
- **End-to-End (Playwright)**:
  - Full workflow: upload → overview → gene search → export
  - Assessment workflow: upload raw data → run pipeline → verify panels populate
  - Seurat upload → conversion → visualization

### Documentation
- **README.md**: Project overview with screenshots, quick start, feature list
- **Screenshots**: Capture each panel after UI polish is done

---

## Near-Term Enhancements

### Multi-Dataset Comparison
- Side-by-side scatter plots for two datasets
- Aligned embedding comparison (e.g., before/after treatment)
- Cross-dataset gene expression comparison

### Improved Heatmap
- Top N markers per cluster heatmap using Plotly
- Row/column dendrograms
- Smart downsampling for >5k cells (uniform sample per cluster)
- Custom gene list heatmap

### 3D Embedding View
- `OrbitView` + `PointCloudLayer` for 3D embeddings (PCA, UMAP 3D)
- Smooth transition between 2D and 3D views
- Mouse-drag rotation, scroll zoom

### Spatial Transcriptomics Support
- Load Visium/MERFISH/Slide-seq spatial coordinates
- Tissue image background overlay
- Spatial gene expression maps

### Enhanced Marker Gene Analysis
- Interactive volcano plot (logFC vs -log10 p-value)
- Click to select genes from volcano plot
- Gene set enrichment directly from selected markers

### Session Persistence
- Save/restore analysis sessions (selected genes, panel state, view settings)
- Shareable session URLs for collaboration
- Bookmarkable views

---

## Medium-Term Features

### Cell Type Annotation
- Integration with CellTypist or scType for automated cell type annotation
- Manual annotation interface with undo/redo
- Annotation confidence scores
- Community reference datasets

### Batch Effect Visualization
- Batch-specific embeddings colored by batch
- Integration score metrics (kBET, LISI)
- Before/after batch correction comparison

### Differential Expression On-the-fly
- Select two groups via lasso → compute DE genes between them
- Real-time DE using Wilcoxon rank-sum (fast for <50k cells)
- Results appear in a temporary table with volcano plot

### Advanced Gene Set Analysis
- GSVA/ssGSEA scoring beyond simple mean-based scoring
- Hallmark, C2, C5, C7 MSigDB collections
- Custom GMT file upload
- Leading-edge gene analysis

### Export Enhancements
- **PDF reports**: Multi-page PDF with all panels, generated server-side via kaleido
- **deck.gl canvas capture**: PNG export of the WebGL scatter plot
- **Plotly SVG export**: Publication-quality vector graphics
- **Session export**: Download complete analysis state as JSON

---

## Long-Term Vision

### Public Hosting Mode
- Multi-tenant authentication (OAuth / API keys)
- User accounts with private dataset storage
- Shared public datasets with read-only access
- Resource quotas per user

### Collaborative Features
- Real-time shared sessions (multiple users viewing the same dataset)
- Annotation collaboration (multiple users labeling cell types)
- Comments and notes on specific cell populations

### Integration Ecosystem
- **Jupyter widget**: Embed scView panels in Jupyter notebooks
- **R/Shiny bridge**: Use scView visualizations from R/Shiny apps
- **CellxGene compatibility**: Import/export CellxGene datasets
- **Observatory integration**: Direct upload from data portals (HCA, CZI CELLxGENE)

### Advanced AI Features
- **Natural language queries**: "Show me the top differentially expressed genes in cluster 3"
- **Automated analysis reports**: LLM generates a written summary of the dataset
- **Anomaly detection**: Flag unusual cell populations or quality issues
- **Parameter optimization**: LLM suggests optimal resolution for clustering based on silhouette scores

### Performance at Scale
- **1M+ cells**: WebGL instanced rendering or texture-based point rendering
- **Server-side tiling**: Pre-compute tile pyramids for very large embeddings
- **Distributed backend**: Ray or Dask for parallel computation on large datasets
- **Streaming Arrow**: Incremental data loading for smooth progressive rendering

---

## Technical Debt to Address

| Item | Priority | Notes |
|------|----------|-------|
| WebSocket progress implementation | High | Stubs exist; need full implementation for upload/pipeline progress |
| TanStack Table integration | Medium | Marker genes panel uses plain HTML tables; should use TanStack Virtual |
| Tailwind CSS v4 migration | Low | Currently uses v4 imports; verify all utility classes work as expected |
| TypeScript strict mode | Medium | Enable strict mode and fix any type gaps |
| Bundle size audit | Low | Check tree-shaking of plotly.js, deck.gl, apache-arrow |
| Accessibility (a11y) | Medium | Keyboard navigation, ARIA labels, screen reader support |
| Internationalization | Low | Not a priority for a scientific tool, but structure supports it |

---

## Contributing

To contribute to scView:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Follow the coding patterns in [DEVELOPMENT.md](./DEVELOPMENT.md)
4. Add tests for new functionality
5. Submit a pull request with a clear description

For bug reports and feature requests, open an issue on GitHub.
