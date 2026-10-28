##
## make_fixtures.R <out_dir> [<visium spatial dir> <visium filtered .h5>]
## The small .h5Seurat files test_h5seurat.py reads, written with SeuratDisk's own
## SaveH5Seurat, plus the same Visium objects as .rds for the converter tests
## (converter/tests). SeuratDisk is unmaintained and not in any scView image: this ran
## once in the converter image (R 4.4, Seurat 5.1.0, SeuratObject 5.4.0) with
## SeuratDisk 0.0.0.9021 from GitHub (mojaveazure/seurat-disk) and hdf5r in a
## throwaway library. The Visium objects are cut from 10x's V1_Human_Lymph_Node
## (Space Ranger 1.1): 60 neighbouring spots, 120 genes, the low-res image kept at every 10th pixel
## (its scale factor divided by 10 to match).
##

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratDisk)
  library(Matrix)
})
options(Seurat.object.assay.version = "v3")  # SeuratDisk writes v3 assays only

args <- commandArgs(trailingOnly = TRUE)
out <- args[1]
if (is.na(out)) stop("usage: make_fixtures.R <out_dir> [<spatial dir> <filtered .h5>]")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
set.seed(7)

save_h5 <- function(obj, name) {
  path <- file.path(out, paste0(name, ".h5Seurat"))
  SaveH5Seurat(obj, filename = path, overwrite = TRUE, verbose = FALSE)
  cat(sprintf("wrote %s (%.0f kB)\n", path, file.size(path) / 1024))
}

# ---- 1. pbmc_small (SeuratObject's example: 80 cells x 230 genes, PCA + tSNE) ------
pbmc <- UpdateSeuratObject(pbmc_small)
pbmc$is_high <- pbmc$nCount_RNA > median(pbmc$nCount_RNA)  # a logical column
pbmc$note <- ifelse(seq_len(ncol(pbmc)) %% 5 == 0, NA, "kept")  # strings with NA
save_h5(pbmc, "pbmc_small")

# ---- 2. the same with a second assay: a 4-antibody ADT panel -------------------------
adt <- matrix(rpois(4 * ncol(pbmc), 30), 4, ncol(pbmc),
              dimnames = list(c("CD3", "CD4", "CD8", "IgG1-ctrl"), colnames(pbmc)))
pbmc_adt <- pbmc
pbmc_adt[["ADT"]] <- CreateAssayObject(counts = adt)
DefaultAssay(pbmc_adt) <- "RNA"
save_h5(pbmc_adt, "pbmc_small_adt")

# ---- 3. Visium: VisiumV2 (Seurat >= 5.1) and VisiumV1 images -------------------------
if (length(args) >= 3) {
  spatial_dir <- args[2]
  counts <- Read10X_h5(args[3])
  img <- Read10X_Image(spatial_dir)  # VisiumV2: centroids, low-res image, scale factors
  # 60 neighbouring spots (a patch around the middle one), so the grid can be recovered
  xy <- GetTissueCoordinates(img)
  xy <- xy[intersect(rownames(xy), colnames(counts)), c("x", "y")]
  mid <- colMeans(xy)
  cells <- rownames(xy)[order((xy$x - mid[1])^2 + (xy$y - mid[2])^2)][1:60]
  counts <- counts[, cells]
  genes <- names(sort(Matrix::rowSums(counts), decreasing = TRUE))[1:120]
  counts <- counts[genes, ]

  shrink <- function(image) {
    small <- image@image[seq(1, dim(image@image)[1], by = 10),
                         seq(1, dim(image@image)[2], by = 10), , drop = FALSE]
    image@image <- small
    image@scale.factors$lowres <- image@scale.factors$lowres / 10
    image
  }

  v2 <- CreateSeuratObject(counts, assay = "Spatial")
  img2 <- shrink(img[cells])
  DefaultAssay(img2) <- "Spatial"
  v2[["slice1"]] <- img2
  v2$region <- factor(ifelse(seq_along(cells) <= 30, "cortex", "medulla"))
  save_h5(v2, "visium_v2")
  saveRDS(v2, file.path(out, "visium_v2.rds"))

  # VisiumV1 (Seurat 3.2-5.0): a coordinates table with the array grid (row, col) and
  # full-res pixels (imagerow, imagecol).
  pos <- read.csv(file.path(spatial_dir, "tissue_positions_list.csv"), header = FALSE,
                  col.names = c("barcode", "tissue", "row", "col", "imagerow", "imagecol"))
  rownames(pos) <- pos$barcode
  pos <- pos[cells, c("tissue", "row", "col", "imagerow", "imagecol")]
  sf <- img@scale.factors
  v1img <- new(Class = "VisiumV1", image = img2@image, scale.factors = scalefactors(
    spot = sf$spot, fiducial = sf$fiducial, hires = sf$hires, lowres = sf$lowres / 10),
    coordinates = pos, spot.radius = sf$fiducial * (sf$lowres / 10) / max(dim(img2@image)),
    assay = "Spatial", key = "slice1_")
  v1 <- CreateSeuratObject(counts, assay = "Spatial")
  v1$region <- v2$region
  v1[["slice1"]] <- v1img
  save_h5(v1, "visium_v1")
  saveRDS(v1, file.path(out, "visium_v1.rds"))

  # what the tests check against: the positions file itself
  write.csv(data.frame(barcode = cells, pos), file.path(out, "visium_positions.csv"),
            row.names = FALSE)
  writeLines(jsonlite::toJSON(list(spot = sf$spot, fiducial = sf$fiducial, hires = sf$hires,
                                   lowres = sf$lowres / 10, image_dim = dim(img2@image)),
                              auto_unbox = TRUE, digits = NA),
             file.path(out, "visium_scalefactors.json"))
}
