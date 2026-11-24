##
## server.R
## Minimal HTTP server for the scView converter service.
## Listens on port 8001 and exposes two routes:
##   GET  /health  -> health-check endpoint, with the memory this container has
##                    (cgroup limit, use, OOM kills) and when the server started
##   POST /convert -> accepts JSON {input_path, output_path[, progress_path]},
##                    runs conversion (Seurat, SingleCellExperiment,
##                    SummarizedExperiment) and returns what was kept / skipped
##   POST /export  -> accepts JSON {input_path (staged h5ad), output_path (.rds),
##                    format: seurat | sce[, progress_path]} (export_rds.R)
##

library(httpuv)
library(jsonlite)

source("convert.R")
source("convert_sce.R")
source("export_rds.R")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

json_response <- function(body_list, status = 200L) {
  list(
    status  = status,
    headers = list("Content-Type" = "application/json"),
    body    = toJSON(body_list, auto_unbox = TRUE)
  )
}

#' The JSON body as a list, or list(error = why).
read_json_body <- function(req) {
  raw <- tryCatch(req$rook.input$read_lines(), error = function(e) NULL)
  if (is.null(raw) || length(raw) == 0 || !nzchar(paste(raw, collapse = ""))) {
    return(list(error = "Empty request body"))
  }
  params <- tryCatch(fromJSON(paste(raw, collapse = "\n")), error = function(e) NULL)
  if (!is.list(params)) list(error = "Invalid JSON in request body") else params
}

log_request <- function(req) {
  cat(sprintf(
    "[%s] %s %s\n",
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    toupper(req$REQUEST_METHOD),
    req$PATH_INFO
  ))
}

# ---------------------------------------------------------------------------
# Memory (cgroup v2; NULL where it cannot be read, as on cgroup v1)
# ---------------------------------------------------------------------------

CGROUP <- Sys.getenv("SCVIEW_CGROUP_DIR", "/sys/fs/cgroup")
STARTED_AT <- format(Sys.time(), "%Y-%m-%dT%H:%M:%OS3%z")

cgroup_value <- function(name) {
  v <- tryCatch(readLines(file.path(CGROUP, name), warn = FALSE)[1], error = function(e) NA_character_)
  if (is.na(v) || v == "max") return(NULL)
  n <- suppressWarnings(as.numeric(v))
  if (is.na(n)) NULL else n
}

#' One field of a "key value" cgroup file (memory.events, memory.stat).
cgroup_field <- function(name, field) {
  lines <- tryCatch(readLines(file.path(CGROUP, name), warn = FALSE), error = function(e) character(0))
  hit <- grep(paste0("^", field, " "), lines, value = TRUE)
  if (length(hit) == 0) return(NULL)
  as.numeric(sub("^\\S+ ", "", hit[1]))
}

#' The memory the container has: limit (NULL = none set), in use (file cache the
#' kernel can drop left out, as the backend's gauge does) and OOM kills so far.
memory_state <- function() {
  current <- cgroup_value("memory.current")
  cache <- cgroup_field("memory.stat", "inactive_file") %||% 0
  list(
    memory_limit_bytes = cgroup_value("memory.max"),
    memory_used_bytes  = if (is.null(current)) NULL else max(0, current - cache),
    oom_kills          = cgroup_field("memory.events", "oom_kill")
  )
}

#' Run a conversion in a forked child. When it needs more memory than the container
#' has, the kernel kills the child (the largest process), not this server: the answer
#' then says it ran out of memory (memory.events oom_kill went up) instead of the
#' connection dropping, and the server stays up. All of a conversion's memory goes back
#' with the child.
run_in_child <- function(work, what) {
  before <- memory_state()$oom_kills
  job <- parallel::mcparallel(work(), silent = FALSE)
  # a child that was killed delivers nothing (mccollect warns; the answer below says why)
  res <- suppressWarnings(parallel::mccollect(job, wait = TRUE))[[1]]
  if (is.list(res)) return(res)
  if (inherits(res, "try-error")) {
    return(list(success = FALSE, error = paste0("Conversion failed: ", conditionMessage(attr(res, "condition")))))
  }
  mem <- memory_state()
  oom <- !is.null(mem$oom_kills) && mem$oom_kills > (before %||% 0)
  limit <- mem$memory_limit_bytes
  if (oom) {
    return(list(
      success = FALSE, error_kind = "out_of_memory", memory_limit_bytes = limit,
      error = sprintf("The converter ran out of memory %s%s.", what,
                      if (is.null(limit)) "" else sprintf(" (it has %.1f GB)", limit / 1024^3))
    ))
  }
  list(success = FALSE, error_kind = "stopped",
       error = sprintf("The converter's process stopped %s without an answer (it was killed).", what))
}

#' The error body: the sentence, and for a conversion that ran out of memory its kind
#' and the limit, so the backend can say what to do.
error_body <- function(result) {
  body <- list(status = "error", error = result$error)
  if (!is.null(result$error_kind)) body$error_kind <- result$error_kind
  if (!is.null(result$memory_limit_bytes)) body$memory_limit_bytes <- result$memory_limit_bytes
  body
}

# ---------------------------------------------------------------------------
# Application handler
# ---------------------------------------------------------------------------

app <- list(
  call = function(req) {

    log_request(req)

    method <- toupper(req$REQUEST_METHOD)
    path   <- req$PATH_INFO

    # ---- GET /health ------------------------------------------------------
    if (method == "GET" && path == "/health") {
      return(json_response(c(list(status = "ok", started_at = STARTED_AT), memory_state())))
    }

    # ---- POST /convert ----------------------------------------------------
    if (method == "POST" && path == "/convert") {

      # Read the request body
      body_raw <- tryCatch(
        {
          req$rook.input$read_lines()
        },
        error = function(e) NULL
      )

      if (is.null(body_raw) || length(body_raw) == 0 || nchar(body_raw) == 0) {
        return(json_response(
          list(status = "error", error = "Empty request body"),
          status = 400L
        ))
      }

      params <- tryCatch(
        fromJSON(body_raw),
        error = function(e) NULL
      )

      if (is.null(params)) {
        return(json_response(
          list(status = "error", error = "Invalid JSON in request body"),
          status = 400L
        ))
      }

      input_path    <- params$input_path
      output_path   <- params$output_path
      progress_path <- params$progress_path

      if (is.null(input_path) || is.null(output_path)) {
        return(json_response(
          list(status = "error", error = "Missing input_path or output_path"),
          status = 400L
        ))
      }

      cat(sprintf("  input_path  = %s\n", input_path))
      cat(sprintf("  output_path = %s\n", output_path))

      # Run conversion (in a child process: its memory goes back when it ends)
      result <- run_in_child(function() tryCatch(
        {
          convert_rds_to_h5ad(input_path, output_path, progress_path)
        },
        error = function(e) {
          list(success = FALSE, error = conditionMessage(e))
        }
      ), sprintf("converting %s", basename(input_path)))

      if (isTRUE(result$success)) {
        return(json_response(list(
          status  = "ok",
          message = result$message,
          report  = result$report
        )))
      } else {
        return(json_response(error_body(result), status = 500L))
      }
    }

    # ---- POST /export: staged h5ad -> Seurat / SCE .rds --------------------
    if (method == "POST" && path == "/export") {
      params <- read_json_body(req)
      if (!is.null(params$error)) {
        return(json_response(list(status = "error", error = params$error), status = 400L))
      }
      if (is.null(params$input_path) || is.null(params$output_path)) {
        return(json_response(
          list(status = "error", error = "Missing input_path or output_path"), status = 400L))
      }
      result <- run_in_child(function() tryCatch(
        export_h5ad_to_rds(params$input_path, params$output_path,
                           params$format %||% "seurat", params$progress_path),
        error = function(e) list(success = FALSE, error = conditionMessage(e))
      ), "writing the .rds file")
      if (isTRUE(result$success)) {
        return(json_response(list(status = "ok", message = result$message, report = result$report)))
      }
      return(json_response(error_body(result), status = 500L))
    }

    # ---- fallback: 404 ----------------------------------------------------
    json_response(
      list(status = "error", error = paste0("Not found: ", method, " ", path)),
      status = 404L
    )
  }
)

# ---------------------------------------------------------------------------
# Start server
# ---------------------------------------------------------------------------

port <- as.integer(Sys.getenv("PORT", "8001"))

cat(sprintf("scView converter service starting on port %d ...\n", port))

runServer("0.0.0.0", port, app)
