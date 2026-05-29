##
## server.R
## Minimal HTTP server for the scView converter service.
## Listens on port 8001 and exposes two routes:
##   GET  /health  -> health-check endpoint
##   POST /convert -> accepts JSON {input_path, output_path}, runs conversion
##

library(httpuv)
library(jsonlite)

source("convert.R")

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

log_request <- function(req) {
  cat(sprintf(
    "[%s] %s %s\n",
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    toupper(req$REQUEST_METHOD),
    req$PATH_INFO
  ))
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
      return(json_response(list(status = "ok")))
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

      input_path  <- params$input_path
      output_path <- params$output_path

      if (is.null(input_path) || is.null(output_path)) {
        return(json_response(
          list(status = "error", error = "Missing input_path or output_path"),
          status = 400L
        ))
      }

      cat(sprintf("  input_path  = %s\n", input_path))
      cat(sprintf("  output_path = %s\n", output_path))

      # Run conversion
      result <- tryCatch(
        {
          convert_seurat_to_h5ad(input_path, output_path)
        },
        error = function(e) {
          list(success = FALSE, error = conditionMessage(e))
        }
      )

      if (isTRUE(result$success)) {
        return(json_response(list(
          status  = "ok",
          message = result$message
        )))
      } else {
        return(json_response(
          list(status = "error", error = result$error),
          status = 500L
        ))
      }
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
