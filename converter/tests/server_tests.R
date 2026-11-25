##
## server_tests.R
## The server's conversion child (run_in_child) and its memory report, against a fake
## cgroup folder: a child that is killed is said as killed, one the OOM killer took
## (memory.events oom_kill went up) as out of memory with the limit. Run from the
## converter folder (server.R sources convert.R by relative path).
##

cg <- tempfile("cgroup")
dir.create(cg)
put <- function(name, text) writeLines(text, file.path(cg, name))
put("memory.max", "3221225472")
put("memory.current", "500000000")
put("memory.stat", c("anon 300000000", "inactive_file 100000000", "active_file 5"))
put("memory.events", c("low 0", "high 0", "max 4", "oom 1", "oom_kill 1"))
Sys.setenv(SCVIEW_CGROUP_DIR = cg)

srv <- new.env()
srv$runServer <- function(...) invisible(NULL)  # define the handlers, do not serve
sys.source("server.R", envir = srv)

failed <- 0
check <- function(what, ok) {
  cat(sprintf("[server] %s %s\n", if (isTRUE(ok)) "PASS" else "FAIL", what))
  if (!isTRUE(ok)) failed <<- failed + 1
}

m <- srv$memory_state()
check("memory limit from memory.max", identical(m$memory_limit_bytes, 3221225472))
check("use leaves out the file cache", identical(m$memory_used_bytes, 4e8))
check("OOM kills from memory.events", identical(m$oom_kills, 1))

st <- tempfile("status")
writeLines(c("Name:\tR", "VmPeak:\t  999999 kB", "VmHWM:\t    2048 kB", "VmRSS:\t    1024 kB"), st)
check("the peak is VmHWM, in bytes", identical(srv$process_peak_bytes(st), 2097152))
check("no status file, no peak", is.null(srv$process_peak_bytes(file.path(cg, "none"))))

res <- srv$run_in_child(function() list(success = TRUE, message = "done"), "converting x.rds")
check("a child's answer comes back", isTRUE(res$success) && res$message == "done")
check("with the child's measured peak", is.numeric(res$peak_bytes) && res$peak_bytes > 1e6)
hb <- jsonlite::fromJSON(srv$app$call(list(REQUEST_METHOD = "GET", PATH_INFO = "/health"))$body)
check("/health says the last conversion's peak",
      hb$last_conversion_peak_bytes == res$peak_bytes && isTRUE(hb$last_conversion_ok) &&
        hb$last_conversion_what == "converting x.rds" && hb$last_conversion_memory_limit_bytes == 3221225472)

res <- srv$run_in_child(function() stop("boom"), "converting x.rds")
check("an error in the child is an error, not a crash",
      identical(res$success, FALSE) && grepl("boom", res$error))

res <- srv$run_in_child(function() tools::pskill(Sys.getpid(), tools::SIGKILL), "converting x.rds")
check("a killed child is said as killed", identical(res$error_kind, "stopped") &&
        grepl("stopped converting x.rds", res$error))

oom <- function() {
  put("memory.events", c("low 0", "high 0", "max 9", "oom 2", "oom_kill 2"))
  tools::pskill(Sys.getpid(), tools::SIGKILL)
}
res <- srv$run_in_child(oom, "converting SCA.rds")
check("an OOM-killed child is out of memory", identical(res$error_kind, "out_of_memory"))
check("the sentence names the limit once, then the file",
      identical(res$error, "The converter ran out of its 3.0 GB converting SCA.rds."))
check("a killed child has no peak", is.null(res$peak_bytes))
body <- srv$error_body(res)
check("the error body carries the kind and the limit",
      identical(body$error_kind, "out_of_memory") && identical(body$memory_limit_bytes, 3221225472))
hb <- jsonlite::fromJSON(srv$app$call(list(REQUEST_METHOD = "GET", PATH_INFO = "/health"))$body)
check("/health: the last run ran out of memory, no peak",
      identical(hb$last_conversion_error_kind, "out_of_memory") && identical(hb$last_conversion_ok, FALSE) &&
        is.null(hb$last_conversion_peak_bytes))

put("memory.max", "max")
check("no limit set reads as none", is.null(srv$memory_state()$memory_limit_bytes))

h <- srv$app$call(list(REQUEST_METHOD = "GET", PATH_INFO = "/health"))
hb <- jsonlite::fromJSON(h$body)
check("/health says when it started and its memory",
      h$status == 200L && nzchar(hb$started_at) && hb$oom_kills == 2 && hb$memory_used_bytes == 4e8)

# DietSeurat is advised after an out-of-memory kill only when it would drop a layer
# (convert.R, which server.R sources into the global environment)
suppressPackageStartupMessages(library(SeuratObject))
m <- Matrix::rsparsematrix(30, 20, 0.3, rand.x = function(n) rpois(n, 5) + 1)
dimnames(m) <- list(paste0("g", 1:30), paste0("c", 1:20))
so <- suppressWarnings(CreateSeuratObject(counts = m))
check("counts only: nothing for DietSeurat to drop", identical(seurat_droppable_layers(so), FALSE))
LayerData(so, layer = "data") <- log1p(m)
check("a data layer: DietSeurat would drop it", identical(seurat_droppable_layers(so), TRUE))

cat(sprintf("[server] %d failed\n", failed))
quit(status = if (failed) 1 else 0)
