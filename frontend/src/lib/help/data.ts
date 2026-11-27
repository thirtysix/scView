/** Help for the Data panel (importing data). Keys: `<area>.<thing>`. */
export const DATA_HELP: Record<string, string> = {
  "upload.time-left":
    "From the upload speed over the last few seconds, so it changes when the network does. It shows after a few seconds of data; a busy or slow connection makes it grow.",
  "upload.convert":
    "scView's R converter reads the .rds or .RData object and writes it as an AnnData file scView can open. A large object takes minutes. It carries on on the server if you leave this page; the dataset then appears in your library.",
  "storage.by-rows":
    "Rewrites scView's copy of the matrix so each cell's values sit together. No value, label or result changes, and your uploaded file is not touched. It needs free disk for a second copy and takes about the time shown. History records it. Only large-dataset mode needs it.",
};
