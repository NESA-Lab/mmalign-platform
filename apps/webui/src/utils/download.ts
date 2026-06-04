// Client-side file export helpers (CSV / JSON) used by overview & benchmark pages.

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadJson(filename: string, data: unknown): void {
  triggerDownload(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
    filename,
  );
}

// RFC 4180 escaping: quote cells containing comma/quote/newline; double inner quotes.
function csvCell(value: unknown): string {
  const s = value === null || value === undefined ? "" : String(value);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function downloadCsv(
  filename: string,
  header: string[],
  rows: unknown[][],
): void {
  const lines = [header, ...rows].map((row) => row.map(csvCell).join(","));
  triggerDownload(new Blob([lines.join("\n")], { type: "text/csv" }), filename);
}
