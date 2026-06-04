const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api/v1";
const RETRYABLE_STATUS = new Set([502, 503, 504]);
const RETRY_DELAYS_MS = [250, 750, 1500];

export class HttpError extends Error {
  status: number;
  code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.code = code;
  }
}

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function isRetryableRequest(init?: RequestInit) {
  const method = init?.method?.toUpperCase() ?? "GET";
  return method === "GET";
}

async function fetchOnce<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({})) as { error?: { code?: string; message?: string } };
    throw new HttpError(
      res.status,
      body?.error?.code ?? "UNKNOWN",
      body?.error?.message ?? res.statusText,
    );
  }
  return res.json() as Promise<T>;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const shouldRetry = isRetryableRequest(init);

  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt += 1) {
    try {
      return await fetchOnce<T>(path, init);
    } catch (error) {
      const retryableStatus = error instanceof HttpError && RETRYABLE_STATUS.has(error.status);
      const retryableNetworkError = !(error instanceof HttpError);

      if (!shouldRetry || attempt === RETRY_DELAYS_MS.length || (!retryableStatus && !retryableNetworkError)) {
        throw error;
      }

      await delay(RETRY_DELAYS_MS[attempt]);
    }
  }

  throw new Error("Unreachable retry state");
}

export function apiGet<T>(path: string, params?: Record<string, string | undefined>): Promise<T> {
  // Drop undefined/null/empty values so they are not serialized as the literal string "undefined".
  const entries = Object.entries(params ?? {}).filter(([, v]) => v != null && v !== "") as [string, string][];
  const qs = entries.length ? "?" + new URLSearchParams(entries).toString() : "";
  return apiFetch<T>(`${path}${qs}`);
}

export function apiPost<T>(path: string, body: unknown): Promise<T> {
  return apiFetch<T>(path, { method: "POST", body: JSON.stringify(body) });
}

export function apiPut<T>(path: string, body: unknown): Promise<T> {
  return apiFetch<T>(path, { method: "PUT", body: JSON.stringify(body) });
}

export function apiDelete<T>(path: string): Promise<T> {
  return apiFetch<T>(path, { method: "DELETE" });
}
