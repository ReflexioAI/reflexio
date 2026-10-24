/** Raised for any failed call: network, timeout, HTTP error, or a response that is not the API's. */
export class ReflexioError extends Error {
  /** HTTP status, when the server answered. */
  readonly status?: number;
  /** Decoded response body (JSON when the server sent JSON, otherwise text). */
  readonly body?: unknown;

  constructor(message: string, options: { status?: number; body?: unknown; cause?: unknown } = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = "ReflexioError";
    this.status = options.status;
    this.body = options.body;
  }
}
