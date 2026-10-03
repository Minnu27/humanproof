// Calibration aid for demo deployments. Keeps the last attempt's raw measurements
// in memory (numbers only: no images, no audio) so they can be copied from the
// result screen and compared with how the server scored them. Nothing here is
// sent anywhere; it is only read when the person presses "Copy diagnostics".
const capture: Record<string, unknown> = {};

export function recordCapture(step: "gaze" | "motor" | "voice", data: unknown): void {
  capture[step] = data;
}

export function getCapture(): Record<string, unknown> {
  return capture;
}
