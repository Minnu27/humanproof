import { useEffect, useRef } from "react";

export interface JobHandle {
  alive: () => boolean;
  onCleanup: (f: () => void) => void;
}

/**
 * Runs an async job exactly once per component instance (checkpoints must not be
 * started twice, even under React StrictMode's double effects) and runs the
 * job's registered cleanups when the component really unmounts.
 */
export function useOnceJob(job: (h: JobHandle) => Promise<void>): void {
  const started = useRef(false);
  const alive = useRef(true);
  const cleanups = useRef<(() => void)[]>([]);
  const jobRef = useRef(job);
  jobRef.current = job;

  useEffect(() => {
    alive.current = true;
    if (!started.current) {
      started.current = true;
      void jobRef.current({ alive: () => alive.current, onCleanup: (f) => cleanups.current.push(f) });
    }
    return () => {
      alive.current = false;
      const fs = cleanups.current;
      cleanups.current = [];
      fs.forEach((f) => f());
    };
  }, []);
}
