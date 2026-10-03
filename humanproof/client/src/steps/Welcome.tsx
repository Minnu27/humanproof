export function Welcome({ onStart }: { onStart: () => void }) {
  return (
    <section className="card">
      <p className="eyebrow">Verification</p>
      <h1>Show you're a real person</h1>
      <p className="lead">Three short tasks, about 45 seconds in total. Nothing to type, nothing to upload.</p>
      <ol className="tasks">
        <li>
          <span className="task-n">1</span>
          <div>
            <strong>Follow a dot with your eyes</strong>
            <span className="muted">Your camera measures how your eyes move.</span>
          </div>
        </li>
        <li>
          <span className="task-n">2</span>
          <div>
            <strong>Trace a line</strong>
            <span className="muted">With your mouse, finger, or arrow keys.</span>
          </div>
        </li>
        <li>
          <span className="task-n">3</span>
          <div>
            <strong>Read five words aloud</strong>
            <span className="muted">We listen for a live human voice.</span>
          </div>
        </li>
      </ol>
      <button className="btn btn--primary" onClick={onStart}>
        Start
      </button>
    </section>
  );
}
