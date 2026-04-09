async function tryLoad(path) {
  const response = await fetch(path);
  if (!response.ok) {
    return null;
  }
  const payload = await response.json();
  if (!payload.runs || payload.runs.length === 0) {
    return null;
  }
  return payload;
}

export async function loadScenario(paths) {
  const candidates = Array.isArray(paths) ? paths : [paths];
  for (const path of candidates) {
    const payload = await tryLoad(path);
    if (payload) {
      return payload;
    }
  }
  throw new Error(`Failed to load scenario from candidates: ${candidates.join(", ")}`);
}
