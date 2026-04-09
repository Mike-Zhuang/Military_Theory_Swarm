export async function loadScenario(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load scenario: ${response.status} ${response.statusText}`);
  }
  const payload = await response.json();
  if (!payload.runs || payload.runs.length === 0) {
    throw new Error("Scenario file has no runs.");
  }
  return payload;
}
