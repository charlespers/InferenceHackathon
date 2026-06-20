export function getDefaultBase(): string {
  return (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";
}
