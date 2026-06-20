import { test, expect } from "@playwright/test";

test("streams a turn and lights up telemetry", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("backend url").fill("http://localhost:8000");
  await page.getByPlaceholder(/message/i).fill("hello");
  await page.getByRole("button", { name: /send/i }).click();

  // assistant text streams in
  await expect(page.getByText(/Routing across eight H100s/i)).toBeVisible({ timeout: 10000 });
  // a latency stat populates
  await expect(page.getByText("tokens")).toBeVisible();
  // the GPU/expert viz is present and active
  await expect(page.getByText("GPU / expert routing")).toBeVisible();
});
