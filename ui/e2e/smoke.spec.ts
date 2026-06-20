import { test, expect } from "@playwright/test";

test("race view: two engines stream and a speedup is measured", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("backend url").fill("http://localhost:8000");
  // Race is the default view.
  await page.getByRole("button", { name: /run race/i }).click();

  // Both engine lanes stream content.
  await expect(page.getByText("Conifer").first()).toBeVisible();
  await expect(page.getByText("vLLM").first()).toBeVisible();
  await expect(page.getByText(/Routing across eight H100s/i).first()).toBeVisible({ timeout: 15000 });

  // A measured speedup locks in (× appears in the center badge).
  await expect(page.getByText(/faster .* end-to-end/i)).toBeVisible({ timeout: 20000 });
  // Test-time-compute panel translates speed into quality.
  await expect(page.getByText(/test-time compute/i)).toBeVisible();
});

test("console view: single stream lights up telemetry", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("backend url").fill("http://localhost:8000");
  await page.getByRole("button", { name: /^console$/i }).click();
  await page.getByPlaceholder(/message/i).fill("hello");
  await page.getByRole("button", { name: /send/i }).click();

  // assistant text streams in
  await expect(page.getByText(/Routing across eight H100s/i)).toBeVisible({ timeout: 10000 });
  // a latency stat populates
  await expect(page.getByText("tokens")).toBeVisible();
  // the GPU/expert viz is present and active
  await expect(page.getByText("GPU / expert routing")).toBeVisible();
});
