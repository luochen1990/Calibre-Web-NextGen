import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium, expect } from '@playwright/test';

// Standalone component check using the existing Vite/Playwright dependencies.
// From frontend/: node tests/browser/browsePaging.check.mjs
// All API responses are mocked; no running Calibre-Web server is needed.
const server = await createServer({ configFile: false, esbuild: { jsx: 'automatic' },
  server: { host: '127.0.0.1', port: 0 },
  plugins: [{ name: 'browse-unit-harness', configureServer(server) {
    server.middlewares.use((request, response, next) => {
      if (request.url !== '/') return next();
      response.setHeader('Content-Type', 'text/html');
      response.end('<div id="root"></div><script type="module" src="/tests/browser/browsePaging.fixture.tsx"></script>');
    });
  } }],
});
let browser;
try {
  await server.listen();
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const requests = [];
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/v1/**', async route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/api/v1/auth/me') {
      return route.fulfill({ json: { id: 1, locale: 'en', role: {} } });
    }
    requests.push(url);
    const paged = url.pathname.endsWith('authors') || url.pathname.endsWith('series');
    const total = url.searchParams.has('q') ? 1 : url.searchParams.has('min_count') ? 240 : 500;
    const offset = Number(url.searchParams.get('offset') ?? 0);
    const length = paged ? Math.min(200, Math.max(0, total - offset)) : 12;
    const items = Array.from({ length }, (_, index) => ({ id: offset + index + 1,
      name: url.searchParams.has('q') ? 'Moon' : `Entity ${offset + index + 1}`, count: 3 }));
    return route.fulfill({ json: paged ? { items, total, unfiltered_total: 500,
      next_offset: offset + length < total ? offset + length : null } : { items } });
  });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/`);
  await expect(page.getByRole('listitem')).toHaveCount(200);
  await expect(page.getByRole('heading', { level: 1 }).locator('..')).toHaveText('Authors500');
  await expect(page.getByRole('status')).toHaveText('Showing 500 of 500 authors.');
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await expect(page.getByRole('listitem')).toHaveCount(400);
  assert.ok(requests.some(url => url.searchParams.get('offset') === '200'));
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await expect(page.getByRole('listitem')).toHaveCount(500);
  console.log('PASS: real scrolling loads 200, then 400, then 500 authors and terminates');

  for (const plural of ['authors', 'series']) {
    await page.evaluate(plural => window.renderBrowse(plural), plural);
    const search = page.getByRole('searchbox');
    await expect(search).toHaveValue('');
    const before = requests.length;
    await search.fill('Moon');
    await page.waitForTimeout(100);
    assert.equal(requests.length, before, 'name request must be debounced');
    await expect(page.getByRole('listitem')).toHaveCount(1);
    assert.ok(requests.some(url => url.pathname.endsWith(plural) && url.searchParams.get('q') === 'Moon'));
    await expect(page.getByRole('status')).toHaveText(`Showing 1 of 500 ${plural}.`);
    await expect(page.getByRole('heading', { level: 1 }).locator('..')).toHaveText(`${plural === 'authors' ? 'Authors' : 'Series'}500`);
    const minimum = page.getByLabel('Minimum books', { exact: true });
    await minimum.fill('2');
    await expect.poll(() => requests.at(-1).searchParams.get('min_count')).toBe('2');
    const validRequests = requests.length;
    await minimum.fill('-1');
    await expect(minimum).toHaveAttribute('aria-invalid', 'true');
    await expect(page.getByRole('alert')).toHaveText('Enter a whole number of at least 1.');
    await search.fill('Different');
    await page.waitForTimeout(400);
    assert.equal(requests.length, validRequests, 'invalid minimum must prevent every list request');
    await expect(page.getByRole('status')).toHaveText(`Showing 0 of 500 ${plural}.`);
    await minimum.fill('');
    await expect(page.getByRole('alert')).toHaveCount(0);
    await page.getByRole('button', { name: 'List view', exact: true }).click();
    console.log(`PASS: ${plural} debounce, server filters, count semantics, invalid minimum and recovery`);
  }
  await page.evaluate(() => window.renderBrowse('tags'));
  await expect(page.getByRole('listitem')).toHaveCount(12);
  await expect(page.getByRole('button', { name: 'List view', exact: true })).toHaveAttribute('aria-pressed', 'true');
  const tagRequests = requests.length;
  await page.getByRole('searchbox').fill('Entity 12');
  await expect(page.getByRole('listitem')).toHaveCount(1);
  assert.equal(requests.length, tagRequests, 'tags filter locally');
  assert.equal(requests.at(-1).search, '');
  assert.deepEqual(errors, []);
  console.log('PASS: plural reset, compact toggle persistence, full-list tags and local filtering; no browser errors');
} finally {
  await browser?.close();
  await server.close();
}
