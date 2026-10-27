import { createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowseList } from '../../src/pages/BrowseList';
import { I18nProvider } from '../../src/lib/i18n';

declare global { interface Window { renderBrowse: (plural: string) => void } }

const client = new QueryClient({ defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } } });
client.setQueryData(['me'], { id: 1, locale: 'en', role: {} });
const root = createRoot(document.getElementById('root')!);
window.renderBrowse = (plural: string) => root.render(createElement(QueryClientProvider, { client },
  createElement(I18nProvider, { locale: 'en', children: createElement(BrowseList,
    { plural, title: plural === 'authors' ? 'Authors' : plural === 'series' ? 'Series' : 'Tags' }) })));
window.renderBrowse('authors');
