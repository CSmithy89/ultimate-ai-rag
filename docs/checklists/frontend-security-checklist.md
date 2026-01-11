# Frontend Security Review Checklist

This checklist helps catch common frontend security issues before code review. Use it when developing or reviewing React/Next.js components.

Origin: Epic 21 Retrospective (Action Item 1)

## 1. Sensitive Data Handling

### Key/Credential Detection

- [ ] **Comprehensive regex pattern** covers all sensitive keys:
  - `password`, `secret`, `token`, `key`, `auth`, `bearer`, `jwt`
  - `session`, `cookie`, `oauth`, `credential`, `api_key`
  - `private_key`, `access_token`, `refresh_token`, `client_secret`
  - `signature`, `ssn`, `credit_card`

- [ ] **Word boundaries used** to avoid false positives (e.g., `monkey` matching `key`)
  ```typescript
  // Good: Uses word boundary
  const SENSITIVE_PATTERN = /\b(password|secret|token|key|auth)\b/i;

  // Bad: No word boundary
  const SENSITIVE_PATTERN = /(password|secret|token|key|auth)/i;
  ```

- [ ] **Value-embedded credentials detected** (e.g., connection strings)
  ```typescript
  // Detect patterns like: password=secret, token=abc123
  const VALUE_PATTERN = /(?:password|token|secret|key)[:=]\s*\S+/gi;
  ```

### Storage Security

- [ ] **localStorage/sessionStorage data validated** with Zod before use
  ```typescript
  const stored = localStorage.getItem('settings');
  const parsed = SettingsSchema.safeParse(JSON.parse(stored || '{}'));
  if (!parsed.success) { /* handle invalid data */ }
  ```

- [ ] **No sensitive data in client storage** (tokens in httpOnly cookies instead)

- [ ] **No sensitive data logged** to console in production
  ```typescript
  if (process.env.NODE_ENV !== 'production') {
    console.debug('Debug info:', data);
  }
  ```

## 2. UI Rendering Security

### Markdown/HTML Sanitization

- [ ] **Markdown rendered with sanitization** (rehype-sanitize or similar)
  ```typescript
  import rehypeSanitize from 'rehype-sanitize';

  <ReactMarkdown rehypePlugins={[rehypeSanitize]}>
    {userContent}
  </ReactMarkdown>
  ```

- [ ] **User-provided HTML escaped** or sanitized before rendering

- [ ] **No `dangerouslySetInnerHTML`** without sanitization
  ```typescript
  // Bad
  <div dangerouslySetInnerHTML={{ __html: userInput }} />

  // Good - sanitize first
  import DOMPurify from 'dompurify';
  <div dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(userInput) }} />
  ```

### External Resources

- [ ] **External images proxied** through backend or allowlisted domains
  ```typescript
  const ALLOWED_IMAGE_DOMAINS = ['cdn.example.com', 'images.trusted.com'];

  function isAllowedImageUrl(url: string): boolean {
    try {
      const parsed = new URL(url);
      return ALLOWED_IMAGE_DOMAINS.includes(parsed.hostname);
    } catch {
      return false;
    }
  }
  ```

- [ ] **iframes sandboxed** with appropriate restrictions
  ```tsx
  <iframe
    src={trustedUrl}
    sandbox="allow-scripts allow-same-origin"
    referrerPolicy="no-referrer"
  />
  ```

- [ ] **External links use** `rel="noopener noreferrer"`
  ```tsx
  <a href={externalUrl} target="_blank" rel="noopener noreferrer">
    External Link
  </a>
  ```

### Dynamic Components

- [ ] **Dynamic component rendering uses allowlist**
  ```typescript
  const ALLOWED_COMPONENTS = {
    card: CardWidget,
    table: TableWidget,
    chart: ChartWidget,
  } as const;

  function renderWidget(type: string) {
    const Component = ALLOWED_COMPONENTS[type as keyof typeof ALLOWED_COMPONENTS];
    if (!Component) return <UnknownWidget />;
    return <Component />;
  }
  ```

## 3. Network & API Security

### Error Handling

- [ ] **API errors don't expose internal details** in production
  ```typescript
  catch (error) {
    // Log full error for debugging
    console.error('API error:', error);

    // Return sanitized error to user
    return { error: 'An error occurred. Please try again.' };
  }
  ```

- [ ] **Error boundaries prevent full app crashes** from revealing state
  ```tsx
  <ErrorBoundary fallback={<ErrorPage />}>
    <App />
  </ErrorBoundary>
  ```

### Credentials & URLs

- [ ] **Credentials not included in URLs** (use headers/body)
  ```typescript
  // Bad
  fetch(`/api/data?token=${apiToken}`);

  // Good
  fetch('/api/data', {
    headers: { Authorization: `Bearer ${apiToken}` },
  });
  ```

- [ ] **CORS headers verified** for API endpoints

- [ ] **Rate limiting applied** to user-facing endpoints

### Multi-Tenancy

- [ ] **Tenant isolation verified** for multi-tenant operations
- [ ] **tenant_id included** in all data-scoped API requests

## 4. React-Specific Security

- [ ] **No user input interpolated** into event handlers
  ```typescript
  // Bad - XSS risk
  <button onClick={() => eval(userCode)}>Run</button>

  // Good - controlled actions only
  <button onClick={() => handleAction(userChoice)}>Run</button>
  ```

- [ ] **State updates don't occur during render**
  ```typescript
  // Bad - causes infinite loops
  function Component() {
    const [count, setCount] = useState(0);
    setCount(count + 1); // Called during render!
    return <div>{count}</div>;
  }

  // Good - use useEffect
  function Component() {
    const [count, setCount] = useState(0);
    useEffect(() => {
      setCount(c => c + 1);
    }, []);
    return <div>{count}</div>;
  }
  ```

- [ ] **useEffect dependencies correctly specified**
- [ ] **Memoization used appropriately** (useMemo, useCallback)

## 5. Third-Party Dependencies

- [ ] **npm audit shows no high/critical vulnerabilities**
  ```bash
  npm audit --audit-level=high
  ```

- [ ] **External scripts loaded from trusted sources** or self-hosted

- [ ] **Dependency versions pinned** in package.json
  ```json
  {
    "dependencies": {
      "react": "18.2.0",  // Pinned, not "^18.2.0"
    }
  }
  ```

## Quick Reference: Common Issues from Epic 21

| Issue | Example | Fix |
|-------|---------|-----|
| Incomplete SENSITIVE_PATTERNS | Missing `jwt`, `bearer`, `session` | Add all credential-related terms |
| False positive key detection | `monkey` matches `key` | Use word boundaries `\bkey\b` |
| Value-embedded credentials | `password=secret` in strings | Add value pattern detection |
| Markdown XSS | Unsanitized user markdown | Use rehype-sanitize |
| External image leaks | Direct image URLs | Proxy or allowlist domains |
| setState during render | Direct state update in component body | Move to useEffect |

## Usage

1. **Before submitting PR**: Run through this checklist for any frontend changes
2. **During code review**: Reference specific sections when commenting
3. **For new components**: Use as a design guide

## Related Documentation

- [Protocol Compliance Checklist](../quality/protocol-compliance-checklist.md)
- [PR Template](../../.github/PULL_REQUEST_TEMPLATE.md)
