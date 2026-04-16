# Thomas Chang - Site Generation Workflow

This guide explains how to manage, preview, and build the **Thomas Chang** personal website within the multi-persona Hugo project.

## 1. Quick Start (Development)

To start the local development server for Thomas's site:

```bash
make dev-thomas_chang
```

*   **URL**: [http://localhost:1313](http://localhost:1313)
*   **Self-Healing**: This command now **automatically kills** any existing process on port 1313 before starting, so you won't run into "address already in use" errors.
*   **Features**: Auto-reload is enabled. Changes to content or themes will refresh the browser instantly.

## 2. Port Management

If you see an error like `bind: address already in use`, it means a previous session is still holding port 1313. Use this command to clear it:

```bash
kill -9 $(lsof -t -i:1313)
```

## 3. Building for Production

To generate the static HTML files for deployment:

```bash
make build-thomas_chang
```

*   **Output**: The built site will be in `sites/thomas_chang/public/`.

## 4. Managing Content

Thomas's content is located in `sites/thomas_chang/content/`.

| Content Type | Location |
| :--- | :--- |
| **Homepage Bio** | `content/_index.md` |
| **Blog Posts** | `content/blogs/` |
| **Book Notes** | `content/books/` |
| **Resume Info** | `content/resume/` |

### Adding a New Blog Post
1. Create a new `.md` file in `sites/thomas_chang/content/blogs/`.
2. Ensure you include the front matter:
   ```markdown
   ---
   title: "My New Post"
   date: 2024-04-12
   categories: ["ML"]
   ---
   ```

## 5. Styling & Visual Identity

*   **Global Layouts**: Modified in `themes/shared/layouts/`.
*   **Thomas's Palette**: To change colors specific to Thomas, edit `sites/thomas_chang/static/css/persona.css`.
