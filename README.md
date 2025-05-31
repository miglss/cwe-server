# CWE Analyzer API (FastAPI Service)

This FastAPI service provides a `/classify` endpoint that accepts a `POST` request containing a ZIP archive of C/C++ source files. When invoked (e.g., by the VS Code extension), the service will:

1. Unzip the uploaded archive.
2. Run a preprocessing and inference pipeline using a pre-trained CodeBERT model.
3. Return a JSON response with predicted CWE class labels.

Once deployed on your server, point the VS Code extension’s `apiUrl` to this endpoint, and it will automatically send source-code ZIPs to `/classify` and display the returned CWE labels.
