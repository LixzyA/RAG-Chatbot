import { useState, useRef, useCallback } from "react";
import { apiFetch } from "@/lib/api";
import { Button } from "@/components/ui/button";

interface UploadResponse {
  status: number;
  num_chunk: number;
}


export default function Files() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [response, setResponse] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (selected: File | null) => {
    setFile(selected);
    setResponse(null);
    setError(null);
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  }, []);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    setResponse(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await apiFetch("/ingest/", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => null);
        throw new Error(
          errorData?.detail || `Upload failed with status ${res.status}`
        );
      }

      const data: UploadResponse = await res.json();
      setResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred");
    } finally {
      setUploading(false);
    }
  };

  const removeFile = () => {
    setFile(null);
    setResponse(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const fileExtension = file?.name.split(".").pop()?.toLowerCase();

  return (
    <div className="flex min-h-svh justify-center p-6 pt-20">
      <div className="flex w-full max-w-2xl flex-col gap-6">
        {/* Header */}
        <div className="flex flex-col gap-1">
          <h1 className="text-3xl font-semibold tracking-tight">
            Upload Files
          </h1>
          <p className="text-muted-foreground text-sm">
            Upload <code className="rounded bg-muted px-1.5 py-0.5 text-xs font-mono">.txt</code> or{" "}
            <code className="rounded bg-muted px-1.5 py-0.5 text-xs font-mono">.pdf</code> files to process and chunk for the knowledge base.
          </p>
        </div>

        {/* Drop zone */}
        <div
          id="file-dropzone"
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`
            group relative flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed
            px-6 py-14 transition-all duration-200
            ${
              dragActive
                ? "border-primary/60 bg-primary/5 scale-[1.01]"
                : "border-border hover:border-primary/40 hover:bg-muted/50"
            }
          `}
        >
          {/* Upload icon */}
          <div
            className={`
              flex size-12 items-center justify-center rounded-full transition-colors duration-200
              ${dragActive ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground group-hover:bg-primary/10 group-hover:text-primary"}
            `}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="22"
              height="22"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>

          <div className="flex flex-col items-center gap-1 text-center">
            <p className="text-sm font-medium">
              {dragActive ? "Drop your file here" : "Click or drag & drop to upload"}
            </p>
            <p className="text-muted-foreground text-xs">
              Supports TXT and PDF files
            </p>
          </div>

          <input
            id="file-input"
            ref={inputRef}
            type="file"
            accept=".txt,.pdf"
            className="hidden"
            onChange={(e) => handleFileChange(e.target.files?.[0] ?? null)}
          />
        </div>

        {/* Selected file chip */}
        {file && (
          <div className="flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-3 shadow-sm animate-in fade-in slide-in-from-top-2 duration-200">
            <div
              className={`
                flex size-9 shrink-0 items-center justify-center rounded-md text-xs font-bold uppercase
                ${fileExtension === "pdf" ? "bg-red-500/10 text-red-600 dark:text-red-400" : "bg-blue-500/10 text-blue-600 dark:text-blue-400"}
              `}
            >
              {fileExtension}
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">{file.name}</p>
              <p className="text-muted-foreground text-xs">
                {(file.size / 1024).toFixed(1)} KB
              </p>
            </div>
            <button
              id="remove-file-btn"
              onClick={removeFile}
              className="text-muted-foreground hover:text-foreground rounded-md p-1 transition-colors"
              aria-label="Remove file"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        )}

        {/* Upload button */}
        <Button
          id="upload-btn"
          onClick={handleUpload}
          disabled={!file || uploading}
          size="lg"
          className="w-full"
        >
          {uploading ? (
            <span className="flex items-center gap-2">
              <svg className="size-4 animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Uploading…
            </span>
          ) : (
            "Upload & Process"
          )}
        </Button>

        {/* Error message */}
        {error && (
          <div
            id="upload-error"
            className="flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive animate-in fade-in slide-in-from-top-2 duration-200"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mt-0.5 shrink-0">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
            <p>{error}</p>
          </div>
        )}

        {/* Response card */}
        {response && (
          <div
            id="upload-response"
            className="flex flex-col gap-4 rounded-xl border border-border bg-card p-5 shadow-sm animate-in fade-in slide-in-from-bottom-3 duration-300"
          >
            {/* Status header */}
            <div className="flex items-center gap-2">
              <div className="flex size-7 items-center justify-center rounded-full bg-green-500/10">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-green-600 dark:text-green-400">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              </div>
              <h2 className="text-base font-semibold">Upload Successful</h2>
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg bg-muted/50 px-4 py-3">
                <p className="text-muted-foreground text-xs font-medium uppercase tracking-wider">
                  Status
                </p>
                <p className="mt-1 text-sm font-semibold capitalize">
                  {response.status}
                </p>
              </div>
              <div className="rounded-lg bg-muted/50 px-4 py-3">
                <p className="text-muted-foreground text-xs font-medium uppercase tracking-wider">
                  Chunks
                </p>
                <p className="mt-1 text-sm font-semibold">
                  {response.num_chunk}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
