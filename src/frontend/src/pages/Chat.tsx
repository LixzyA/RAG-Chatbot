import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";

// ── Types ──────────────────────────────────────────────────────────

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface ChatSummary {
  chat_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

// ── Helpers ────────────────────────────────────────────────────────

function generateUUID(): string {
  return crypto.randomUUID();
}

const API_BASE = "http://localhost:8000";

// ── Component ──────────────────────────────────────────────────────

export default function Chat() {
  const [chatId, setChatId] = useState<string>(generateUUID);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [chatList, setChatList] = useState<ChatSummary[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const initialLoadDone = useRef(false);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Focus input on mount and after each response
  useEffect(() => {
    if (!streaming) inputRef.current?.focus();
  }, [streaming]);

  const { token } = useAuth();

  // Helper to build headers with auth
  const authHeaders = useCallback(
    (extra?: Record<string, string>): Record<string, string> => ({
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...extra,
    }),
    [token]
  );

  // ── Load messages for a given chatId from backend ───────────────

  const loadChatHistory = useCallback(
    async (cid: string) => {
      if (!token) return false;
      try {
        const res = await fetch(`${API_BASE}/chat/history/${cid}`, {
          headers: authHeaders(),
        });
        if (!res.ok) {
          if (res.status === 404) {
            setMessages([]);
            return true; // 404 is "ok" — just empty
          }
          console.error(`loadChatHistory ${cid.slice(0, 8)}: ${res.status} ${res.statusText}`);
          return false;
        }
        const data = await res.json();
        setMessages(data.messages ?? []);
        return true;
      } catch (err) {
        console.error(`loadChatHistory ${cid.slice(0, 8)}:`, err);
        return false;
      }
    },
    [token, authHeaders]
  );

  // ── Fetch chat list (all user's histories) ──────────────────────

  const fetchChatList = useCallback(async () => {
    if (!token) return [];
    try {
      const res = await fetch(`${API_BASE}/chat/histories`, {
        headers: authHeaders(),
      });
      if (res.ok) {
        return (await res.json()) as ChatSummary[];
      }
      console.error(`fetchChatList: ${res.status} ${res.statusText}`);
    } catch (err) {
      console.error("fetchChatList:", err);
    }
    return [];
  }, [token, authHeaders]);

  // ── On mount: load chat list, then open most recent chat ────────

  useEffect(() => {
    if (!token || initialLoadDone.current) return;
    let cancelled = false;

    (async () => {
      const list = await fetchChatList();
      if (cancelled) return;
      setChatList(list);

      if (list.length > 0) {
        // Open the most recent chat
        const cid = list[0].chat_id;
        setChatId(cid);
        await loadChatHistory(cid);
      }
      initialLoadDone.current = true;
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  // Cleanup abort controller on unmount
  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  // ── Switch to a different chat ──────────────────────────────────

  const switchChat = useCallback(
    async (newChatId: string) => {
      abortRef.current?.abort();
      setChatId(newChatId);
      setMessages([]);
      setInput("");
      setError(null);
      setStreaming(false);
      setSidebarOpen(false);
      await loadChatHistory(newChatId);
    },
    [loadChatHistory]
  );

  // ── Submit handler ───────────────────────────────────────────────

  const handleSubmit = useCallback(
    async (e?: React.FormEvent) => {
      e?.preventDefault();
      const trimmed = input.trim();
      if (!trimmed || streaming) return;

      setError(null);
      setInput("");

      // Add user message
      const userMsg: Message = {
        id: generateUUID(),
        role: "user",
        content: trimmed,
      };
      setMessages((prev) => [...prev, userMsg]);

      // Prepare assistant placeholder
      const assistantId = generateUUID();
      const assistantMsg: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setStreaming(true);

      try {
        const controller = new AbortController();
        abortRef.current = controller;

        const res = await fetch(`${API_BASE}/chat/v2`, {
          method: "POST",
          headers: authHeaders(),
          body: JSON.stringify({ prompt: trimmed, top_k: 10, chat_id: chatId }),
          signal: controller.signal,
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => null);
          throw new Error(
            errData?.detail || `Request failed with status ${res.status}`
          );
        }

        const reader = res.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) throw new Error("No response body");

        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Parse SSE lines
          const lines = buffer.split("\n");
          // Keep the last potentially incomplete line in the buffer
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (line.startsWith("event: done")) {
              break;
            }
            if (line.startsWith("data: ")) {
              const data = line.slice(6);
              if (data === "[DONE]") break;

              // Append token to assistant message
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, content: msg.content + data }
                    : msg
                )
              );
            }
          }
        }
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setError(
          err instanceof Error ? err.message : "An unknown error occurred"
        );
        // Remove the empty assistant placeholder on error
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last?.id === assistantId && last.content === "") {
            return prev.slice(0, -1);
          }
          return prev;
        });
      } finally {
        setStreaming(false);
        abortRef.current = null;
        // Refresh the sidebar list (fire-and-forget, never touches messages)
        fetchChatList().then((list) => {
          if (list) setChatList(list);
        });
      }
    },
    [input, streaming, chatId, authHeaders, fetchChatList]
  );

  // ── New chat handler ─────────────────────────────────────────────

  const handleNewChat = () => {
    abortRef.current?.abort();
    const newId = generateUUID();
    setChatId(newId);
    setMessages([]);
    setInput("");
    setError(null);
    setStreaming(false);
    setSidebarOpen(false);
    // Focus after state update
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  // ── Render ───────────────────────────────────────────────────────

  return (
    <div className="flex h-[calc(100svh-3.5rem)] overflow-hidden">
      {/* ── History sidebar ─────────────────────────────────────── */}
      {/* Mobile overlay (only when open) — hidden on desktop */}
      {sidebarOpen && (
        <aside className="fixed left-0 top-14 z-20 flex h-[calc(100svh-3.5rem)] w-72 flex-col border-r border-border bg-background sm:hidden">
          <div className="flex items-center justify-between border-b border-border px-4 py-3">
            <h2 className="text-sm font-semibold">Chat History</h2>
            <button
              onClick={() => setSidebarOpen(false)}
              className="rounded-md p-1 text-muted-foreground hover:bg-muted"
              aria-label="Close sidebar"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
          <nav className="flex-1 overflow-y-auto p-2">
            {chatList.length === 0 ? (
              <p className="px-2 py-8 text-center text-xs text-muted-foreground">No past chats yet</p>
            ) : (
              <ul className="flex flex-col gap-1">
                {chatList.map((chat) => (
                  <li key={chat.chat_id}>
                    <button
                      onClick={() => switchChat(chat.chat_id)}
                      className={cn(
                        "w-full rounded-md px-3 py-2 text-left text-xs transition-colors",
                        chat.chat_id === chatId
                          ? "bg-primary/10 text-primary font-medium"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground"
                      )}
                    >
                      <span className="line-clamp-1">{chat.title}</span>
                      <span className="mt-0.5 block text-[10px] opacity-60">
                        {chat.message_count} message{chat.message_count !== 1 ? "s" : ""}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </nav>
        </aside>
      )}

      {/* Desktop sidebar — responds to burger toggle */}
      <aside className={cn(
        "hidden flex-col border-r border-border bg-background w-72 shrink-0",
        sidebarOpen ? "sm:flex" : "sm:hidden"
      )}>
        <nav className="flex-1 overflow-y-auto p-2">
          {chatList.length === 0 ? (
            <p className="px-2 py-8 text-center text-xs text-muted-foreground">No past chats yet</p>
          ) : (
            <ul className="flex flex-col gap-1">
              {chatList.map((chat) => (
                <li key={chat.chat_id}>
                  <button
                    onClick={() => switchChat(chat.chat_id)}
                    className={cn(
                      "w-full rounded-md px-3 py-2 text-left text-xs transition-colors",
                      chat.chat_id === chatId
                        ? "bg-primary/10 text-primary font-medium"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                    )}
                  >
                    <span className="line-clamp-1">{chat.title}</span>
                    <span className="mt-0.5 block text-[10px] opacity-60">
                      {chat.message_count} message{chat.message_count !== 1 ? "s" : ""}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </nav>
      </aside>

      {/* ── Overlay for mobile sidebar ── */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-10 bg-black/30 sm:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* ── Main chat area — scrolls independently ──────────────── */}
      <div className="flex flex-1 flex-col overflow-y-auto">
        {/* Top bar */}
        <header className="sticky top-14 z-10 flex items-center justify-between border-b border-border bg-background/80 px-4 py-3 backdrop-blur-md sm:px-6">
          <div className="flex items-center gap-3">
            {/* History toggle */}
            <button
              onClick={() => setSidebarOpen((prev) => !prev)}
              className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              aria-label="Toggle chat history"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="3" y1="6" x2="21" y2="6" />
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="18" x2="21" y2="18" />
              </svg>
            </button>

            <h1 className="text-lg font-semibold tracking-tight">Chat</h1>
          </div>

          {/* New Chat button with confirmation dialog */}
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button id="new-chat-btn" variant="outline" size="sm" disabled={messages.length === 0 && !streaming}>
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  data-icon="inline-start"
                  className="mr-1"
                >
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
                New Chat
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Start a new chat?</AlertDialogTitle>
                <AlertDialogDescription>
                  This will clear the current chat history. This action cannot be
                  undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction id="confirm-new-chat-btn" onClick={handleNewChat}>
                  Continue
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </header>

        {/* Messages area */}
        <main className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-4">
          {messages.length === 0 && !streaming && (
            <div className="flex flex-1 flex-col items-center justify-center gap-3 py-24 text-center">
              <div className="flex size-14 items-center justify-center rounded-full bg-muted">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="text-muted-foreground"
                >
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
              </div>
              <div>
                <p className="font-medium">Start a conversation</p>
                <p className="text-muted-foreground text-sm">
                  Ask a question about your uploaded documents.
                </p>
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <Card
                size="sm"
                className={`max-w-[85%] sm:max-w-[75%] ${msg.role === "user"
                    ? "bg-primary text-primary-foreground ring-primary/20"
                    : "bg-card ring-foreground/10"
                  }`}
              >
                <CardContent>
                  {/* Sender label */}
                  <p
                    className={`mb-1.5 text-[11px] font-semibold uppercase tracking-wider ${msg.role === "user"
                        ? "text-primary-foreground/70"
                        : "text-muted-foreground"
                      }`}
                  >
                    {msg.role === "user" ? "You" : "Assistant"}
                  </p>

                  {/* Message content */}
                  {msg.content ? (
                    msg.role === "user" ? (
                      <p className="whitespace-pre-wrap text-sm leading-relaxed">
                        {msg.content}
                      </p>
                    ) : (
                      <div className="prose-chat text-sm leading-relaxed">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            h1: ({ children }) => (
                              <h1 className="mb-3 mt-4 text-lg font-bold first:mt-0">{children}</h1>
                            ),
                            h2: ({ children }) => (
                              <h2 className="mb-2 mt-3 text-base font-semibold first:mt-0">{children}</h2>
                            ),
                            h3: ({ children }) => (
                              <h3 className="mb-2 mt-3 text-sm font-semibold first:mt-0">{children}</h3>
                            ),
                            p: ({ children }) => (
                              <p className="mb-2 last:mb-0">{children}</p>
                            ),
                            ul: ({ children }) => (
                              <ul className="mb-2 ml-4 list-disc space-y-1 last:mb-0">{children}</ul>
                            ),
                            ol: ({ children }) => (
                              <ol className="mb-2 ml-4 list-decimal space-y-1 last:mb-0">{children}</ol>
                            ),
                            li: ({ children }) => (
                              <li className="pl-1">{children}</li>
                            ),
                            a: ({ href, children }) => (
                              <a
                                href={href}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-primary underline underline-offset-2 hover:text-primary/80"
                              >
                                {children}
                              </a>
                            ),
                            blockquote: ({ children }) => (
                              <blockquote className="my-2 border-l-2 border-muted-foreground/30 pl-3 italic text-muted-foreground">
                                {children}
                              </blockquote>
                            ),
                            code: ({ className, children, ...props }) => {
                              const isBlock = className?.includes("language-");
                              if (isBlock) {
                                return (
                                  <pre className="my-2 overflow-x-auto rounded-md bg-muted/60 p-3">
                                    <code className="text-xs" {...props}>
                                      {children}
                                    </code>
                                  </pre>
                                );
                              }
                              return (
                                <code
                                  className="rounded bg-muted/60 px-1.5 py-0.5 text-xs font-mono"
                                  {...props}
                                >
                                  {children}
                                </code>
                              );
                            },
                            pre: ({ children }) => <>{children}</>,
                            table: ({ children }) => (
                              <div className="my-2 overflow-x-auto rounded-md border border-border">
                                <table className="w-full text-xs">{children}</table>
                              </div>
                            ),
                            thead: ({ children }) => (
                              <thead className="bg-muted/40">{children}</thead>
                            ),
                            th: ({ children }) => (
                              <th className="px-3 py-1.5 text-left font-semibold">{children}</th>
                            ),
                            td: ({ children }) => (
                              <td className="border-t border-border px-3 py-1.5">{children}</td>
                            ),
                            hr: () => <hr className="my-3 border-border" />,
                            strong: ({ children }) => (
                              <strong className="font-semibold">{children}</strong>
                            ),
                          }}
                        >
                          {msg.content}
                        </ReactMarkdown>
                      </div>
                    )
                  ) : (
                    // Streaming placeholder dots
                    <div className="flex items-center gap-1 py-1">
                      <span className="bg-muted-foreground/40 size-1.5 animate-pulse rounded-full" />
                      <span className="bg-muted-foreground/40 size-1.5 animate-pulse rounded-full [animation-delay:150ms]" />
                      <span className="bg-muted-foreground/40 size-1.5 animate-pulse rounded-full [animation-delay:300ms]" />
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          ))}

          {/* Error message */}
          {error && (
            <div
              id="chat-error"
              className="flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive animate-in fade-in slide-in-from-top-2 duration-200"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="mt-0.5 shrink-0"
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <p>{error}</p>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input bar */}
      <footer className="sticky bottom-0 border-t border-border bg-background/80 px-4 py-3 backdrop-blur-md sm:px-6">
        <form
          onSubmit={handleSubmit}
          className="mx-auto flex w-full max-w-3xl items-center gap-2"
        >
          <Input
            id="chat-input"
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message…"
            disabled={streaming}
            autoComplete="off"
            className="flex-1"
          />
          <Button
            id="send-btn"
            type="submit"
            disabled={!input.trim() || streaming}
          >
            {streaming ? (
              <svg
                className="size-4 animate-spin"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
            ) : (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            )}
          </Button>
        </form>
      </footer>
      {/* closes main chat area div */}
    </div>
    {/* closes outer flex div */}
    </div>
  );
}
