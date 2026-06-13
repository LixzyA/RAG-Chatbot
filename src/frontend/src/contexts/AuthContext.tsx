import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";

// ── Types ──────────────────────────────────────────────────────────

interface User {
  id: string;
  username: string;
  email: string;
}

interface AuthContextValue {
  user: User | null;
  token: string | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, email: string, password: string) => Promise<void>;
  logout: () => void;
}

// ── Context ────────────────────────────────────────────────────────

const AuthContext = createContext<AuthContextValue | null>(null);

// ── Helpers ────────────────────────────────────────────────────────

const API_BASE = "http://localhost:8000";

function getStoredToken(): string | null {
  return localStorage.getItem("auth_token");
}

function storeToken(token: string) {
  localStorage.setItem("auth_token", token);
}

function clearToken() {
  localStorage.removeItem("auth_token");
}

function getStoredUser(): User | null {
  const raw = localStorage.getItem("auth_user");
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function storeUser(user: User) {
  localStorage.setItem("auth_user", JSON.stringify(user));
}

function clearUser() {
  localStorage.removeItem("auth_user");
}

// ── Provider ───────────────────────────────────────────────────────

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(getStoredUser);
  const [token, setToken] = useState<string | null>(getStoredToken);
  const [loading, setLoading] = useState(true);

  // Verify stored token on mount
  useEffect(() => {
    const savedToken = getStoredToken();
    if (!savedToken) {
      setLoading(false);
      return;
    }

    fetch(`${API_BASE}/auth/me`, {
      headers: { Authorization: `Bearer ${savedToken}` },
    })
      .then((res) => {
        if (!res.ok) throw new Error("Token invalid");
        return res.json();
      })
      .then((userData) => {
        setUser(userData);
        setToken(savedToken);
        storeUser(userData);
      })
      .catch(() => {
        // Token expired or invalid — clear state
        clearToken();
        clearUser();
        setUser(null);
        setToken(null);
      })
      .finally(() => setLoading(false));
  }, []);

  // ── Login ────────────────────────────────────────────────────────

  const login = useCallback(async (username: string, password: string) => {
    const formData = new URLSearchParams();
    formData.append("username", username);
    formData.append("password", password);

    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => null);
      throw new Error(err?.detail || "Login failed");
    }

    const data = await res.json();
    storeToken(data.access_token);
    storeUser(data.user);
    setToken(data.access_token);
    setUser(data.user);
  }, []);

  // ── Register ─────────────────────────────────────────────────────

  const register = useCallback(
    async (username: string, email: string, password: string) => {
      const res = await fetch(`${API_BASE}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => null);
        throw new Error(err?.detail || "Registration failed");
      }

      // Registration succeeded — automatically log in
      await login(username, password);
    },
    [login]
  );

  // ── Logout ───────────────────────────────────────────────────────

  const logout = useCallback(() => {
    clearToken();
    clearUser();
    setToken(null);
    setUser(null);
  }, []);

  // ── Value ────────────────────────────────────────────────────────

  return (
    <AuthContext.Provider
      value={{ user, token, loading, login, register, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}

// ── Hook ───────────────────────────────────────────────────────────

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
