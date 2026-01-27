// frontend-admin/lib/apiClient.ts

import axios from "axios";
import { getAuthToken, clearAuthToken } from "./authToken";

const apiClient = axios.create({
  baseURL:
    process.env.NEXT_PUBLIC_BACKEND_API_URL ||
    "http://127.0.0.1:8000",
  timeout: 30000,
});

// --------------------------------------------------
// REQUEST INTERCEPTOR — attach backend JWT
// --------------------------------------------------
apiClient.interceptors.request.use(
  (config) => {
    const token = getAuthToken();

    if (token) {
      config.headers = config.headers ?? {};
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error) => Promise.reject(error)
);

// --------------------------------------------------
// RESPONSE INTERCEPTOR — global auth failure handling
// --------------------------------------------------
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error?.response?.status === 401) {
      // Backend rejected token → force logout
      clearAuthToken();

      if (typeof window !== "undefined") {
        window.location.href = "/auth/login";
      }
    }

    return Promise.reject(error);
  }
);

export default apiClient;
