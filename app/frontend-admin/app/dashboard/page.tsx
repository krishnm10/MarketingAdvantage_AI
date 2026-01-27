"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";
import RequireRole from "@/components/RequireRole";
import IngestionFeed from "./ingestion-feed";
import apiClient from "@/lib/apiClient";

/* ----------------------------------------
 * Tabs (typed, stable)
 * ------------------------------------- */
const TABS = {
  FILES: "Ingestion Files",
  UPLOAD: "Upload",
  SYNC: "Sync",
  HEALTH: "Health",
  FEED: "Live Feed",
} as const;

type Tab = typeof TABS[keyof typeof TABS];

export default function UnifiedDashboard() {
  const [activeTab, setActiveTab] = useState<Tab>(TABS.FILES);
  const [files, setFiles] = useState<any[]>([]);
  const [health, setHealth] = useState<any>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadMsg, setUploadMsg] = useState("");
  const [syncMsg, setSyncMsg] = useState("");
  const [loading, setLoading] = useState(false);

  /* ----------------------------------------
   * Fetchers
   * ------------------------------------- */
  const fetchFiles = async () => {
    try {
      const res = await apiClient.get("/api/v2/ingestion-admin/files");
      setFiles(res.data ?? []);
    } catch {
      setFiles([]);
    }
  };

  const fetchHealth = async () => {
    try {
      const res = await apiClient.get("/api/v2/ingestion/health");
      setHealth(res.data);
    } catch {
      setHealth({ status: "offline" });
    }
  };

  /* ----------------------------------------
   * Upload
   * ------------------------------------- */
  const handleUpload = async () => {
    if (!uploadFile) return;

    setLoading(true);
    setUploadMsg("");

    try {
      const formData = new FormData();
      formData.append("file", uploadFile);

      await apiClient.post("/api/v2/ingestion/upload", formData);
      setUploadMsg("✅ Upload successful");
      fetchFiles();
    } catch (err: any) {
      setUploadMsg(`❌ Upload failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  /* ----------------------------------------
   * Sync
   * ------------------------------------- */
  const handleSync = async (endpoint: string, label: string) => {
    try {
      const res = await apiClient.post(
        `/api/v2/ingestion-admin/sync/${endpoint}`
      );
      setSyncMsg(`✅ ${label}: ${res.data?.message ?? "done"}`);
    } catch (err: any) {
      setSyncMsg(`❌ ${label} failed: ${err.message}`);
    }
  };

  /* ----------------------------------------
   * Init + Polling
   * ------------------------------------- */
  useEffect(() => {
    fetchFiles();
    fetchHealth();

    const id = setInterval(() => {
      fetchFiles();
      fetchHealth();
    }, 15000);

    return () => clearInterval(id);
  }, []);

  /* ----------------------------------------
   * Render
   * ------------------------------------- */
  return (
    <RequireRole>
      <div className="p-6">
        <h1 className="text-2xl font-bold mb-6">
          Ingestion Dashboard
        </h1>

        {/* Tabs */}
        <div className="flex space-x-4 border-b mb-6">
          {Object.values(TABS).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={clsx(
                "pb-2 text-lg font-medium transition-colors",
                activeTab === tab
                  ? "border-b-2 border-blue-600 text-blue-600"
                  : "text-gray-600 hover:text-blue-600"
              )}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Ingestion Files */}
        {activeTab === TABS.FILES && (
          <section>
            {files.length === 0 ? (
              <p className="text-gray-500">No files available.</p>
            ) : (
              <table className="min-w-full text-sm border rounded">
                <thead className="bg-gray-100">
                  <tr>
                    <th className="p-2">File Name</th>
                    <th className="p-2">Status</th>
                    <th className="p-2">Uploaded</th>
                  </tr>
                </thead>
                <tbody>
                  {files.map((f) => (
                    <tr key={f.id} className="border-b hover:bg-gray-50">
                      <td className="p-2">{f.file_name}</td>
                      <td className="p-2">{f.status}</td>
                      <td className="p-2">
                        {new Date(f.created_at).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        )}

        {/* Upload */}
        {activeTab === TABS.UPLOAD && (
          <section>
            <input
              type="file"
              onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
              className="border p-2 rounded mb-3"
            />
            <button
              onClick={handleUpload}
              disabled={!uploadFile || loading}
              className="bg-green-600 text-white px-4 py-2 rounded"
            >
              {loading ? "Uploading..." : "Upload"}
            </button>
            {uploadMsg && <p className="mt-3 text-sm">{uploadMsg}</p>}
          </section>
        )}

        {/* Sync */}
        {activeTab === TABS.SYNC && (
          <section className="space-y-3">
            <button
              onClick={() => handleSync("orphans", "Delete Orphans")}
              className="bg-red-500 text-white px-4 py-2 rounded"
            >
              Delete Orphans
            </button>
            <button
              onClick={() =>
                handleSync("fix/chroma-to-db", "Fix Chroma → DB")
              }
              className="bg-blue-500 text-white px-4 py-2 rounded"
            >
              Fix Chroma → DB
            </button>
            <button
              onClick={() =>
                handleSync("fix/db-to-chroma", "Fix DB → Chroma")
              }
              className="bg-green-600 text-white px-4 py-2 rounded"
            >
              Fix DB → Chroma
            </button>
            {syncMsg && <p className="text-sm">{syncMsg}</p>}
          </section>
        )}

        {/* Health */}
        {activeTab === TABS.HEALTH && (
          <pre className="bg-white p-4 rounded shadow text-xs">
            {JSON.stringify(health, null, 2)}
          </pre>
        )}

        {/* Live Feed */}
        {activeTab === TABS.FEED && <IngestionFeed />}
      </div>
    </RequireRole>
  );
}
