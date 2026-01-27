"use client";

import { useState } from "react";
import RequireRole from "@/components/RequireRole";
import apiClient from "@/lib/apiClient";
import { confirmAction } from "@/lib/confirm";

export default function IntegrityControlsPage() {
  const [loading, setLoading] = useState<null | "db-to-chroma" | "chroma-to-db">(
    null
  );
  const [message, setMessage] = useState<string | null>(null);

  /* ----------------------------------------
   * Actions
   * ------------------------------------- */
  const runDbToChroma = async () => {
    if (
      !confirmAction(
        "Re-embed DB content into Chroma?\n\nThis operation is SAFE and idempotent."
      )
    )
      return;

    try {
      setLoading("db-to-chroma");
      setMessage(null);

      await apiClient.post(
        "/api/v2/ingestion-admin/integrity/fix/db-to-chroma"
      );

      setMessage("✅ DB → Chroma fix completed successfully");
    } catch (err) {
      console.error("DB → Chroma fix failed:", err);
      setMessage("❌ DB → Chroma fix failed. Check backend logs.");
    } finally {
      setLoading(null);
    }
  };

  const runChromaToDb = async () => {
    if (
      !confirmAction(
        "⚠️ DANGER ZONE ⚠️\n\nThis operation MUTATES the database.\nProceed ONLY if you fully understand the impact."
      )
    )
      return;

    try {
      setLoading("chroma-to-db");
      setMessage(null);

      await apiClient.post(
        "/api/v2/ingestion-admin/integrity/fix/chroma-to-db"
      );

      setMessage("✅ Chroma → DB cleanup completed successfully");
    } catch (err) {
      console.error("Chroma → DB fix failed:", err);
      setMessage("❌ Chroma → DB cleanup failed. Check backend logs.");
    } finally {
      setLoading(null);
    }
  };

  /* ----------------------------------------
   * Render
   * ------------------------------------- */
  return (
    <RequireRole>
      <div className="space-y-6">
        <h1 className="text-xl font-semibold">Integrity Controls</h1>

        <div className="space-x-4">
          <button
            onClick={runDbToChroma}
            disabled={loading !== null}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {loading === "db-to-chroma"
              ? "Running DB → Chroma..."
              : "Fix DB → Chroma"}
          </button>

          <button
            onClick={runChromaToDb}
            disabled={loading !== null}
            className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
          >
            {loading === "chroma-to-db"
              ? "Running Chroma → DB..."
              : "Fix Chroma → DB"}
          </button>
        </div>

        {message && (
          <p className="text-sm text-gray-700 bg-gray-100 p-3 rounded">
            {message}
          </p>
        )}

        <ul className="list-disc pl-6 text-sm text-gray-600">
          <li>
            <strong>DB → Chroma</strong> is safe and idempotent
          </li>
          <li>
            <strong>Chroma → DB</strong> deletes vectors without DB rows
          </li>
          <li>All actions are manual, explicit, and auditable</li>
        </ul>
      </div>
    </RequireRole>
  );
}
