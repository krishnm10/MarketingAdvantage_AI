"use client";

import RequireRole from "@/components/RequireRole";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <RequireRole>{children}</RequireRole>;
}
