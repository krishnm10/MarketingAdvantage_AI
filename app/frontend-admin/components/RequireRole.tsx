"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getAuthToken } from "@/lib/authToken";

export default function RequireRole({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const token = getAuthToken();

    if (!token) {
      router.replace("/auth/login");
      return;
    }

    setReady(true);
  }, [router]);

  if (!ready) return null;

  return <>{children}</>;
}
