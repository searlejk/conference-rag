import { createClient } from "jsr:@supabase/supabase-js@2";
import { jsonResponse } from "./cors.ts";

/**
 * Verify the caller is a logged-in user before we spend OpenAI money.
 *
 * Functions are deployed with --no-verify-jwt because the Edge gateway can't
 * yet verify Supabase Auth's newer ES256 session tokens. Instead we ask the
 * Auth service itself (auth.getUser), which validates the token's signature
 * and expiry and returns the user it belongs to.
 *
 * Note: the anon key alone is also a valid JWT, but it has no user inside it,
 * so getUser() fails for it — anonymous visitors are rejected here.
 */
export async function requireAuth(req: Request): Promise<
    { user: { id: string; email?: string } } | { error: Response }
> {
    const authHeader = req.headers.get("Authorization");
    if (!authHeader?.startsWith("Bearer ")) {
        return { error: jsonResponse(req, { error: "Missing Authorization header" }, 401) };
    }

    const supabase = createClient(
        Deno.env.get("SUPABASE_URL") ?? "",
        Deno.env.get("SUPABASE_ANON_KEY") ?? "",
        { global: { headers: { Authorization: authHeader } } },
    );

    const { data: { user }, error } = await supabase.auth.getUser();
    if (error || !user) {
        return { error: jsonResponse(req, { error: "Invalid or expired token" }, 401) };
    }

    return { user };
}
