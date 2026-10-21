// Browsers only let these origins call our Edge Functions.
// CORS is not authentication (curl ignores it) — it stops OTHER websites
// from using a visitor's browser to call these functions.
const ALLOWED_ORIGINS = [
    "https://searlejk.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
];

export function corsHeaders(req: Request): Record<string, string> {
    const origin = req.headers.get("Origin") ?? "";
    return {
        "Access-Control-Allow-Origin": ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0],
        "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Vary": "Origin",
    };
}

export function jsonResponse(req: Request, body: unknown, status = 200): Response {
    return new Response(JSON.stringify(body), {
        status,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
    });
}
