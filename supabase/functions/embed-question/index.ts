import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { corsHeaders, jsonResponse } from "../_shared/cors.ts";
import { requireAuth } from "../_shared/auth.ts";

// Every call costs OpenAI tokens, so cap what one request can send
const MAX_QUESTION_CHARS = 1000;

Deno.serve(async (req) => {
    // CORS preflight: the browser asks permission before the real POST
    if (req.method === "OPTIONS") {
        return new Response("ok", { headers: corsHeaders(req) });
    }
    if (req.method !== "POST") {
        return jsonResponse(req, { error: "Method not allowed" }, 405);
    }

    // 1. Who is calling? (logged-in users only)
    const auth = await requireAuth(req);
    if ("error" in auth) return auth.error;

    try {
        // 2. Validate input
        const { question } = await req.json();
        if (!question || typeof question !== "string") {
            return jsonResponse(req, { error: "Missing or invalid 'question' field" }, 400);
        }
        if (question.length > MAX_QUESTION_CHARS) {
            return jsonResponse(req, { error: `Question must be under ${MAX_QUESTION_CHARS} characters` }, 400);
        }

        // 3. The OpenAI key only exists here, on the server (set with `supabase secrets set`)
        const openaiKey = Deno.env.get("OPENAI_API_KEY");
        if (!openaiKey) {
            return jsonResponse(req, { error: "OPENAI_API_KEY not configured" }, 500);
        }

        // 4. Embed with the SAME model used for the stored sentences,
        //    otherwise the vectors live in different "spaces" and can't be compared
        const response = await fetch("https://api.openai.com/v1/embeddings", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${openaiKey}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ model: "text-embedding-3-small", input: question }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            return jsonResponse(req, { error: err.error?.message || "OpenAI API error" }, 502);
        }

        const data = await response.json();
        return jsonResponse(req, { embedding: data.data[0].embedding });
    } catch (err) {
        return jsonResponse(req, { error: (err as Error).message }, 500);
    }
});
