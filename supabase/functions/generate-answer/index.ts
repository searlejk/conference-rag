import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { corsHeaders, jsonResponse } from "../_shared/cors.ts";
import { requireAuth } from "../_shared/auth.ts";

// The browser sends the talk text, so a logged-in user could stuff anything
// in here. Caps keep one request from becoming an expensive GPT-4o call.
const MAX_QUESTION_CHARS = 1000;
const MAX_TALKS = 5;
const MAX_CONTEXT_CHARS = 100_000;

type Talk = { title: string; speaker: string; text: string };

const SYSTEM_PROMPT = `You answer questions about General Conference talks from The Church of Jesus Christ of Latter-day Saints.
Use ONLY the talk excerpts provided. If they don't answer the question, say so plainly instead of guessing.
Cite the talk title and speaker for each point you make, e.g. ("Title" — Speaker).`;

Deno.serve(async (req) => {
    if (req.method === "OPTIONS") {
        return new Response("ok", { headers: corsHeaders(req) });
    }
    if (req.method !== "POST") {
        return jsonResponse(req, { error: "Method not allowed" }, 405);
    }

    const auth = await requireAuth(req);
    if ("error" in auth) return auth.error;

    try {
        const { question, context_talks } = await req.json();

        if (!question || typeof question !== "string") {
            return jsonResponse(req, { error: "Missing or invalid 'question' field" }, 400);
        }
        if (question.length > MAX_QUESTION_CHARS) {
            return jsonResponse(req, { error: `Question must be under ${MAX_QUESTION_CHARS} characters` }, 400);
        }
        if (!Array.isArray(context_talks)) {
            return jsonResponse(req, { error: "Missing or invalid 'context_talks' field" }, 400);
        }

        // The page's readiness check sends no talks on every load — don't pay GPT-4o for that
        if (context_talks.length === 0) {
            return jsonResponse(req, { answer: "I couldn't find any talks related to that question." });
        }

        const talks: Talk[] = context_talks.slice(0, MAX_TALKS).map((t: Partial<Talk>) => ({
            title: String(t.title ?? "Untitled"),
            speaker: String(t.speaker ?? "Unknown"),
            text: String(t.text ?? ""),
        }));
        const contextText = talks
            .map((t) => `"${t.title}" — ${t.speaker}:\n${t.text}`)
            .join("\n\n---\n\n")
            .slice(0, MAX_CONTEXT_CHARS);

        const openaiKey = Deno.env.get("OPENAI_API_KEY");
        if (!openaiKey) {
            return jsonResponse(req, { error: "OPENAI_API_KEY not configured" }, 500);
        }

        // Augment: put the retrieved talks in the prompt; Generate: let GPT-4o answer from them
        const response = await fetch("https://api.openai.com/v1/chat/completions", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${openaiKey}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                model: "gpt-4o",
                messages: [
                    { role: "system", content: SYSTEM_PROMPT },
                    { role: "user", content: `Talk excerpts:\n\n${contextText}\n\n---\n\nQuestion: ${question}` },
                ],
                temperature: 0.3,
                max_tokens: 1024,
            }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            return jsonResponse(req, { error: err.error?.message || "OpenAI API error" }, 502);
        }

        const data = await response.json();
        return jsonResponse(req, { answer: data.choices[0].message.content });
    } catch (err) {
        return jsonResponse(req, { error: (err as Error).message }, 500);
    }
});
