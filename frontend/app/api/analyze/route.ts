import { NextRequest } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const type = formData.get("type") as string;

    if (!type || !["image", "video", "document", "url"].includes(type)) {
      return Response.json(
        { error: "Invalid content type. Must be image, video, document, or url." },
        { status: 400 }
      );
    }

    if (type === "url") {
      const url = formData.get("url") as string;
      if (!url) {
        return Response.json({ error: "URL is required." }, { status: 400 });
      }
    } else {
      const file = formData.get("file") as File | null;
      if (!file) {
        return Response.json({ error: "File is required." }, { status: 400 });
      }
    }

    const backendUrl = process.env.BACKEND_URL;
    if (!backendUrl) {
      return Response.json(
        { error: "Backend service is not configured." },
        { status: 503 }
      );
    }

    const backendResponse = await fetch(`${backendUrl}/analyze`, {
      method: "POST",
      body: formData,
    });

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ error: "Unknown backend error" }));
      console.error("Backend Error Detail:", errorData);
      return Response.json(
        { error: `Backend integration error`, details: errorData },
        { status: backendResponse.status }
      );
    }

    const result = await backendResponse.json();
    return Response.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Internal server error";
    return Response.json({ error: message }, { status: 500 });
  }
}
