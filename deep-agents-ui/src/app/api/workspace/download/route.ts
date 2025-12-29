import { NextResponse } from "next/server";
import path from "path";
import { promises as fs } from "fs";
import { createReadStream } from "fs";
import { Readable } from "stream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const workspaceDir = path.resolve(process.cwd(), "..", "workspace");

async function ensureWorkspaceDir() {
  await fs.mkdir(workspaceDir, { recursive: true });
}

function isWithinWorkspace(targetPath: string) {
  const relative = path.relative(workspaceDir, targetPath);
  return (
    relative === "" ||
    (!relative.startsWith("..") && !path.isAbsolute(relative))
  );
}

function normalizeRelativePath(relativePath: string) {
  return relativePath.replace(/\\/g, "/").replace(/^\/+/, "");
}

function getDownloadName(fileName: string) {
  const fallback = fileName
    .replace(/[^\x20-\x7E]+/g, "_")
    .replace(/["\\]/g, "")
    .trim();
  const safeFallback = fallback || "download";
  const encoded = encodeURIComponent(fileName);
  return { safeFallback, encoded };
}

export async function GET(request: Request) {
  await ensureWorkspaceDir();
  const { searchParams } = new URL(request.url);
  const requestedPath = searchParams.get("path");

  if (!requestedPath) {
    return NextResponse.json(
      { error: "A file path is required." },
      { status: 400 }
    );
  }

  const relativePath = normalizeRelativePath(requestedPath);
  if (!relativePath) {
    return NextResponse.json(
      { error: "A file path is required." },
      { status: 400 }
    );
  }

  const filePath = path.resolve(workspaceDir, relativePath);
  if (!isWithinWorkspace(filePath)) {
    return NextResponse.json({ error: "Invalid file path." }, { status: 400 });
  }

  const entry = await fs.lstat(filePath).catch(() => null);
  if (!entry) {
    return NextResponse.json({ error: "File not found." }, { status: 404 });
  }

  let resolvedPath = filePath;
  if (entry.isSymbolicLink()) {
    const realPath = await fs.realpath(filePath).catch(() => null);
    if (!realPath || !isWithinWorkspace(realPath)) {
      return NextResponse.json({ error: "Invalid file path." }, { status: 400 });
    }
    resolvedPath = realPath;
  }

  const stats = entry.isSymbolicLink()
    ? await fs.stat(resolvedPath).catch(() => null)
    : entry;

  if (!stats || !stats.isFile()) {
    return NextResponse.json({ error: "File not found." }, { status: 404 });
  }

  const fileName = path.basename(filePath);
  const { safeFallback, encoded } = getDownloadName(fileName);
  const stream = Readable.toWeb(createReadStream(resolvedPath));

  return new Response(stream, {
    headers: {
      "Content-Type": "application/octet-stream",
      "Content-Length": stats.size.toString(),
      "Content-Disposition": `attachment; filename="${safeFallback}"; filename*=UTF-8''${encoded}`,
    },
  });
}
